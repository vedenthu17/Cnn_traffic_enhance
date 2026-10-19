"""Training loop.

AdamW + linear warm-up + cosine decay, label smoothing, mixed precision, AugMix/JSD views,
early stopping on validation-C accuracy, CSV log (optional Weights & Biases), checkpoint every
epoch and exact resume (optimizer, scheduler, grad scaler and all random generators are restored).

RoadLens v3 options (all off unless a run config sets them, so older runs are unaffected):
    train.ema              decay of an exponential moving average of the weights (e.g. 0.999). The
                           average is what is validated, selected and saved as best.pt.
    train.distill          {teacher: <run>, weight: w, temperature: T}: add w * T^2 * KL(teacher || student)
                           on every view of every batch (the teacher sees exactly the same augmented
                           images: "consistent" distillation, Beyer et al. 2022).
    train.channels_last    NHWC memory layout for convolutions (faster tensor-core kernels).
    train.cudnn_benchmark  let cuDNN pick the fastest kernels (gives up bitwise reproducibility).
    train.fused_optimizer  the fused CUDA AdamW kernel (one launch for all parameters).
    train.compile          torch.compile the training forward pass.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .augment import TrainAugment, jsd_loss
from .data.loading import as_float, load_split, load_stack_data, load_val_c_split, u8_to_device
from .utils import get_logger, seed_everything

log = get_logger()
LOG_FIELDS = ["epoch", "lr", "loss", "ce", "jsd", "kd", "train_acc", "val_acc", "val_c_acc", "epoch_s", "img_per_s"]


def param_groups(model, weight_decay: float) -> list[dict]:
    """No weight decay on biases, normalisation parameters and position / class tokens."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "pos_embed" in name or "cls_token" in name else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def accuracy(model, x_u8: torch.Tensor, y: torch.Tensor, batch_size: int, amp: bool) -> float:
    model.eval()
    correct = 0
    for s in range(0, len(x_u8), batch_size):
        with torch.autocast(x_u8.device.type, enabled=amp):
            pred = model(as_float(x_u8[s:s + batch_size])).argmax(dim=1)
        correct += int((pred == y[s:s + batch_size]).sum())
    return correct / len(x_u8)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """T^2 * KL(softmax(teacher / T) || softmax(student / T)), averaged over the batch (Hinton et al. 2015)."""
    t = float(temperature)
    return F.kl_div(F.log_softmax(student_logits.float() / t, dim=1), F.log_softmax(teacher_logits.float() / t, dim=1),
                    log_target=True, reduction="batchmean") * t * t


def training_loss(model, views: list[torch.Tensor], y: torch.Tensor, jsd: bool, smoothing: float,
                  jsd_weight: float, amp: bool, ce_all_views: bool = False, teacher=None,
                  kd_weight: float = 0.0, kd_temperature: float = 1.0, stats: dict | None = None):
    """Forward the views of one batch and return (loss, cross-entropy, JSD term, logits of the first view).

    jsd=False: cross-entropy on the single view. jsd=True: cross-entropy on the clean view plus
    jsd_weight * Jensen-Shannon divergence between the clean and the two augmented views.
    ce_all_views: with jsd, average the cross-entropy over all views instead of the clean one only.
    teacher: a frozen model; adds kd_weight * kd_loss on all views (reported as stats["kd"]).
    """
    x = torch.cat(views) if len(views) > 1 else views[0]
    with torch.autocast(y.device.type, enabled=amp):
        logits = model(x)
    kd = torch.zeros((), device=y.device)
    if teacher is not None and kd_weight > 0:
        with torch.no_grad(), torch.autocast(y.device.type, enabled=amp):
            teacher_logits = teacher(x)
        kd = kd_loss(logits, teacher_logits, kd_temperature)
    if stats is not None:
        stats["kd"] = kd.detach()
    if jsd:
        parts = list(logits.chunk(len(views)))
        if ce_all_views:
            ce = sum(F.cross_entropy(p.float(), y, label_smoothing=smoothing) for p in parts) / len(parts)
        else:
            ce = F.cross_entropy(parts[0].float(), y, label_smoothing=smoothing)
        div = jsd_loss(parts)
        return ce + jsd_weight * div + kd_weight * kd, ce, div, parts[0]
    ce = F.cross_entropy(logits.float(), y, label_smoothing=smoothing)
    return ce + kd_weight * kd, ce, torch.zeros((), device=y.device), logits


class ModelEMA:
    """Exponential moving average of a model's weights and buffers (mean-teacher / Polyak averaging).

    decay_t = min(decay, (1 + t) / (10 + t)), so that the average is not dominated by the random
    initial head during the first steps (the usual warm-up of timm / YOLO EMAs).
    """

    def __init__(self, model, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = float(decay), 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(current[k].detach().to(v.dtype), alpha=1 - d)
            else:
                v.copy_(current[k])


def _save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state(device) -> dict:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict, device) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_run(cfg: dict, seed: int, run_dir: Path, model, device, use_wandb: bool = False,
              scratch_dir: Path | None = None, fingerprint: str | None = None, teacher=None) -> dict:
    """Train `model` as described by `cfg` and leave best.pt, log.csv and done.json in `run_dir`.

    Returns the contents of done.json. Safe to call again: a finished run returns immediately, an
    interrupted one continues from its last completed epoch.

    scratch_dir: where the per-epoch checkpoints (last.pt and the in-progress best.pt) are kept;
        defaults to run_dir. best.pt is copied to run_dir once, when the run finishes.
    fingerprint: identifies the configuration; a resume checkpoint with another fingerprint is ignored.
    teacher: the frozen teacher model of a train.distill run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    work = Path(scratch_dir) if scratch_dir else run_dir
    work.mkdir(parents=True, exist_ok=True)
    done_path, log_path, final_best = run_dir / "done.json", run_dir / "log.csv", run_dir / "best.pt"
    last_path, best_path = work / "last.pt", work / "best.pt"
    if done_path.exists():
        return json.loads(done_path.read_text())

    t = cfg["train"]
    n_params = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    if t.get("skip") or not trainable:
        if not t.get("skip"):
            raise RuntimeError(f"run {cfg['name']} has nothing to train; set train.skip: true if that is intended")
        info = {"name": cfg["name"], "seed": seed, "skipped": True, "n_params": n_params, "n_trainable": 0,
                "train_time_s": 0.0, "epochs_run": 0}
        done_path.write_text(json.dumps(info, indent=1))
        return info

    seed_everything(seed)
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    cuda = device.type == "cuda"
    if t.get("cudnn_benchmark") and cuda:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    if t.get("channels_last") and cuda:
        model = model.to(memory_format=torch.channels_last)
    distill = t.get("distill") or {}
    if distill and teacher is None:
        raise ValueError(f"run {cfg['name']} distils from {distill.get('teacher')!r}: pass the teacher model")
    kd_weight = float(distill.get("weight", 1.0)) if distill else 0.0
    kd_temperature = float(distill.get("temperature", 1.0)) if distill else 1.0
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    fit_split = t.get("fit_split", "train")
    if fit_split == "val":
        # Stacked generalisation: fit on one half of the validation split (images the frozen base
        # models never trained on), select on the other half. See data.loading.load_stack_data.
        xtr, ytr, xva, yva, xvc, yvc = load_stack_data(cfg)
    elif fit_split == "train":
        xtr, ytr = load_split(cfg, "train")
        xva, yva = load_split(cfg, "val")
        xvc, yvc, _, _ = load_val_c_split(cfg)
    else:
        raise ValueError(f"train.fit_split must be train | val, got {fit_split!r}")
    Xtr, Ytr = u8_to_device(xtr, device), torch.from_numpy(ytr).to(device)
    Xva, Yva = u8_to_device(xva, device), torch.from_numpy(yva).to(device)
    Xvc, Yvc = u8_to_device(xvc, device), torch.from_numpy(yvc).to(device)

    bs, epochs = int(t["batch_size"]), int(t["epochs"])
    steps_per_epoch = len(Xtr) // bs
    if steps_per_epoch == 0:
        raise ValueError(f"batch size {bs} is larger than the training set ({len(Xtr)})")
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(t.get("warmup_epochs", 0) * steps_per_epoch)
    opt = torch.optim.AdamW(param_groups(model, float(t["weight_decay"])), lr=float(t["lr"]),
                            **({"fused": True} if t.get("fused_optimizer") and cuda else {}))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, warmup_steps))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    aug = TrainAugment(cfg["augment"])
    jsd_weight, smoothing = float(cfg["augment"].get("jsd_weight", 12.0)), float(t.get("label_smoothing", 0.0))
    grad_clip, patience = float(t.get("grad_clip", 0) or 0), int(t.get("patience", 0) or 0)
    eval_bs = int(t.get("eval_batch_size", 512))
    ce_all_views = bool(t.get("ce_all_views", False))
    ema = ModelEMA(model, float(t["ema"])) if t.get("ema") else None
    forward_model = torch.compile(model) if t.get("compile") and cuda else model

    state = {"epoch": 0, "best": -1.0, "best_epoch": 0, "best_val_acc": 0.0, "bad_epochs": 0, "elapsed": 0.0}
    ckpt = torch.load(last_path, map_location=device, weights_only=False) if last_path.exists() else None
    if ckpt is not None and ckpt.get("fingerprint") != fingerprint:
        log.warning(f"[{cfg['name']} seed {seed}] ignoring a resume checkpoint from a different configuration")
        ckpt = None
    if ckpt is not None:
        model.load_checkpoint_state(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and "ema" in ckpt:
            ema.module.load_state_dict(ckpt["ema"])
            ema.updates = int(ckpt.get("ema_updates", 0))
        _set_rng_state(ckpt["rng"], device)
        state = ckpt["state"]
        log.info(f"[{cfg['name']} seed {seed}] resuming after epoch {state['epoch']}")
    rows = []
    if log_path.exists():        # keep only epochs that the checkpoint covers
        with open(log_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= state["epoch"]]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)

    wb = None
    if use_wandb:
        try:
            import wandb
            wb = wandb.init(project="roadlens", name=f"{cfg['name']}-seed{seed}", config=cfg, reinit=True)
        except ImportError:
            log.warning("wandb is not installed; continuing with CSV logging only")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log.info(f"[{cfg['name']} seed {seed}] {n_params / 1e6:.2f} M params ({sum(p.numel() for p in trainable) / 1e6:.2f} M "
             f"trainable), {len(Xtr)} train images, {steps_per_epoch} steps/epoch, views={aug.n_views}, amp={amp}")

    if t.get("eval_at_init") and state["epoch"] == 0 and state["best"] < 0:
        # The untrained model is a candidate too ("epoch 0"). For a gate that starts as the plain
        # average this makes selection do no harm: training is kept only if it beats the average.
        val_acc = accuracy(model, Xva, Yva, eval_bs, amp)
        val_c_acc = accuracy(model, Xvc, Yvc, eval_bs, amp)
        state.update(best=val_c_acc, best_epoch=0, best_val_acc=val_acc)
        _save({"model": model.checkpoint_state(), "epoch": 0, "val_c_acc": val_c_acc, "val_acc": val_acc}, best_path)
        log.info(f"[{cfg['name']} seed {seed}] epoch   0  (untrained)  val {val_acc:.4f}  val-C {val_c_acc:.4f}  *")

    stop = patience > 0 and state["bad_epochs"] >= patience
    for epoch in range(state["epoch"] + 1, epochs + 1):
        if stop:
            break
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Xtr), device=device)
        tot = {"loss": 0.0, "ce": 0.0, "jsd": 0.0, "kd": 0.0, "correct": 0}
        step_stats = {}
        bar = tqdm(range(steps_per_epoch), desc=f"{cfg['name']} s{seed} ep {epoch}/{epochs}", leave=False, mininterval=2.0)
        for i in bar:
            idx = perm[i * bs:(i + 1) * bs]
            y = Ytr[idx]
            views = aug(as_float(Xtr[idx]))
            if t.get("channels_last") and cuda:
                views = [v.contiguous(memory_format=torch.channels_last) for v in views]
            loss, ce, jsd, first = training_loss(forward_model, views, y, aug.mode == "jsd", smoothing, jsd_weight,
                                                 amp, ce_all_views, teacher, kd_weight, kd_temperature, step_stats)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {i}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            tot["kd"] += float(step_stats.get("kd", 0.0))
            tot["loss"] += loss.item()
            tot["ce"] += ce.item()
            tot["jsd"] += jsd.item()
            tot["correct"] += int((first.argmax(dim=1) == y).sum())
            if i % 20 == 0:
                bar.set_postfix(loss=f"{loss.item():.3f}")
        epoch_s = time.time() - t0
        eval_model = ema.module if ema is not None else model
        val_acc = accuracy(eval_model, Xva, Yva, eval_bs, amp)
        val_c_acc = accuracy(eval_model, Xvc, Yvc, eval_bs, amp)
        row = {"epoch": epoch, "lr": f"{opt.param_groups[0]['lr']:.3e}",
               "loss": f"{tot['loss'] / steps_per_epoch:.4f}", "ce": f"{tot['ce'] / steps_per_epoch:.4f}",
               "jsd": f"{tot['jsd'] / steps_per_epoch:.4f}", "kd": f"{tot['kd'] / steps_per_epoch:.4f}",
               "train_acc": f"{tot['correct'] / (steps_per_epoch * bs):.4f}",
               "val_acc": f"{val_acc:.4f}", "val_c_acc": f"{val_c_acc:.4f}",
               "epoch_s": f"{epoch_s:.1f}", "img_per_s": f"{steps_per_epoch * bs / epoch_s:.0f}"}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
        if wb is not None:
            wb.log({k: float(v) for k, v in row.items()})

        state["epoch"], state["elapsed"] = epoch, state["elapsed"] + epoch_s
        if val_c_acc > state["best"]:
            state.update(best=val_c_acc, best_epoch=epoch, best_val_acc=val_acc, bad_epochs=0)
            _save({"model": eval_model.checkpoint_state(), "epoch": epoch, "val_c_acc": val_c_acc, "val_acc": val_acc},
                  best_path)
        else:
            state["bad_epochs"] += 1
        _save({"model": model.checkpoint_state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
               "scaler": scaler.state_dict(), "rng": _rng_state(device), "state": state,
               "fingerprint": fingerprint,
               **({"ema": ema.module.state_dict(), "ema_updates": ema.updates} if ema is not None else {})}, last_path)
        log.info(f"[{cfg['name']} seed {seed}] epoch {epoch:3d}  loss {row['loss']}  train {row['train_acc']}  "
                 f"val {row['val_acc']}  val-C {row['val_c_acc']}  ({epoch_s:.0f}s, {row['img_per_s']} img/s)"
                 + ("  *" if state["bad_epochs"] == 0 else ""))
        stop = patience > 0 and state["bad_epochs"] >= patience

    info = {"name": cfg["name"], "seed": seed, "skipped": False, "n_params": n_params,
            "n_trainable": sum(p.numel() for p in trainable), "epochs_run": state["epoch"],
            "best_epoch": state["best_epoch"], "best_val_c_acc": state["best"], "val_acc_at_best": state["best_val_acc"],
            "train_time_s": round(state["elapsed"], 1), "early_stopped": bool(stop),
            "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated(device) / 2 ** 20) if device.type == "cuda" else None}
    if wb is not None:
        wb.summary.update(info)
        wb.finish()
    if best_path != final_best:             # the one write of the weights to the persistent directory
        tmp = final_best.with_suffix(".pt.tmp")
        shutil.copyfile(best_path, tmp)
        os.replace(tmp, final_best)
        best_path.unlink(missing_ok=True)
    done_path.write_text(json.dumps(info, indent=1))
    last_path.unlink(missing_ok=True)       # only needed for resuming
    return info
