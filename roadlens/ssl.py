"""Self-supervised pre-training (RoadLens v3): SimCLR on the GTSRB training images, labels unused.

Two augmented views of every image (the same AugMix / PRIME / APR ops as the supervised runs, none
of them a test corruption) are pulled together and pushed away from the other images of the batch
with the NT-Xent loss (Chen et al. 2020), through a small projection head on the pooled features.
The result is a backbone whose features are invariant to the training perturbations; a supervised
run then starts from it (`model.init_from: <this run>`).

No label is used for training or for choosing the checkpoint (the last epoch is kept). Labels are
used only to monitor the representation: a weighted k-nearest-neighbour classifier on the features
(k = 20, cosine similarity, Wu et al. 2018) scores the clean validation split and validation-C each
epoch, and the score is logged.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .augment import TrainAugment
from .data.loading import as_float, load_split, load_val_c_split, u8_to_device
from .engine import lr_factor, param_groups
from .utils import get_logger, seed_everything

log = get_logger()
SSL_LOG_FIELDS = ["epoch", "lr", "loss", "knn_val_acc", "knn_val_c_acc", "epoch_s", "img_per_s"]


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent loss of two batches of embeddings [B, D]; row i of z1 and row i of z2 are the positive pair."""
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


@torch.no_grad()
def embed(model, x_u8: torch.Tensor, batch_size: int = 512, amp: bool = True) -> torch.Tensor:
    """L2-normalised pooled features of a uint8 NCHW tensor on the model's device."""
    model.eval()
    out = []
    for s in range(0, len(x_u8), batch_size):
        with torch.autocast(x_u8.device.type, enabled=amp):
            out.append(model.features_logits(as_float(x_u8[s:s + batch_size]))[0].float())
    return F.normalize(torch.cat(out), dim=1)


@torch.no_grad()
def knn_accuracy(bank: torch.Tensor, bank_y: torch.Tensor, query: torch.Tensor, query_y: torch.Tensor,
                 num_classes: int, k: int = 20, tau: float = 0.07) -> float:
    """Weighted kNN accuracy: each of the k most similar bank features votes exp(sim / tau) for its class."""
    correct = 0
    for s in range(0, len(query), 1024):
        sim = query[s:s + 1024] @ bank.t()
        top, idx = sim.topk(k, dim=1)
        votes = torch.zeros(len(top), num_classes, device=bank.device)
        votes.scatter_add_(1, bank_y[idx], torch.exp(top / tau))
        correct += int((votes.argmax(dim=1) == query_y[s:s + 1024]).sum())
    return correct / len(query)


def train_simclr(cfg: dict, seed: int, run_dir: Path, model, device, scratch_dir: Path | None = None,
                 fingerprint: str | None = None) -> dict:
    """Pre-train `model` (a Classifier) with SimCLR. Leaves best.pt (= last epoch), log.csv, done.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    work = Path(scratch_dir) if scratch_dir else run_dir
    work.mkdir(parents=True, exist_ok=True)
    done_path, log_path, final_best, last_path = run_dir / "done.json", run_dir / "log.csv", run_dir / "best.pt", work / "last.pt"
    if done_path.exists():
        return json.loads(done_path.read_text())

    t, s = cfg["train"], cfg.get("ssl", {})
    seed_everything(seed)
    cuda = device.type == "cuda"
    amp = bool(t.get("amp", True)) and cuda
    if t.get("cudnn_benchmark") and cuda:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    xtr, ytr = load_split(cfg, "train")
    xva, yva = load_split(cfg, "val")
    xvc, yvc, _, _ = load_val_c_split(cfg)
    Xtr, Ytr = u8_to_device(xtr, device), torch.from_numpy(ytr).to(device)
    Xva, Yva = u8_to_device(xva, device), torch.from_numpy(yva).to(device)
    Xvc, Yvc = u8_to_device(xvc, device), torch.from_numpy(yvc).to(device)
    bank_idx = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(seed))[:int(s.get("knn_bank", 10000))]
    bank_idx = bank_idx.to(device)

    proj = nn.Sequential(nn.Linear(model.feature_dim, int(s.get("proj_hidden", 512))), nn.ReLU(inplace=True),
                         nn.Linear(int(s.get("proj_hidden", 512)), int(s.get("proj_dim", 128)))).to(device)
    params = param_groups(model, float(t["weight_decay"])) + param_groups(proj, float(t["weight_decay"]))
    opt = torch.optim.AdamW(params, lr=float(t["lr"]), **({"fused": True} if t.get("fused_optimizer") and cuda else {}))
    bs, epochs = int(t["batch_size"]), int(t["epochs"])
    steps = len(Xtr) // bs
    total, warmup = epochs * steps, int(t.get("warmup_epochs", 0) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: lr_factor(i, total, warmup))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    aug = TrainAugment({**cfg["augment"], "mode": "jsd"})          # [jitter(x), augmix(.), augmix(.)]
    temperature = float(s.get("temperature", 0.2))

    start, elapsed, knn = 1, 0.0, (0.0, 0.0)
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("fingerprint") == fingerprint:
            model.load_checkpoint_state(ck["model"])
            proj.load_state_dict(ck["proj"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            start, elapsed = ck["epoch"] + 1, ck["elapsed"]
            log.info(f"[{cfg['name']} seed {seed}] resuming SimCLR after epoch {ck['epoch']}")
    rows = []
    if log_path.exists() and start > 1:
        with open(log_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < start]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SSL_LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    log.info(f"[{cfg['name']} seed {seed}] SimCLR: {len(Xtr)} unlabelled images, {steps} steps/epoch, "
             f"batch {bs} x 2 views, temperature {temperature}")

    for epoch in range(start, epochs + 1):
        model.train()
        proj.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(len(Xtr), device=device)
        for i in tqdm(range(steps), desc=f"{cfg['name']} s{seed} ep {epoch}/{epochs}", leave=False, mininterval=2.0):
            views = aug(as_float(Xtr[perm[i * bs:(i + 1) * bs]]))[1:]
            with torch.autocast(device.type, enabled=amp):
                feats = model.features_logits(torch.cat(views))[0]
                z = proj(feats)
            loss = nt_xent(*z.float().chunk(2), temperature=temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite SimCLR loss at epoch {epoch}, step {i}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(proj.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        epoch_s = time.time() - t0
        elapsed += epoch_s
        bank = embed(model, Xtr[bank_idx], amp=amp)
        k = int(cfg["data"]["num_classes"])
        knn = (knn_accuracy(bank, Ytr[bank_idx], embed(model, Xva, amp=amp), Yva, k),
               knn_accuracy(bank, Ytr[bank_idx], embed(model, Xvc, amp=amp), Yvc, k))
        row = {"epoch": epoch, "lr": f"{opt.param_groups[0]['lr']:.3e}", "loss": f"{tot / steps:.4f}",
               "knn_val_acc": f"{knn[0]:.4f}", "knn_val_c_acc": f"{knn[1]:.4f}", "epoch_s": f"{epoch_s:.1f}",
               "img_per_s": f"{steps * bs / epoch_s:.0f}"}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=SSL_LOG_FIELDS).writerow(row)
        log.info(f"[{cfg['name']} seed {seed}] epoch {epoch:3d}  NT-Xent {row['loss']}  kNN val {row['knn_val_acc']}  "
                 f"kNN val-C {row['knn_val_c_acc']}  ({epoch_s:.0f}s, {row['img_per_s']} img/s)")
        tmp = last_path.with_suffix(".tmp")
        torch.save({"model": model.checkpoint_state(), "proj": proj.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "elapsed": elapsed,
                    "fingerprint": fingerprint}, tmp)
        os.replace(tmp, last_path)

    tmp = final_best.with_suffix(".pt.tmp")
    torch.save({"model": model.checkpoint_state(), "epoch": epochs, "knn_val_c_acc": knn[1]}, tmp)
    os.replace(tmp, final_best)
    info = {"name": cfg["name"], "seed": seed, "skipped": False, "objective": "simclr",
            "n_params": sum(p.numel() for p in model.parameters()), "n_trainable": sum(p.numel() for p in model.parameters()),
            "epochs_run": epochs, "best_epoch": epochs, "best_val_c_acc": knn[1], "val_acc_at_best": knn[0],
            "train_time_s": round(elapsed, 1), "early_stopped": False,
            "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated(device) / 2 ** 20) if cuda else None,
            "note": "best_val_c_acc / val_acc_at_best are kNN monitoring scores; the last epoch is kept"}
    done_path.write_text(json.dumps(info, indent=1))
    last_path.unlink(missing_ok=True)
    if work != run_dir:
        shutil.rmtree(work, ignore_errors=True)
    return info
