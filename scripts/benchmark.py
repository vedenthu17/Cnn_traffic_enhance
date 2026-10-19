"""Measure training / inference speed of every run on this GPU and estimate experiment run times.

    python scripts/benchmark.py                    # all runs used by the experiments (about 3 minutes)
    python scripts/benchmark.py --batch-size 48    # on a GPU with less than ~6 GB memory

No dataset or checkpoint is needed: the models are built with random weights and fed random images
through the real training step (augmentation, mixed precision, loss, backward, optimizer). The
estimates assume every run trains for all of its epochs (early stopping makes real runs shorter)
and are written to results/benchmark.csv.
"""

import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd
import torch

from roadlens.augment import TrainAugment
from roadlens.engine import param_groups, training_loss
from roadlens.models import build_model
from roadlens.runs import load_run_config
from roadlens.utils import REPO_ROOT, apply_profile, get_device, load_config, results_root

N_TRAIN, N_TEST, N_SETS = 35219, 12630, 71          # GTSRB train split, test set, clean + 70 corrupted sets
EXPERIMENTS = ["exp1_clean", "exp2_corruption", "exp3_ablation", "exp4_leave_category_out",
               "exp5_cross_dataset", "exp6_tent_batchsize"]


def build(cfg):
    branches = [build(load_run_config(b)) for b in cfg["model"].get("branches", [])] or None
    return build_model(cfg["model"], cfg["data"]["num_classes"], branches, pretrained=False)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(name, device, batch_size, steps, profile):
    cfg = load_run_config(name, profile)
    t, size = cfg["train"], cfg["data"]["image_size"]
    bs = batch_size or int(t["batch_size"])
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    model = build(cfg).to(device)
    row = {"run": name, "epochs": int(t["epochs"]), "batch_size": bs, "train_img_s": float("nan"), "peak_mem_gb": float("nan")}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    x8 = torch.randint(0, 256, (bs, 3, size, size), device=device, dtype=torch.uint8)
    y = torch.randint(0, cfg["data"]["num_classes"], (bs,), device=device)
    if not t.get("skip"):
        aug = TrainAugment(cfg["augment"])
        opt = torch.optim.AdamW(param_groups(model, float(t["weight_decay"])), lr=float(t["lr"]))
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        model.train()
        for i in range(steps + 5):
            if i == 5:                              # the first steps include cuDNN / allocator warm-up
                sync(device)
                t0 = time.perf_counter()
            loss = training_loss(model, aug(x8.float().div(255)), y, aug.mode == "jsd",
                                 float(t.get("label_smoothing", 0)), float(cfg["augment"].get("jsd_weight", 12)), amp)[0]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sync(device)
        row["train_img_s"] = steps * bs / (time.perf_counter() - t0)
        row["views"] = aug.n_views
    model.eval()
    eb = min(int(t.get("eval_batch_size", 512)), 4 * bs) if batch_size else int(t.get("eval_batch_size", 512))
    xe = torch.randint(0, 256, (eb, 3, size, size), device=device, dtype=torch.uint8)
    with torch.no_grad():
        for i in range(8):
            if i == 3:
                sync(device)
                t0 = time.perf_counter()
            with torch.autocast(device.type, enabled=amp):
                model(xe.float().div(255))
        sync(device)
    row["infer_img_s"] = 5 * eb / (time.perf_counter() - t0)
    if device.type == "cuda":
        row["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
    row["train_min"] = 0.0 if t.get("skip") else row["epochs"] * N_TRAIN / row["train_img_s"] / 60
    row["eval_gtsrb_c_min"] = N_SETS * N_TEST / row["infer_img_s"] / 60
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def experiment_runs(profile):
    """Runs each experiment needs (branches of fusions and the mCE normaliser included)."""
    out = {}
    for name in EXPERIMENTS:
        cfg = apply_profile(load_config(REPO_ROOT / "configs" / "experiments" / f"{name}.yaml"), profile)
        runs = list(cfg.get("runs", [])) + [s["run"] for s in cfg.get("steps", [])] + list(cfg.get("candidates", []))
        runs += [cfg[k] for k in ("baseline", "normalizer") if k in cfg]
        todo, seen = list(dict.fromkeys(runs)), []
        while todo:
            r = todo.pop(0)
            if r not in seen:
                seen.append(r)
                todo += load_run_config(r)["model"].get("branches", [])
        out[name] = (seen, len(cfg["seeds"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None, help="default: every run used by the experiments")
    ap.add_argument("--batch-size", type=int, default=0, help="override the training batch size (small GPUs)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    exps = experiment_runs(args.profile)
    runs = args.runs or list(dict.fromkeys(r for rs, _ in exps.values() for r in rs))
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    rows = []
    for name in runs:
        try:
            rows.append(measure(name, device, args.batch_size, args.steps, args.profile))
        except RuntimeError as err:                 # most likely out of memory
            print(f"{name}: {str(err).splitlines()[0]}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 200)
    print(df.round(1).to_string())
    out = results_root(apply_profile(load_config(REPO_ROOT / "configs" / "base.yaml"), args.profile)) / "benchmark.csv"
    df.round(2).to_csv(out)

    if not args.runs:
        print("\nEstimated GPU time (all epochs, no early stopping; training shared between experiments is counted once):")
        trained, total = set(), 0.0
        for name, (needed, n_seeds) in exps.items():
            new = [r for r in needed if r not in trained and r in df.index]
            trained |= set(new)
            train = n_seeds * sum(df.train_min[r] for r in new)
            evals = n_seeds * sum(df.eval_gtsrb_c_min[r] for r in needed if r in df.index)
            total += train + evals
            print(f"  {name:28s} new training {train / 60:5.1f} h   evaluation {evals / 60:5.1f} h   ({n_seeds} seed(s))")
        print(f"  {'total':28s} {total / 60:5.1f} h   (+ GTSRB-C generation on the CPU, + test-time-adaptation streams)")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
