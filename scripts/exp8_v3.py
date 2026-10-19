"""Experiment 8 - RoadLens v3: one mechanism per row, other architectures, self-supervised
pre-training and distillation into one fast network.

    python scripts/exp8_v3.py --profile pilot

Rows (configs/experiments/exp8_v3.yaml):
    local-global contrast normalisation stem       model.input_norm: local
    amplitude-phase recombination (APR) op set     augment.ops: [..., apr]
    EMA of the weights + GPU speed-ups             train.ema, channels_last, fused AdamW, cuDNN autotune
    native 64 px input                             model.input_size: 64
    SimCLR self-supervised pre-training            model.init_from: resnet18_simclr (roadlens/ssl.py)
    ConvNeXt-Atto / EfficientNet-B0 / MobileNetV3  model.name
    v3 view-expert ensemble                        ensemble_v3
    distilled students                             train.distill (teacher: ensemble_views_in)

Writes to results/exp8_v3/:
    v3.csv/.txt/.tex             accuracy, mCE, ECE, Val-C (sel.), BTSD per row (as in experiment 7)
    per_corruption.*, severity5.*  accuracy per row and corruption
    heatmaps/<row>.png           corruption x severity accuracy heatmap of every row
    per_family.csv/.txt          accuracy per corruption family (noise / blur / weather / digital) and at S5
    efficiency.csv/.txt          parameters, forward passes per image, training minutes, GPU throughput
"""

import json
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
from exp7_improved import ladder

from roadlens.data.loading import load_test
from roadlens.eval.evaluate import predict_logits
from roadlens.eval import plots
from roadlens.experiment import Experiment, make_parser
from roadlens.runs import ensure_trained, load_model, load_run_config, run_dir
from roadlens.utils import corruption_categories, get_logger

log = get_logger()


def per_family(exp, long):
    cats = corruption_categories(exp.cfg)
    corr = long[long.corruption != "clean"].assign(family=lambda d: d.corruption.map(cats))
    order = list(dict.fromkeys(s["label"] for s in exp.cfg["steps"]))
    fam = (100 * corr.groupby(["step", "family"], sort=False).acc.mean()).unstack("family")
    fam = fam[list(exp.cfg["corruptions"]["test"])]
    fam["fog S5"] = 100 * corr[(corr.corruption == "fog") & (corr.severity == 5)].groupby("step").acc.mean()
    fam["all S5"] = 100 * corr[corr.severity == 5].groupby("step").acc.mean()
    fam = fam.reindex(order)
    fam.to_csv(exp.out / "per_family.csv", float_format="%.2f")
    (exp.out / "per_family.txt").write_text(fam.round(1).to_string() + "\n", encoding="utf-8")
    print("\n" + fam.round(1).to_string())


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def throughput(model, x: np.ndarray, device, batch_size: int, repeats: int) -> tuple[float, float]:
    """(images per second at `batch_size`, milliseconds per single image at batch size 1), with AMP."""
    amp = device.type == "cuda"
    predict_logits(model, x[:batch_size], device, batch_size, amp)            # warm-up (kernels, allocator)
    best = 0.0
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        predict_logits(model, x, device, batch_size, amp)
        _sync(device)
        best = max(best, len(x) / (time.perf_counter() - t0))
    one = x[:1]
    predict_logits(model, one, device, 1, amp)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(50):
        predict_logits(model, one, device, 1, amp)
    _sync(device)
    return best, 1000 * (time.perf_counter() - t0) / 50


def train_minutes(run: str, seed: int, profile) -> float:
    """Training time of a run including everything it depends on (branches, teacher, pre-training)."""
    cfg = load_run_config(run, profile)
    own = json.loads((run_dir(cfg, seed) / "done.json").read_text()).get("train_time_s", 0.0) / 60
    deps = list(cfg["model"].get("branches", [])) + [cfg["model"].get("init_from")]
    deps.append((cfg["train"].get("distill") or {}).get("teacher"))
    return own + sum(train_minutes(d, seed, profile) for d in deps if d)


def efficiency(exp):
    sp = exp.cfg.get("speed", {})
    seed = exp.seeds[0]
    runs = list(dict.fromkeys(s["run"] for s in exp.cfg["steps"]))
    x, _ = load_test(load_run_config(runs[0], exp.profile), "gtsrb")
    x = np.asarray(x[:int(sp.get("n_images", 2048))])
    rows = []
    for run in runs:
        ensure_trained(run, seed, exp.profile, exp.device)
        model, cfg = load_model(run, seed, exp.device, exp.profile)
        m = cfg["model"]
        passes = len(m.get("branches", [])) * len((m.get("fusion") or {}).get("views", ["raw"])) if m["name"] == "fusion" else 1
        ips, ms = throughput(model, x, exp.device, int(sp.get("batch_size", 256)), int(sp.get("repeats", 3)))
        rows.append({"run": run, "model": exp.label(run), "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
                     "forward_passes": passes, "input_px": m.get("input_size", 128) if m["name"] != "fusion" else None,
                     "train_min_incl_deps": train_minutes(run, seed, exp.profile),
                     "img_per_s": ips, "ms_per_image_bs1": ms})
        del model
        if exp.device.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(exp.out / "efficiency.csv", index=False, float_format="%.3f")
    gpu = torch.cuda.get_device_name(exp.device) if exp.device.type == "cuda" else "CPU"
    (exp.out / "efficiency.txt").write_text(f"device: {gpu}, AMP, seed {seed}\n" + df.round(2).to_string(index=False)
                                           + "\n", encoding="utf-8")
    print(f"\n{gpu}\n" + df.round(2).to_string(index=False))


def main():
    exp = Experiment(make_parser(__doc__, "configs/experiments/exp8_v3.yaml").parse_args(), "exp8_v3")
    _, long = ladder(exp, stem="v3", title="The v3 approach")
    per_family(exp, long)
    plots.plot_corruption_heatmaps(long, "step", exp.out / "heatmaps")
    efficiency(exp)


if __name__ == "__main__":
    main()
