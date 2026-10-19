"""All figures, as vector PDF in results/figures/ (with the plotted numbers as CSV next to them).

    python scripts/make_figures.py                      # everything
    python scripts/make_figures.py --only curves confusion

    curves       severity_curves.pdf      accuracy vs. severity per corruption, per model
    confusion    confusion_<model>_<condition>.pdf   clean, and the baseline's worst corruption at severity 5
    reliability  reliability.pdf          reliability diagrams, clean and GTSRB-C
    fourier      fourier_heatmap.pdf      Fourier heat map of error (Yin et al. 2019)
    gradcam      gradcam.pdf              Grad-CAM on clean vs. corrupted samples

Uses the cached evaluations of experiments 1-2 and trains anything that is still missing.
"""

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from roadlens.data.classes import GTSRB_SHORT
from roadlens.data.loading import as_float, load_corrupted_set, load_test, u8_to_device
from roadlens.eval import plots
from roadlens.eval.fourier import fourier_heatmaps
from roadlens.eval.gradcam import grad_cam
from roadlens.eval.metrics import mean_std
from roadlens.experiment import Experiment, make_parser
from roadlens.runs import ensure_trained, load_model
from roadlens.utils import corruption_names, get_logger

log = get_logger()
ALL = ["curves", "confusion", "reliability", "fourier", "gradcam"]
ARCH = {"resnet18": "ResNet-18", "deit_small_patch16_224": "DeiT-S", "smallcnn": "SmallCNN",
        "swin_tiny_patch4_window7_224": "Swin-T"}


def curves(exp):
    runs = exp.cfg["curve_runs"]
    long, _ = exp.evaluate(runs, "gtsrb_c")
    plots.plot_severity_curves(long, runs, exp.cfg["labels"], corruption_names(exp.cfg), exp.out / "severity_curves.pdf")
    table = mean_std(long.assign(acc=100 * long.acc), ["run", "corruption", "severity"], ["acc"])
    table.insert(1, "model", table.run.map(exp.label))
    table.to_csv(exp.out / "severity_curves.csv", index=False, float_format="%.3f")


def confusion(exp):
    base, prop = exp.cfg["baseline"], exp.cfg["proposed"]
    long, arrays = exp.evaluate([base, prop], "gtsrb_c")
    top = max(long.severity)

    def worst(run):
        d = long[(long.run == run) & (long.severity == top)]
        return d.groupby("corruption").acc.mean().idxmin()

    conditions = {base: [("clean", 0), (worst(base), top)], prop: [("clean", 0), (worst(base), top)]}
    if worst(prop) != worst(base):
        conditions[prop].append((worst(prop), top))
    for run, conds in conditions.items():
        order = long[(long.run == run) & (long.seed == exp.seeds[0])].reset_index(drop=True)
        for corruption, sev in conds:
            row = order.index[(order.corruption == corruption) & (order.severity == sev)][0]
            cm = sum(arrays[(run, seed)]["confusion"][row] for seed in exp.seeds)
            name = "clean" if corruption == "clean" else f"{corruption}_s{sev}"
            title = f"{exp.label(run)}: " + ("clean test set" if corruption == "clean"
                                              else f"{corruption.replace('_', ' ')}, severity {sev}")
            if len(exp.seeds) > 1:
                title += f", {len(exp.seeds)} seeds pooled"
            plots.plot_confusion(cm, GTSRB_SHORT, title, exp.out / f"confusion_{run}_{name}.pdf")
            pd.DataFrame(cm, index=GTSRB_SHORT, columns=GTSRB_SHORT).to_csv(exp.out / f"confusion_{run}_{name}.csv")


def reliability(exp):
    runs = exp.cfg["reliability_runs"]
    long, arrays = exp.evaluate(runs, "gtsrb_c")
    bins, rows = {}, []
    for run in runs:
        order = long[(long.run == run) & (long.seed == exp.seeds[0])].reset_index(drop=True)
        clean_rows = order.index[order.corruption == "clean"].tolist()
        corr_rows = order.index[order.corruption != "clean"].tolist()
        for cond, sel in (("clean", clean_rows), ("GTSRB-C", corr_rows)):
            b = sum(arrays[(run, seed)]["bins"][sel].sum(0) for seed in exp.seeds)
            bins[(run, cond)] = b
            for i, (count, conf_sum, correct) in enumerate(b):
                rows.append({"run": run, "condition": cond, "bin": i, "count": int(count),
                             "mean_confidence": conf_sum / count if count else np.nan,
                             "accuracy": correct / count if count else np.nan})
    plots.plot_reliability(bins, ["clean", "GTSRB-C"], runs, exp.cfg["labels"], exp.out / "reliability.pdf")
    pd.DataFrame(rows).to_csv(exp.out / "reliability.csv", index=False, float_format="%.5f")


def fourier(exp):
    f = exp.cfg["fourier"]
    models = {}
    for run in f["runs"]:
        ensure_trained(run, f["seed"], exp.profile, exp.device, exp.wandb)
        models[run] = load_model(run, f["seed"], exp.device, exp.profile)[0]
    x, y = load_test(exp.cfg)
    idx = np.sort(np.random.RandomState(0).permutation(len(x))[:int(f["n_images"])])
    maps, clean = fourier_heatmaps(models, x[idx], y[idx], exp.device, eps=float(f["eps"]), stride=int(f["stride"]))
    plots.plot_fourier(maps, clean, exp.cfg["labels"], float(f["eps"]), exp.out / "fourier_heatmap.pdf")
    for run, m in maps.items():
        pd.DataFrame(m).to_csv(exp.out / f"fourier_heatmap_{run}.csv", index=False, header=False, float_format="%.4f")


def gradcam(exp):
    g = exp.cfg["gradcam"]
    x, y = load_test(exp.cfg)
    idx = np.sort(np.random.RandomState(1).permutation(len(x))[:int(g["n_samples"])])
    sev = int(g["severity"])
    names = [g["corruptions"][j % len(g["corruptions"])] for j in range(len(idx))]
    clean = as_float(u8_to_device(x[idx], exp.device))
    corrupted = as_float(u8_to_device(
        np.stack([np.asarray(load_corrupted_set(exp.cfg, "gtsrb", n, sev)[0][i]) for n, i in zip(names, idx)]), exp.device))
    labels = torch.from_numpy(y[idx])

    def to_img(t):
        return t.permute(1, 2, 0).cpu().numpy()

    rows = {"clean": [], "corrupted": []}       # condition -> list of (row label, [cell per sample])
    for run in g["runs"]:
        ensure_trained(run, g["seed"], exp.profile, exp.device, exp.wandb)
        model = load_model(run, g["seed"], exp.device, exp.profile)[0]
        targets = model.cam_targets()
        if not targets:
            log.warning(f"{run}: no Grad-CAM layer known for this architecture; skipped")
            continue
        for cond, batch in (("clean", clean), ("corrupted", corrupted)):
            cams, logits = grad_cam(model, targets, batch)
            pred = logits.argmax(dim=1).cpu()
            for t, cam in zip(targets, cams):
                label = exp.label(run) if len(targets) == 1 else f"{exp.label(run)}\n{ARCH.get(t.name, t.name)} branch"
                cells = [(plots.cam_overlay(to_img(batch[j]), cam[j].cpu().numpy()),
                          f"{GTSRB_SHORT[pred[j]]} " + ("✓" if pred[j] == labels[j] else "✗"))
                         for j in range(len(idx))]
                rows[cond].append((label, cells))
        del model
    row_labels, columns = [], [{"title": f"{GTSRB_SHORT[int(labels[j])]}", "cells": []} for j in range(len(idx))]
    for cond, batch in (("clean", clean), ("corrupted", corrupted)):
        row_labels.append(f"input\n({cond})")
        for j in range(len(idx)):
            columns[j]["cells"].append((to_img(batch[j]), "clean" if cond == "clean" else f"{names[j].replace('_', ' ')} s{sev}"))
        for label, cells in rows[cond]:
            row_labels.append(label)
            for j in range(len(idx)):
                columns[j]["cells"].append(cells[j])
    plots.plot_gradcam(columns, row_labels, exp.out / "gradcam.pdf")


def main():
    ap = make_parser(__doc__, "configs/experiments/figures.yaml")
    ap.add_argument("--only", nargs="+", choices=ALL, default=ALL)
    args = ap.parse_args()
    exp = Experiment(args, "figures")
    for name in args.only:
        log.info(f"figure: {name}")
        globals()[name](exp)
    log.info(f"figures -> {exp.out}")


if __name__ == "__main__":
    main()
