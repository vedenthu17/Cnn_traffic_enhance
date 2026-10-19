"""Corruption x severity accuracy heatmaps (PNG), one per model, from the cached evaluations.

    python scripts/make_heatmaps.py --profile pilot                      # every trained run that has GTSRB-C results
    python scripts/make_heatmaps.py --profile pilot --runs fusion_v2 student_v3
    python scripts/make_heatmaps.py --from-csv results/exp9_classical_unsupervised/classical_long.csv

No model is evaluated: the figures are drawn from <run dir>/eval/gtsrb_c.csv (written by the experiment
scripts) or from a long CSV with columns model, corruption, severity, acc. Output:
    results[_profile]/heatmaps/<run>.png
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from roadlens.eval import plots
from roadlens.runs import load_run_config, run_dir
from roadlens.utils import REPO_ROOT, apply_profile, load_config, results_root


def label_of(cfg_labels: dict, run: str) -> str:
    return cfg_labels.get(run, run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None, help="smoke | pilot")
    ap.add_argument("--runs", nargs="+", default=None, help="run names (default: every run with GTSRB-C results)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-csv", default=None, help="long CSV (model, corruption, severity, acc) instead of run folders")
    ap.add_argument("--dataset", default="gtsrb", choices=["gtsrb", "btsd"])
    ap.add_argument("--out", default=None, help="output folder (default: results[_profile]/heatmaps)")
    args = ap.parse_args()

    base = apply_profile(load_config(REPO_ROOT / "configs" / "experiments" / "exp8_v3.yaml"), args.profile)
    labels = base.get("labels", {})
    out = Path(args.out) if args.out else results_root(base) / "heatmaps"
    name = "GTSRB" if args.dataset == "gtsrb" else "BTSD"
    written = []

    if args.from_csv:
        long = pd.read_csv(args.from_csv)
        for model, g in long.groupby("model", sort=False):
            safe = "".join(ch if ch.isalnum() else "_" for ch in model).strip("_").lower()
            written.append(plots.plot_corruption_heatmap(g, f"{name} - {model} accuracy (%)", out / f"{safe}.png"))
    else:
        runs = args.runs
        if runs is None:
            runs = sorted(p.name for p in (run_dir(load_run_config("resnet18_base", args.profile), 0).parent.parent).iterdir()
                          if p.is_dir())
        for run in runs:
            cfg = load_run_config(run, args.profile)
            csv = run_dir(cfg, args.seed) / "eval" / f"{args.dataset}_c.csv"
            if not csv.exists():
                print(f"skip {run}: no {csv.name}")
                continue
            written.append(plots.plot_corruption_heatmap(pd.read_csv(csv), f"{name} - {label_of(labels, run)} accuracy (%)",
                                                         out / f"{run}.png"))
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
