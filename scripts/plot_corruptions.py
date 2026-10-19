"""Visual + numeric sanity check of the corruptions at the working resolution.

    python scripts/plot_corruptions.py

Writes to results/corruptions/:
  grid_<i>.pdf            14 test corruptions x 5 severities for one test image (several images)
  grid_val_c.pdf          the 4 held-out validation corruptions x 5 severities
  package_vs_roadlens.pdf imagecorruptions defaults vs. the 64 px calibration used here
  corruption_stats.csv    mean PSNR (dB) vs. clean per corruption/severity, both parameter sets
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roadlens.data import corruptions as C
from roadlens.data.classes import GTSRB_SHORT
from roadlens.data.gtsrb import load_gtsrb_cfg
from roadlens.utils import REPO_ROOT, data_root, get_logger, load_config, corruption_names

log = get_logger()


def _show(ax, img):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def severity_grid(img, names, seed, title, path):
    fig, axes = plt.subplots(len(names), 6, figsize=(6 * 1.05 + 1.3, len(names) * 1.05 + 0.5), squeeze=False)
    for r, name in enumerate(names):
        _show(axes[r, 0], img)
        axes[r, 0].set_ylabel(name.replace("_", " "), rotation=0, ha="right", va="center", fontsize=8)
        for s in range(1, 6):
            _show(axes[r, s], C.corrupt(img, name, s, C.derive_seed(seed, name, s, 0)))
    for s, t in enumerate(["clean"] + [f"severity {i}" for i in range(1, 6)]):
        axes[0, s].set_title(t, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=0.15, w_pad=0.15)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=110)
    plt.close(fig)


def comparison_grid(img, names, seed, path):
    sevs = [1, 3, 5]
    fig, axes = plt.subplots(len(names), 7, figsize=(7 * 1.05 + 1.3, len(names) * 1.05 + 0.7), squeeze=False)
    for r, name in enumerate(names):
        _show(axes[r, 0], img)
        axes[r, 0].set_ylabel(name.replace("_", " "), rotation=0, ha="right", va="center", fontsize=8)
        for k, s in enumerate(sevs):
            sd = C.derive_seed(seed, name, s, 0)
            _show(axes[r, 1 + k], C.package_default(img, name, s, sd))
            _show(axes[r, 4 + k], C.corrupt(img, name, s, sd))
    axes[0, 0].set_title("clean", fontsize=8)
    for k, s in enumerate(sevs):
        axes[0, 1 + k].set_title(f"package s{s}", fontsize=8)
        axes[0, 4 + k].set_title(f"RoadLens s{s}", fontsize=8)
    fig.suptitle("imagecorruptions defaults (ImageNet-scale) vs. 64 px calibration", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=0.15, w_pad=0.15)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=110)
    plt.close(fig)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2, axis=(1, 2, 3))
    return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-8))


def stats(x, names, seed):
    rows = []
    for name in names:
        for s in range(1, 6):
            ours = np.stack([C.corrupt(im, name, s, C.derive_seed(seed, name, s, i)) for i, im in enumerate(x)])
            pkg = np.stack([C.package_default(im, name, s, C.derive_seed(seed, name, s, i))
                            for i, im in enumerate(x)])
            rows.append({"corruption": name, "severity": s,
                         "psnr_roadlens": float(psnr(x, ours).mean()),
                         "psnr_package_default": float(psnr(x, pkg).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--classes", type=int, nargs="+", default=[2, 14, 25, 38],
                    help="GTSRB class ids to draw one example each from")
    ap.add_argument("--stats-n", type=int, default=300, help="test images used for the PSNR table (0 = skip)")
    ap.add_argument("-o", "--override", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="config override, repeatable")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed = cfg["corruptions"]["seed"]
    x, y = load_gtsrb_cfg(cfg, data_root(cfg), "test")
    out = REPO_ROOT / "results" / "corruptions"
    out.mkdir(parents=True, exist_ok=True)
    names = corruption_names(cfg)

    rng = np.random.RandomState(0)
    picks = [int(rng.choice(np.flatnonzero(y == c))) for c in args.classes]
    for k, i in enumerate(picks):
        severity_grid(x[i], names, seed, f"GTSRB-C at {x.shape[1]} px - test image {i} ({GTSRB_SHORT[y[i]]})",
                      out / f"grid_{k}.pdf")
    severity_grid(x[picks[0]], cfg["corruptions"]["val"], seed,
                  "validation-C corruptions (held out from the test set)", out / "grid_val_c.pdf")
    comparison_grid(x[picks[0]], names, seed, out / "package_vs_roadlens.pdf")
    log.info(f"figures -> {out}")

    if args.stats_n:
        sub = x[rng.choice(len(x), args.stats_n, replace=False)]
        df = stats(sub, names + cfg["corruptions"]["val"], seed)
        df.to_csv(out / "corruption_stats.csv", index=False, float_format="%.2f")
        print(df.pivot(index="corruption", columns="severity", values="psnr_roadlens").round(1).to_string())


if __name__ == "__main__":
    main()
