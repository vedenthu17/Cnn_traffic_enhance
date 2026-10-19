"""Contact sheet for building / auditing a cross-dataset class map.

    python scripts/check_mapping.py --dataset btsd

One row per source class: its mapped GTSRB class (one GTSRB example, blank if unmapped) followed by
random examples of the source class. Writes results/mappings/<dataset>_mapping.pdf (+ .png pages).
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roadlens.data.classes import GTSRB_SHORT
from roadlens.data.external import load_external
from roadlens.data.gtsrb import load_gtsrb_cfg
from roadlens.utils import REPO_ROOT, data_root, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", required=True, choices=["btsd", "ctsd"])
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--rows-per-page", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root, d = data_root(cfg), cfg["data"]
    x, y, y_src = load_external(root, args.dataset, d["image_size"], d["crop_roi"],
                                mapped_only=False, allow_unverified=True)
    gx, gy = load_gtsrb_cfg(cfg, root, "test")
    rng = np.random.RandomState(0)
    out = REPO_ROOT / "results" / "mappings"
    out.mkdir(parents=True, exist_ok=True)

    classes = np.unique(y_src)
    k = args.examples
    pages = [classes[i:i + args.rows_per_page] for i in range(0, len(classes), args.rows_per_page)]
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out / f"{args.dataset}_mapping.pdf") as pdf:
        for pi, page in enumerate(pages):
            fig, axes = plt.subplots(len(page), k + 1, figsize=((k + 1) * 0.95 + 1.6, len(page) * 0.95), squeeze=False)
            for r, c in enumerate(page):
                idx = np.flatnonzero(y_src == c)
                g = int(y[idx[0]])
                for ax in axes[r]:
                    ax.set_xticks([]); ax.set_yticks([])
                label = f"src {c} (n={len(idx)})\n-> " + (f"{g} {GTSRB_SHORT[g]}" if g >= 0 else "unmapped")
                axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
                if g >= 0:
                    axes[r, 0].imshow(gx[rng.choice(np.flatnonzero(gy == g))])
                else:
                    axes[r, 0].axis("off")
                    axes[r, 0].text(-0.1, 0.5, label, ha="right", va="center", fontsize=7,
                                    transform=axes[r, 0].transAxes)
                for j, i in enumerate(rng.choice(idx, min(k, len(idx)), replace=False)):
                    axes[r, 1 + j].imshow(x[i])
                for j in range(min(k, len(idx)), k):
                    axes[r, 1 + j].axis("off")
            axes[0, 0].set_title("GTSRB", fontsize=7)
            fig.tight_layout(h_pad=0.1, w_pad=0.1)
            pdf.savefig(fig)
            fig.savefig(out / f"{args.dataset}_mapping_p{pi}.png", dpi=100)
            plt.close(fig)
    print(f"wrote {out / (args.dataset + '_mapping.pdf')}")


if __name__ == "__main__":
    main()
