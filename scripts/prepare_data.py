"""Download datasets and cache them at the working resolution.

    python scripts/prepare_data.py                       # GTSRB only
    python scripts/prepare_data.py --datasets gtsrb btsd ctsd
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from roadlens.data.external import load_external, prepare_external
from roadlens.data.gtsrb import load_gtsrb_cfg, prepare_gtsrb
from roadlens.utils import data_root, get_logger, load_config

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--datasets", nargs="+", default=["gtsrb"], choices=["gtsrb", "btsd", "ctsd"])
    ap.add_argument("-o", "--override", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="config override, repeatable (e.g. -o data.image_size=32)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    root, d = data_root(cfg), cfg["data"]
    log.info(f"data root: {root}")

    if "gtsrb" in args.datasets:
        prepare_gtsrb(root, d["image_size"], d["crop_roi"])
        for split in ("train", "val", "test"):
            x, y = load_gtsrb_cfg(cfg, root, split)
            counts = np.bincount(y, minlength=d["num_classes"])
            log.info(f"gtsrb {split:5s}: {x.shape}  classes={len(np.unique(y))}  "
                     f"min/max per class={counts.min()}/{counts.max()}")
    for name in ("btsd", "ctsd"):
        if name in args.datasets:
            prepare_external(root, name, d["image_size"], d["crop_roi"])
            x, y, y_src = load_external(root, name, d["image_size"], d["crop_roi"],
                                        mapped_only=False, allow_unverified=True)
            log.info(f"{name}: {len(x)} images, {len(np.unique(y_src))} source classes; "
                     f"{int((y >= 0).sum())} images in {len(np.unique(y[y >= 0]))} classes shared with GTSRB")


if __name__ == "__main__":
    main()
