"""Generate GTSRB-C (14 corruptions x 5 severities of the test set) and validation-C, once.

    python scripts/make_gtsrb_c.py                 # full: 70 sets, ~10.9 GB at 64 px
    python scripts/make_gtsrb_c.py --profile pilot # the 2,000-image subset used by --profile pilot

Re-running skips finished sets. Cross-dataset corrupted sets: add --dataset btsd / ctsd.
"""

import argparse
import time

import _bootstrap  # noqa: F401

from roadlens.data.corrupted import generate_corrupted_sets, generate_val_c, val_c_path
from roadlens.data.gtsrb import load_gtsrb_cfg
from roadlens.data.loading import corrupted_dir, load_test
from roadlens.utils import apply_profile, corruption_names, data_root, get_logger, load_config

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", default="gtsrb", choices=["gtsrb", "btsd", "ctsd"])
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores (max 8)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test images (same as data.test_subset=N)")
    ap.add_argument("--profile", default=None, help="smoke | pilot: generate that profile's test subset")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-val-c", action="store_true")
    ap.add_argument("--val-c-only", action="store_true", help="only build validation-C (needed before training)")
    ap.add_argument("-o", "--override", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="config override, repeatable (e.g. -o corruptions.seed=1)")
    args = ap.parse_args()
    cfg = apply_profile(load_config(args.config, args.overrides), args.profile)
    if args.limit is not None:
        cfg["data"]["test_subset"] = args.limit
    root, c = data_root(cfg), cfg["corruptions"]

    if args.val_c_only:
        xv, yv = load_gtsrb_cfg(cfg, root, "val")
        generate_val_c(xv, yv, val_c_path(root, cfg["data"]["image_size"]), c["val"], c["val_severities"],
                       c["seed"], overwrite=args.overwrite)
        return

    x, y = load_test(cfg, args.dataset)
    out_dir = corrupted_dir(cfg, args.dataset)
    names = corruption_names(cfg)
    gb = len(names) * len(c["severities"]) * x.nbytes / 1e9
    log.info(f"{args.dataset}-C: {len(names)} corruptions x {len(c['severities'])} severities x "
             f"{len(x)} images -> {out_dir} ({gb:.1f} GB)")
    t0 = time.time()
    generate_corrupted_sets(x, y, out_dir, names, c["severities"], c["seed"],
                            workers=args.workers, overwrite=args.overwrite)
    log.info(f"done in {(time.time() - t0) / 60:.1f} min")

    if args.dataset == "gtsrb" and not args.skip_val_c:
        # validation-C is always built from the full validation split; debug runs take a prefix of it
        xv, yv = load_gtsrb_cfg(cfg, root, "val")
        generate_val_c(xv, yv, val_c_path(root, cfg["data"]["image_size"]), c["val"], c["val_severities"],
                       c["seed"], overwrite=args.overwrite)


if __name__ == "__main__":
    main()
