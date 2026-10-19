"""Train one named run (configs/runs/<name>.yaml) for one or more seeds.

    python scripts/train.py --run resnet18_base --seeds 0 1 2
    python scripts/train.py --run fusion_full --seeds 0          # trains its two branches first if needed

Finished runs are skipped and interrupted ones resume from their last completed epoch. The experiment
scripts call the same code, so running this first is optional.
"""

import argparse

import _bootstrap  # noqa: F401

from roadlens.runs import ensure_trained
from roadlens.utils import get_device, get_logger

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+", help="run name(s): file stems in configs/runs/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--profile", default=None, help="smoke | pilot (configs/profiles/); default: full scale")
    ap.add_argument("--device", default=None)
    ap.add_argument("--wandb", action="store_true", help="also log to Weights & Biases")
    args = ap.parse_args()
    device = get_device(args.device)
    for name in args.run:
        for seed in args.seeds:
            info = ensure_trained(name, seed, args.profile, device, args.wandb)
            if info.get("skipped"):
                log.info(f"{name} seed {seed}: nothing to train")
            else:
                log.info(f"{name} seed {seed}: best val-C {100 * info['best_val_c_acc']:.2f}% at epoch "
                         f"{info['best_epoch']}/{info['epochs_run']}, {info['train_time_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
