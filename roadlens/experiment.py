"""Shared plumbing of the experiment scripts: CLI, config loading, train-if-needed + cached evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .eval.metrics import corruption_summary, mean_std
from .runs import ensure_trained, evaluate_run
from .utils import REPO_ROOT, apply_profile, get_device, get_logger, load_config, results_root

log = get_logger()


def make_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--profile", default=None, help="smoke | pilot (configs/profiles/); default: full scale")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override the seeds of the config")
    ap.add_argument("--device", default=None)
    ap.add_argument("--wandb", action="store_true", help="also log training to Weights & Biases")
    ap.add_argument("-o", "--override", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="experiment-config override, repeatable (e.g. -o n_per_set=500)")
    return ap


class Experiment:
    """One experiment = an experiment config + a results directory + access to runs."""

    def __init__(self, args, name: str):
        cfg = apply_profile(load_config(REPO_ROOT / args.config if not Path(args.config).is_absolute() else args.config,
                                        args.overrides), args.profile)
        if args.seeds:
            cfg["seeds"] = args.seeds
        self.cfg, self.profile, self.wandb = cfg, args.profile, args.wandb
        self.device = get_device(args.device)
        self.seeds = list(cfg["seeds"])
        self.out = results_root(cfg) / name
        self.out.mkdir(parents=True, exist_ok=True)
        log.info(f"{name}: seeds {self.seeds}, device {self.device}, results -> {self.out}"
                 + (f"  [profile: {self.profile}]" if self.profile else ""))

    def label(self, run: str) -> str:
        return self.cfg.get("labels", {}).get(run, run)

    def train(self, runs: list[str]) -> pd.DataFrame:
        """Make sure every (run, seed) is trained; one seed at a time so partial results arrive early."""
        rows = []
        for seed in self.seeds:
            for run in runs:
                info = ensure_trained(run, seed, self.profile, self.device, self.wandb)
                rows.append({"run": run, "seed": seed, **info})
        return pd.DataFrame(rows)

    def evaluate(self, runs: list[str], suite: str, **kw) -> tuple[pd.DataFrame, dict]:
        """Long table over runs x seeds (columns run, seed + one row per test set) and the cached arrays."""
        frames, arrays = [], {}
        for seed in self.seeds:
            for run in runs:
                ensure_trained(run, seed, self.profile, self.device, self.wandb)
                df, arr = evaluate_run(run, seed, suite, self.profile, self.device, **kw)
                frames.append(df.assign(run=run, seed=seed))
                arrays[(run, seed)] = arr
        return pd.concat(frames, ignore_index=True), arrays

    def normalizer_table(self, suite: str) -> pd.DataFrame:
        """Per-set accuracy of the mCE reference model (SmallCNN), averaged over seeds."""
        ref, _ = self.evaluate([self.cfg["normalizer"]], suite)
        return ref.groupby(["corruption", "severity"], as_index=False).acc.mean()

    def summarize(self, long: pd.DataFrame, suite: str, key: str = "run") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per-seed and mean+-std corruption summaries (clean acc, corruption acc, mCE, relative mCE, F1, ECE)."""
        norm = self.normalizer_table(suite)
        rows = []
        for (k, seed), g in long.groupby([key, "seed"], sort=False):
            rows.append({key: k, "seed": seed, **corruption_summary(g, norm)})
        per_seed = pd.DataFrame(rows)
        return per_seed, mean_std(per_seed, key)
