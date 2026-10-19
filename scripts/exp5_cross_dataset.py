"""Experiment 5 - cross-dataset: models trained on GTSRB, tested on BTSD / CTSD (classes shared with
GTSRB only), clean and corrupted.

    python scripts/prepare_data.py --datasets btsd          # once
    python scripts/make_gtsrb_c.py --dataset btsd           # once (corrupted BTSD, 2.3 GB)
    python scripts/exp5_cross_dataset.py

The models keep all 43 outputs ("Acc."); "Acc. (shared)" restricts the arg-max to the classes that
exist in the target dataset. A dataset that is not prepared, or whose class map is not verified, is
skipped with a message. Writes to results/exp5_cross_dataset/:
    <dataset>_per_seed.csv, <dataset>_summary_per_seed.csv, <dataset>.csv/.txt/.tex
"""

import _bootstrap  # noqa: F401

from roadlens.data.external import load_mapping
from roadlens.data.loading import external_unavailable
from roadlens.eval.metrics import mean_std
from roadlens.eval.tables import over_seeds, tex_escape, write_table
from roadlens.experiment import Experiment, make_parser
from roadlens.utils import get_logger

log = get_logger()
COLUMNS = {
    "clean_acc": ("Clean acc.", 2, "max"),
    "clean_acc_shared": ("Clean acc. (shared)", 2, "max"),
    "clean_f1": ("Clean F1", 2, "max"),
    "corr_acc": ("Corr. acc.", 2, "max"),
    "corr_acc_shared": ("Corr. acc. (shared)", 2, "max"),
    "mce": ("mCE", 1, "min"),
    "corr_ece": ("Corr. ECE", 2, "min"),
}


def main():
    exp = Experiment(make_parser(__doc__, "configs/experiments/exp5_cross_dataset.yaml").parse_args(),
                     "exp5_cross_dataset")
    runs, done = exp.cfg["runs"], []
    for dataset in exp.cfg["datasets"]:
        reason = external_unavailable(exp.cfg, dataset)
        if reason:
            log.warning(f"skipping {dataset}: {reason}")
            continue
        suite = f"{dataset}_c"
        long, _ = exp.evaluate(runs, suite)
        long["model"] = long.run.map(exp.label)
        long.to_csv(exp.out / f"{dataset}_per_seed.csv", index=False, float_format="%.5f")
        per_seed, _ = exp.summarize(long, suite)
        shared = long.assign(kind=long.corruption.eq("clean").map({True: "clean_acc_shared", False: "corr_acc_shared"}))
        shared = (100 * shared.groupby(["run", "seed", "kind"], sort=False).acc_shared.mean()).unstack("kind").reset_index()
        per_seed = per_seed.merge(shared, on=["run", "seed"])
        per_seed.insert(1, "model", per_seed.run.map(exp.label))
        per_seed.to_csv(exp.out / f"{dataset}_summary_per_seed.csv", index=False, float_format="%.4f")
        summary = mean_std(per_seed, ["run", "model"])
        n_img = int(long[long.corruption == "clean"].n.iloc[0])
        n_cls = len(set(load_mapping(dataset)["map"].values()))
        table = write_table(exp.out, dataset, summary, COLUMNS,
                            caption=f"Cross-dataset transfer GTSRB $\\rightarrow$ {dataset.upper()} ({n_img} test images "
                                    f"of the {n_cls} classes shared with GTSRB; \\%, {over_seeds(len(exp.seeds))}). "
                                    f"No {dataset.upper()} image is used for training. mCE is normalised by "
                                    f"{tex_escape(exp.label(exp.cfg['normalizer']))} on the same data.",
                            label=f"tab:cross-{dataset}")
        print(f"\n{dataset.upper()}\n" + table.to_string(index=False))
        done.append(dataset)
    if not done:
        log.warning("no cross-dataset test set was available; nothing written")


if __name__ == "__main__":
    main()
