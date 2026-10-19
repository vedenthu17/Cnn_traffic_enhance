"""Experiment 1 - clean performance: train / test accuracy, macro-F1 and ECE of every model.

    python scripts/exp1_clean.py

Trains whatever is missing (resumable), then writes to results/exp1_clean/:
    per_seed.csv   one row per model and seed
    clean.csv/.txt/.tex   mean +- std over seeds (booktabs table)
"""

import _bootstrap  # noqa: F401
import pandas as pd

from roadlens.eval.metrics import mean_std
from roadlens.eval.tables import over_seeds, write_table
from roadlens.experiment import Experiment, make_parser


def main():
    exp = Experiment(make_parser(__doc__, "configs/experiments/exp1_clean.yaml").parse_args(), "exp1_clean")
    runs = exp.cfg["runs"]
    info = exp.train(runs).set_index(["run", "seed"])
    long, _ = exp.evaluate(runs, "gtsrb_clean")

    rows = []
    for (run, seed), g in long.groupby(["run", "seed"], sort=False):
        s = g.set_index("split")
        rows.append({"run": run, "model": exp.label(run), "seed": seed,
                     "params_m": info.loc[(run, seed), "n_params"] / 1e6,
                     "train_acc": 100 * s.acc["train"], "val_acc": 100 * s.acc["val"],
                     "test_acc": 100 * s.acc["test"], "test_f1": 100 * s.macro_f1["test"],
                     "test_ece": 100 * s.ece["test"],
                     "train_min": info.loc[(run, seed), "train_time_s"] / 60})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(exp.out / "per_seed.csv", index=False, float_format="%.4f")
    summary = mean_std(per_seed, ["run", "model"])
    table = write_table(exp.out, "clean", summary, {
        "params_m": ("Params (M)", 1, None),
        "train_acc": ("Train acc.", 2, None),
        "test_acc": ("Test acc.", 2, "max"),
        "test_f1": ("Test macro-F1", 2, "max"),
        "test_ece": ("Test ECE", 2, "min"),
    }, caption=f"Clean GTSRB performance (\\%, {over_seeds(len(exp.seeds))}).", label="tab:clean")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
