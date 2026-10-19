"""Experiment 3 - cumulative ablation on GTSRB-C:
baseline -> + mixing -> + JSD -> + Fourier/PRIME ops -> + fusion -> + TENT.

    python scripts/exp3_ablation.py

Writes to results/exp3_ablation/:
    per_seed.csv              summary metrics per step and seed
    ablation.csv/.txt/.tex    mean +- std over seeds (booktabs table)
    per_severity.csv          accuracy per step / corruption / severity (mean +- std over seeds)
"""

import _bootstrap  # noqa: F401
import pandas as pd

from roadlens.eval.metrics import mean_std
from roadlens.eval.tables import over_seeds, tex_escape, write_table
from roadlens.experiment import Experiment, make_parser

COLUMNS = {
    "clean_acc": ("Clean acc.", 2, "max"),
    "corr_acc": ("Corr. acc.", 2, "max"),
    "mce": ("mCE", 1, "min"),
    "rel_mce": ("Rel. mCE", 1, "min"),
    "corr_f1": ("Corr. F1", 2, "max"),
    "corr_ece": ("Corr. ECE", 2, "min"),
}


def main():
    exp = Experiment(make_parser(__doc__, "configs/experiments/exp3_ablation.yaml").parse_args(), "exp3_ablation")
    frames = []
    for step in exp.cfg["steps"]:
        long, _ = exp.evaluate([step["run"]], "gtsrb_c", tta=step.get("tta"))
        frames.append(long.assign(step=step["label"]))
    long = pd.concat(frames, ignore_index=True)
    mean_std(long, ["step", "corruption", "severity"], ["acc", "macro_f1", "ece"]).to_csv(
        exp.out / "per_severity.csv", index=False, float_format="%.5f")

    per_seed, summary = exp.summarize(long, "gtsrb_c", key="step")
    per_seed.to_csv(exp.out / "per_seed.csv", index=False, float_format="%.4f")
    table = write_table(exp.out, "ablation", summary, COLUMNS, label_col="step", label_header="Configuration",
                        caption=f"Cumulative ablation on GTSRB-C (\\%, {over_seeds(len(exp.seeds))}). Each row adds one "
                                f"component to the row above. mCE is normalised by "
                                f"{tex_escape(exp.label(exp.cfg['normalizer']))}.",
                        label="tab:ablation")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
