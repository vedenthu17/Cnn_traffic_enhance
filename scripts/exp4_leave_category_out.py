"""Experiment 4 - leave-category-out: does an augmentation choice tuned on some corruption
categories transfer to the others?

    python scripts/exp4_leave_category_out.py

For each direction (e.g. tune on noise+blur, test on weather+digital) the candidate op set with the
best mean accuracy on the tuning categories is selected, and its accuracy on the held-out categories
is compared with the baseline and with the candidate that would have been best on the held-out
categories themselves (the oracle). Writes to results/exp4_leave_category_out/:
    candidates.csv/.txt/.tex   accuracy of every candidate per corruption category
    selection.csv/.txt/.tex    selected vs. oracle vs. baseline on the held-out categories
"""

import _bootstrap  # noqa: F401
import pandas as pd

from roadlens.eval.metrics import mean_std
from roadlens.eval.tables import over_seeds, pm, tex_escape, write_latex, write_table
from roadlens.experiment import Experiment, make_parser
from roadlens.utils import corruption_categories


def main():
    exp = Experiment(make_parser(__doc__, "configs/experiments/exp4_leave_category_out.yaml").parse_args(),
                     "exp4_leave_category_out")
    cfg = exp.cfg
    baseline, candidates = cfg["baseline"], list(cfg["candidates"])
    runs = [baseline] + candidates
    long, _ = exp.evaluate(runs, "gtsrb_c")
    cats = corruption_categories(cfg)
    categories = list(cfg["corruptions"]["test"])
    corr = long[long.corruption != "clean"].assign(category=lambda d: d.corruption.map(cats))

    def group_acc(names):   # accuracy (%) over all test sets of the given categories, per run and seed
        return 100 * corr[corr.category.isin(names)].groupby(["run", "seed"], sort=False).acc.mean()

    per_seed = pd.DataFrame({c: group_acc([c]) for c in categories})
    per_seed["all"] = group_acc(categories)
    per_seed["clean"] = 100 * long[long.corruption == "clean"].set_index(["run", "seed"]).acc
    for i, d in enumerate(cfg["directions"]):
        per_seed[f"tune{i}"], per_seed[f"test{i}"] = group_acc(d["tune"]), group_acc(d["test"])
    per_seed = per_seed.reset_index()
    per_seed.insert(1, "model", per_seed.run.map(exp.label))
    per_seed.to_csv(exp.out / "per_seed.csv", index=False, float_format="%.4f")
    summary = mean_std(per_seed, ["run", "model"])
    n = len(exp.seeds)
    table = write_table(exp.out, "candidates", summary,
                        {"clean": ("Clean", 2, "max"), **{c: (c.capitalize(), 2, "max") for c in categories},
                         "all": ("All 14", 2, "max")},
                        caption=f"Accuracy (\\%) per corruption category for each augmentation op set "
                                f"(ResNet-18, {over_seeds(n)}).", label="tab:lco-candidates")
    print(table.to_string(index=False))

    s = summary.set_index("run")
    rows, tex_rows = [], []
    for i, d in enumerate(cfg["directions"]):
        cand = s.loc[candidates]
        selected = cand[f"tune{i}_mean"].idxmax()
        oracle = cand[f"test{i}_mean"].idxmax()

        def cell(run, tex=False):
            return pm(s.loc[run, f"test{i}_mean"], s.loc[run, f"test{i}_std"], 2, tex)

        row = {"tuned_on": "+".join(d["tune"]), "tested_on": "+".join(d["test"]),
               "selected": exp.label(selected), "selected_test_acc": cell(selected),
               "oracle": exp.label(oracle), "oracle_test_acc": cell(oracle),
               "baseline_test_acc": cell(baseline),
               "gap_to_oracle": round(float(s.loc[oracle, f"test{i}_mean"] - s.loc[selected, f"test{i}_mean"]), 3)}
        rows.append(row)
        tex_rows.append([tex_escape(row["tuned_on"]), tex_escape(row["tested_on"]), tex_escape(row["selected"]),
                         cell(selected, True), tex_escape(row["oracle"]), cell(oracle, True), cell(baseline, True),
                         f"{row['gap_to_oracle']:.2f}"])
    sel = pd.DataFrame(rows)
    sel.to_csv(exp.out / "selection.csv", index=False)
    (exp.out / "selection.txt").write_text(sel.to_string(index=False) + "\n", encoding="utf-8")
    write_latex(exp.out / "selection.tex",
                ["Tuned on", "Tested on", "Selected op set", "Acc.", "Oracle op set", "Acc.", "Baseline acc.", "Gap"],
                tex_rows, align="lllclccc",
                caption=f"Leave-category-out selection of the augmentation op set (\\%, {over_seeds(n)}). "
                        f"The op set is chosen on the tuning categories only; accuracies are on the held-out "
                        f"categories. Gap = oracle minus selected.", label="tab:lco")
    print(sel.to_string(index=False))


if __name__ == "__main__":
    main()
