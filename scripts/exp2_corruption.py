"""Experiment 2 - full corruption evaluation on GTSRB-C (14 corruptions x 5 severities).

    python scripts/exp2_corruption.py

Writes to results/exp2_corruption/:
    per_seed.csv            accuracy, macro-F1, ECE, NLL for every model / seed / corruption / severity
    per_severity.csv        the same, mean +- std over seeds
    summary_per_seed.csv    clean acc, mean corruption acc, mCE, relative mCE, macro-F1, ECE per model and seed
    main.csv/.txt/.tex      main results table (mean +- std over seeds)
    per_corruption.csv/.txt/.tex   accuracy per corruption (mean over severities and seeds)
    per_category.csv        accuracy per corruption category
"""

import _bootstrap  # noqa: F401

from roadlens.eval.metrics import mean_std
from roadlens.eval.tables import over_seeds, tex_escape, write_latex, write_table
from roadlens.experiment import Experiment, make_parser
from roadlens.utils import corruption_categories, corruption_names

MAIN_COLUMNS = {
    "clean_acc": ("Clean acc.", 2, "max"),
    "corr_acc": ("Corr. acc.", 2, "max"),
    "mce": ("mCE", 1, "min"),
    "rel_mce": ("Rel. mCE", 1, "min"),
    "clean_f1": ("Clean F1", 2, "max"),
    "corr_f1": ("Corr. F1", 2, "max"),
    "clean_ece": ("Clean ECE", 2, "min"),
    "corr_ece": ("Corr. ECE", 2, "min"),
}
SHORT = {"gaussian_noise": "Gauss.", "impulse_noise": "Impulse", "shot_noise": "Shot", "motion_blur": "Motion",
         "zoom_blur": "Zoom", "defocus_blur": "Defocus", "snow": "Snow", "fog": "Fog", "frost": "Frost",
         "brightness": "Bright.", "contrast": "Contr.", "pixelate": "Pixel", "elastic_transform": "Elastic",
         "jpeg_compression": "JPEG"}


def main():
    exp = Experiment(make_parser(__doc__, "configs/experiments/exp2_corruption.yaml").parse_args(), "exp2_corruption")
    runs = exp.cfg["runs"]
    long, _ = exp.evaluate(runs, "gtsrb_c")
    long["model"] = long.run.map(exp.label)
    cols = ["run", "model", "seed", "corruption", "severity", "n", "acc", "macro_f1", "ece", "nll"]
    long[cols].to_csv(exp.out / "per_seed.csv", index=False, float_format="%.5f")
    mean_std(long, ["run", "model", "corruption", "severity"], ["acc", "macro_f1", "ece"]).to_csv(
        exp.out / "per_severity.csv", index=False, float_format="%.5f")

    per_seed, summary = exp.summarize(long, "gtsrb_c")
    per_seed.insert(1, "model", per_seed.run.map(exp.label))
    per_seed.to_csv(exp.out / "summary_per_seed.csv", index=False, float_format="%.4f")
    summary.insert(1, "model", summary.run.map(exp.label))
    n = len(exp.seeds)
    table = write_table(exp.out, "main", summary, MAIN_COLUMNS,
                        caption=f"GTSRB and GTSRB-C results (\\%, {over_seeds(n)}). Corr.: mean over 14 "
                                f"corruptions $\\times$ 5 severities. mCE and relative mCE are normalised by "
                                f"{tex_escape(exp.label(exp.cfg['normalizer']))} (= 100).",
                        label="tab:main")
    print(table.to_string(index=False))

    # accuracy per corruption: mean over severities, then over seeds
    names = corruption_names(exp.cfg)
    corr = long[long.corruption != "clean"]
    per_corr = (100 * corr.groupby(["run", "seed", "corruption"], sort=False).acc.mean()).reset_index()
    wide = per_corr.groupby(["run", "corruption"], sort=False).acc.mean().unstack("corruption").reindex(runs)[names]
    wide.insert(0, "model", [exp.label(r) for r in wide.index])
    wide.to_csv(exp.out / "per_corruption.csv", float_format="%.2f")
    (exp.out / "per_corruption.txt").write_text(wide.round(1).to_string(index=False) + "\n", encoding="utf-8")
    best = wide[names].round(1).max()
    rows = [[tex_escape(r.model)] + [(r"\textbf{%.1f}" if round(r[c], 1) == best[c] else "%.1f") % r[c] for c in names]
            for _, r in wide.iterrows()]
    write_latex(exp.out / "per_corruption.tex", ["Model"] + [SHORT.get(c, c) for c in names], rows,
                caption=f"Accuracy (\\%) per corruption on GTSRB-C, averaged over 5 severities and {n} seed(s).",
                label="tab:per-corruption", align="l" + "r" * len(names))

    cats = corruption_categories(exp.cfg)
    per_cat = per_corr.assign(category=per_corr.corruption.map(cats)).groupby(["run", "seed", "category"], sort=False).acc.mean()
    cat_summary = mean_std(per_cat.reset_index(), ["run", "category"], ["acc"])
    cat_summary.insert(1, "model", cat_summary.run.map(exp.label))
    cat_summary.to_csv(exp.out / "per_category.csv", index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
