"""Experiment 7 - the v2 approach, one mechanism per row.

    python scripts/exp7_improved.py

Rows (configs/experiments/exp7_improved.yaml): the v1 models, then per-image input standardisation,
view experts (median / Gaussian low-pass copies of the input), the stacked degradation-aware gate,
and post-hoc temperature scaling. Every row is evaluated on GTSRB-C and on the cross-dataset sets
that are prepared. A second table measures test-time BatchNorm with a source-statistics prior at
small batch sizes.

"Val-C (sel.)" is the model-selection metric (validation corruptions on the select half of the
validation split; no test corruption, no test image). Choose between rows with that column.

Writes to results/exp7_improved/:
    per_seed.csv                  all summary metrics per step and seed
    improved.csv/.txt/.tex        mean +- std over seeds (booktabs table)
    per_corruption.csv/.txt       GTSRB-C accuracy per step and corruption (mean over severities and seeds)
    severity5.csv/.txt            the same at severity 5 only
    selection.csv                 trained runs: selected epoch (0 = the untrained model), temperature
    adaptation.csv/.txt/.tex      test-time BN, with and without the prior, per batch size
"""

import json

import _bootstrap  # noqa: F401
import pandas as pd

from roadlens.data.loading import external_unavailable
from roadlens.eval.metrics import mean_std
from roadlens.eval.tables import over_seeds, pm, tex_escape, write_latex, write_table
from roadlens.experiment import Experiment, make_parser
from roadlens.runs import ensure_trained, evaluate_run, load_run_config, run_dir, temperature
from roadlens.utils import corruption_names, get_logger

log = get_logger()
COLUMNS = {
    "sel_valc_acc": ("Val-C (sel.)", 2, "max"),
    "clean_acc": ("Clean acc.", 2, "max"),
    "corr_acc": ("Corr. acc.", 2, "max"),
    "mce": ("mCE", 1, "min"),
    "corr_f1": ("Corr. F1", 2, "max"),
    "clean_ece": ("Clean ECE", 2, "min"),
    "corr_ece": ("Corr. ECE", 2, "min"),
}
METHODS = [("none", "No adaptation", None), ("bn", "Test-time BN", 0.0), ("bn_prior", "Test-time BN + prior", None)]


def step_eval(exp, step, suite, **kw):
    long, _ = exp.evaluate([step["run"]], suite, tta=step.get("tta"), calibrate=bool(step.get("calibrate", False)), **kw)
    return long.assign(step=step["label"])


def ladder(exp, stem: str = "improved", title: str = "The v2 approach"):
    """Evaluate every step of exp.cfg['steps']; write <stem>.*, per_corruption.* and severity5.*. Returns per_seed."""
    steps, names = exp.cfg["steps"], corruption_names(exp.cfg)
    long = pd.concat([step_eval(exp, s, "gtsrb_c") for s in steps], ignore_index=True)
    per_seed, _ = exp.summarize(long, "gtsrb_c", key="step")

    sel = pd.concat([step_eval(exp, s, "gtsrb_sel") for s in steps], ignore_index=True)
    sel = sel.assign(kind=sel.corruption.map({"clean": "sel_clean_acc", "val_c": "sel_valc_acc"}))
    sel = (100 * sel.groupby(["step", "seed", "kind"], sort=False).acc.mean()).unstack("kind").reset_index()
    per_seed = per_seed.merge(sel, on=["step", "seed"])

    columns = dict(COLUMNS)
    for dataset in exp.cfg.get("datasets", []):
        reason = external_unavailable(exp.cfg, dataset)
        if reason:
            log.warning(f"skipping {dataset}: {reason}")
            continue
        suite = f"{dataset}_c"
        ext = pd.concat([step_eval(exp, s, suite) for s in steps], ignore_index=True)
        summ, _ = exp.summarize(ext, suite, key="step")
        shared = (100 * ext[ext.corruption != "clean"].groupby(["step", "seed"], sort=False).acc_shared.mean())
        summ = summ.merge(shared.rename("corr_acc_shared").reset_index(), on=["step", "seed"])
        keep = {"corr_acc": f"{dataset}_corr_acc", "corr_acc_shared": f"{dataset}_corr_acc_shared", "mce": f"{dataset}_mce"}
        per_seed = per_seed.merge(summ[["step", "seed", *keep]].rename(columns=keep), on=["step", "seed"])
        columns[f"{dataset}_corr_acc"] = (f"{dataset.upper()} corr.", 2, "max")
        columns[f"{dataset}_corr_acc_shared"] = (f"{dataset.upper()} corr. (shared)", 2, "max")
        columns[f"{dataset}_mce"] = (f"{dataset.upper()} mCE", 1, "min")

    per_seed.to_csv(exp.out / "per_seed.csv", index=False, float_format="%.4f")
    summary = mean_std(per_seed, "step")
    table = write_table(exp.out, stem, summary, columns, label_col="step", label_header="Configuration",
                        caption=f"{title}, one mechanism per row (\\%, {over_seeds(len(exp.seeds))}). "
                                f"Val-C (sel.): accuracy under the validation corruptions on the select half of the "
                                f"validation split, the model-selection metric. Corr.: mean over 14 corruptions "
                                f"$\\times$ 5 severities. mCE is normalised by "
                                f"{tex_escape(exp.label(exp.cfg['normalizer']))}.",
                        label=f"tab:{stem}")
    print(table.to_string(index=False))

    corr = long[long.corruption != "clean"]
    order = [s["label"] for s in steps]
    for part, rows in (("per_corruption", corr), ("severity5", corr[corr.severity == corr.severity.max()])):
        wide = (100 * rows.groupby(["step", "corruption"], sort=False).acc.mean()).unstack("corruption").reindex(order)[names]
        wide.to_csv(exp.out / f"{part}.csv", float_format="%.2f")
        (exp.out / f"{part}.txt").write_text(wide.round(1).to_string() + "\n", encoding="utf-8")
    return per_seed, long


def selection(exp):
    """What model selection did: the epoch kept for every trained run, and the fitted temperatures."""
    rows = []
    for seed in exp.seeds:
        for step in exp.cfg["steps"]:
            cfg = load_run_config(step["run"], exp.profile)
            done = json.loads((run_dir(cfg, seed) / "done.json").read_text())
            rows.append({"step": step["label"], "run": step["run"], "seed": seed,
                         "epochs_run": done.get("epochs_run"), "best_epoch": done.get("best_epoch"),
                         "best_val_c_acc": done.get("best_val_c_acc"),
                         "temperature": temperature(step["run"], seed, exp.profile, exp.device)
                         if step.get("calibrate") else None})
    pd.DataFrame(rows).to_csv(exp.out / "selection.csv", index=False, float_format="%.4f")


def adaptation(exp):
    cfg = exp.cfg
    sizes, sevs, prior = list(cfg["batch_sizes"]), list(cfg["severities"]), float(cfg["prior"])
    n_per_set, n_bs1 = int(cfg["n_per_set"]), int(cfg.get("n_per_set_bs1", cfg["n_per_set"]))
    frames = []
    for seed in exp.seeds:
        for run in cfg["adaptation_runs"]:
            ensure_trained(run, seed, exp.profile, exp.device, exp.wandb)
            for method, _, p in METHODS:
                for bs in sizes:
                    tta = None if method == "none" else {"method": "bn", "batch_size": bs, "prior": prior if p is None else p}
                    df, _ = evaluate_run(run, seed, "gtsrb_c", exp.profile, exp.device, tta=tta, severities=sevs,
                                         n_per_set=n_bs1 if bs == 1 else n_per_set)
                    frames.append(df.assign(run=run, seed=seed, method=method, batch_size=bs))
    long = pd.concat(frames, ignore_index=True)
    corr = long[long.corruption != "clean"]
    per_seed = (100 * corr.groupby(["run", "method", "batch_size", "seed"], sort=False).acc.mean()).reset_index()
    summary = mean_std(per_seed, ["run", "method", "batch_size"], ["acc"])
    summary.insert(1, "model", summary.run.map(exp.label))
    summary.to_csv(exp.out / "adaptation.csv", index=False, float_format="%.4f")

    label = {m: (f"{text} (N = {prior:g})" if m == "bn_prior" else text) for m, text, _ in METHODS}
    rows, tex_rows = [], []
    for (run, method), g in summary.groupby(["run", "method"], sort=False):
        g = g.set_index("batch_size")
        rows.append({"Model": exp.label(run), "Method": label[method],
                     **{f"bs={bs}": pm(g.acc_mean[bs], g.acc_std[bs], 2) for bs in sizes}})
        tex_rows.append([tex_escape(exp.label(run)), label[method]] + [pm(g.acc_mean[bs], g.acc_std[bs], 2, tex=True) for bs in sizes])
    table = pd.DataFrame(rows)
    (exp.out / "adaptation.txt").write_text(table.to_string(index=False) + "\n", encoding="utf-8")
    write_latex(exp.out / "adaptation.tex", ["Model", "Method"] + [f"Batch {bs}" for bs in sizes], tex_rows,
                align="ll" + "c" * len(sizes),
                caption=f"Test-time BatchNorm statistics with and without a source prior of {prior:g} pseudo-samples: "
                        f"mean accuracy over 14 corruptions at severity {', '.join(map(str, sevs))} (\\%, "
                        f"{over_seeds(len(exp.seeds))}); {n_per_set} images per corruption ({n_bs1} for batch size 1).",
                label="tab:tta-prior")
    print("\n" + table.to_string(index=False))


def main():
    exp = Experiment(make_parser(__doc__, "configs/experiments/exp7_improved.yaml").parse_args(), "exp7_improved")
    ladder(exp)
    selection(exp)
    adaptation(exp)


if __name__ == "__main__":
    main()
