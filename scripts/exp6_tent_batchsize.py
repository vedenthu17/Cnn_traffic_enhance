"""Experiment 6 - test-time adaptation vs. test batch size (1, 16, 64, 200).

    python scripts/exp6_tent_batchsize.py

Methods: none (source model), bn (test-time BatchNorm statistics), tent (TENT). Each corruption of
GTSRB-C at the configured severities is an independent, shuffled online stream; the model is reset
before each stream. Writes to results/exp6_tent_batchsize/:
    per_seed.csv              accuracy / ECE per model, seed, method, batch size and corruption
    sweep.csv/.txt/.tex       mean corruption accuracy per model x method x batch size
    tent_batchsize.pdf        the same as a figure
"""

import _bootstrap  # noqa: F401
import pandas as pd

from roadlens.eval.metrics import mean_std
from roadlens.eval.tables import over_seeds, pm, tex_escape, write_latex
from roadlens.experiment import Experiment, make_parser
from roadlens.runs import ensure_trained, evaluate_run
from roadlens.utils import get_logger

log = get_logger()
METHOD_LABEL = {"none": "No adaptation", "bn": "Test-time BN", "tent": "TENT"}


def main():
    exp = Experiment(make_parser(__doc__, "configs/experiments/exp6_tent_batchsize.yaml").parse_args(),
                     "exp6_tent_batchsize")
    cfg = exp.cfg
    sizes, sevs, n_per_set = list(cfg["batch_sizes"]), list(cfg["severities"]), int(cfg["n_per_set"])
    n_bs1 = int(cfg.get("n_per_set_bs1", n_per_set))
    frames = []
    for seed in exp.seeds:
        for run in cfg["runs"]:
            ensure_trained(run, seed, exp.profile, exp.device, exp.wandb)
            for method in cfg["methods"]:
                for bs in sizes:
                    tta = None if method == "none" else {"method": method, "batch_size": bs, "lr": cfg["lr"]}
                    n = n_bs1 if bs == 1 else n_per_set     # every method in a column sees the same images
                    try:
                        df, _ = evaluate_run(run, seed, "gtsrb_c", exp.profile, exp.device, tta=tta,
                                             severities=sevs, n_per_set=n)
                    except ValueError as err:       # model without BatchNorm
                        log.warning(f"{run}: {err}")
                        break
                    frames.append(df.assign(run=run, seed=seed, method=method, batch_size=bs))
    long = pd.concat(frames, ignore_index=True)
    long["model"] = long.run.map(exp.label)
    long.to_csv(exp.out / "per_seed.csv", index=False, float_format="%.5f")

    corr = long[long.corruption != "clean"]
    per_seed = (100 * corr.groupby(["run", "model", "method", "batch_size", "seed"], sort=False)[["acc", "ece"]].mean()).reset_index()
    summary = mean_std(per_seed, ["run", "model", "method", "batch_size"], ["acc", "ece"])
    summary.to_csv(exp.out / "sweep.csv", index=False, float_format="%.4f")

    rows, tex_rows = [], []
    for (run, method), g in summary.groupby(["run", "method"], sort=False):
        g = g.set_index("batch_size")
        cells = {bs: (g.acc_mean[bs], g.acc_std[bs]) for bs in sizes}
        rows.append({"Model": exp.label(run), "Method": METHOD_LABEL.get(method, method),
                     **{f"bs={bs}": pm(m, s, 2) for bs, (m, s) in cells.items()}})
        tex_rows.append([tex_escape(exp.label(run)), METHOD_LABEL.get(method, method)]
                        + [pm(m, s, 2, tex=True) for m, s in cells.values()])
    table = pd.DataFrame(rows)
    (exp.out / "sweep.txt").write_text(table.to_string(index=False) + "\n", encoding="utf-8")
    sev_txt = ", ".join(str(s) for s in sevs)
    write_latex(exp.out / "sweep.tex", ["Model", "Method"] + [f"Batch {bs}" for bs in sizes], tex_rows,
                align="ll" + "c" * len(sizes),
                caption=f"Test-time adaptation on GTSRB-C (severity {sev_txt}): mean accuracy over 14 corruptions (\\%, "
                        f"{over_seeds(len(exp.seeds))}) as a function of the test batch size. Online adaptation on "
                        f"{n_per_set or 'all'} images per corruption ({n_bs1} for batch size 1), model reset "
                        f"between corruptions.",
                label="tab:tta-batchsize")
    print(table.to_string(index=False))

    from roadlens.eval.plots import plot_tta_sweep
    plot_tta_sweep(summary, sizes, METHOD_LABEL, exp.out / "tent_batchsize.pdf",
                   title=f"GTSRB-C severity {sev_txt}: accuracy vs. test batch size")


if __name__ == "__main__":
    main()
