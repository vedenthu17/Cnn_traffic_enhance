"""Metrics on hand-checkable inputs."""

import numpy as np
import pandas as pd
import pytest
import torch

from roadlens.eval import metrics as M
from roadlens.eval.tables import format_summary, pm


def test_confusion_and_macro_f1():
    y = torch.tensor([0, 0, 0, 1, 1, 2])
    pred = torch.tensor([0, 0, 1, 1, 2, 2])
    cm = M.confusion_matrix(pred, y, 4).numpy()
    assert cm.tolist() == [[2, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    # F1: class0 = 2*2/(3+2) = .8, class1 = 2*1/(2+2) = .5, class2 = 2*1/(1+2) = 2/3; class 3 absent -> ignored
    assert M.macro_f1(cm) == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)


def test_macro_f1_counts_predictions_of_absent_classes_as_errors():
    cm = np.array([[5, 0, 5], [0, 10, 0], [0, 0, 0]])     # class 2 never occurs but is predicted 5 times
    assert M.macro_f1(cm) == pytest.approx((2 * 5 / (10 + 5) + 1.0) / 2)


def test_ece():
    conf = torch.full((100,), 0.8)
    correct = torch.zeros(100, dtype=torch.bool)
    correct[:60] = True
    bins = M.reliability_bins(conf, correct).numpy()
    assert bins[:, 0].sum() == 100 and bins[11, 0] == 100           # 0.8 falls in (11/15, 12/15]
    assert M.ece_from_bins(bins) == pytest.approx(0.2, abs=1e-6)
    perfect = M.reliability_bins(torch.ones(10), torch.ones(10, dtype=torch.bool)).numpy()
    assert M.ece_from_bins(perfect) == pytest.approx(0.0, abs=1e-6)


def test_summarize_logits():
    logits = torch.tensor([[4.0, 0, 0], [0, 4.0, 0], [0, 4.0, 0], [0, 0, 4.0]])
    y = torch.tensor([0, 1, 2, 2])
    stats, cm, bins = M.summarize_logits(logits, y, 3, shared=[0, 2])
    assert stats["acc"] == pytest.approx(0.75) and stats["n"] == 4
    assert cm.sum() == 4 and bins[:, 0].sum() == 4
    # restricted to classes {0, 2}: sample 1 (true 1) can no longer be right, sample 2 (true 2) ties 0 vs 2 -> class 0
    assert stats["acc_shared"] == pytest.approx(0.5)


def _table(clean, accs):
    rows = [{"corruption": "clean", "severity": 0, "acc": clean, "macro_f1": clean, "ece": 0.0}]
    for name, per_sev in accs.items():
        rows += [{"corruption": name, "severity": s + 1, "acc": a, "macro_f1": a, "ece": 0.1} for s, a in enumerate(per_sev)]
    return pd.DataFrame(rows)


def test_mce_and_relative_mce():
    ref = _table(0.9, {"noise": [0.8, 0.6], "blur": [0.7, 0.5]})          # errors: noise .2/.4, blur .3/.5
    assert M.corruption_summary(ref, ref)["mce"] == pytest.approx(100.0)
    assert M.corruption_summary(ref, ref)["rel_mce"] == pytest.approx(100.0)
    model = _table(0.95, {"noise": [0.9, 0.8], "blur": [0.85, 0.75]})     # errors: noise .1/.2, blur .15/.25
    s = M.corruption_summary(model, ref)
    assert s["mce"] == pytest.approx(100 * 0.5)                           # exactly half the errors everywhere
    # relative: noise (0.15 - 0.05) / (0.3 - 0.1) = 0.5 ; blur (0.2 - 0.05) / (0.4 - 0.1) = 0.5
    assert s["rel_mce"] == pytest.approx(50.0)
    assert s["clean_acc"] == pytest.approx(95.0) and s["corr_acc"] == pytest.approx(100 * np.mean([0.9, 0.8, 0.85, 0.75]))


def test_mean_std_and_formatting():
    df = pd.DataFrame({"run": ["a", "a", "a", "b"], "seed": [0, 1, 2, 0], "acc": [1.0, 2.0, 3.0, 5.0]})
    out = M.mean_std(df, "run").set_index("run")
    assert out.loc["a", "acc_mean"] == pytest.approx(2.0) and out.loc["a", "acc_std"] == pytest.approx(1.0)
    assert out.loc["a", "n_seeds"] == 3 and np.isnan(out.loc["b", "acc_std"])
    assert pm(2.0, 1.0, 1) == "2.0 +- 1.0" and pm(5.0, float("nan"), 1) == "5.0"
    plain, tex = format_summary(out.reset_index().rename(columns={"run": "model"}), {"acc": ("Acc", 1, "max")})
    assert plain["Acc"].tolist() == ["2.0 +- 1.0", "5.0"]
    assert tex[1][1] == r"\textbf{5.0}" and "textbf" not in tex[0][1]
