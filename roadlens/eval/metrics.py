"""Classification, calibration and corruption-robustness metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

N_BINS = 15


def confusion_matrix(pred: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """[k, k] counts, rows = true class, columns = predicted class."""
    return torch.bincount(y * k + pred, minlength=k * k).view(k, k)


def macro_f1(conf) -> float:
    """Unweighted mean F1 over the classes that occur in the ground truth.

    (On GTSRB that is all 43 classes; on a cross-dataset test set it is the shared classes, and
    predictions of an absent class simply count as errors of the true class.)
    """
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diag(conf)
    support, predicted = conf.sum(1), conf.sum(0)
    present = support > 0
    f1 = 2 * tp[present] / np.maximum(support[present] + predicted[present], 1e-12)
    return float(f1.mean()) if present.any() else float("nan")


def reliability_bins(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
    """[n_bins, 3] = (count, sum of confidence, number correct) over equal-width bins (a, b]."""
    idx = torch.clamp(torch.ceil(confidence * n_bins).long() - 1, 0, n_bins - 1)
    out = torch.zeros(n_bins, 3, dtype=torch.float64)
    out[:, 0] = torch.bincount(idx, minlength=n_bins).double()
    out[:, 1] = torch.bincount(idx, weights=confidence.double(), minlength=n_bins)
    out[:, 2] = torch.bincount(idx, weights=correct.double(), minlength=n_bins)
    return out


def ece_from_bins(bins) -> float:
    """Expected calibration error: sum_b (n_b / N) * |accuracy_b - confidence_b|."""
    bins = np.asarray(bins, dtype=np.float64)
    n = bins[:, 0].sum()
    if n == 0:
        return float("nan")
    return float(np.abs(bins[:, 2] - bins[:, 1]).sum() / n)


def summarize_logits(logits: torch.Tensor, y: torch.Tensor, num_classes: int, shared=None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Metrics of one test set from its logits (CPU tensors).

    Returns (scalars, confusion [K, K], reliability bins [15, 3]). With `shared` (class ids present
    in a cross-dataset test set), `acc_shared` is the accuracy when the arg-max is restricted to them.
    """
    logits = logits.float()
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred == y
    cm = confusion_matrix(pred, y, num_classes).numpy()
    bins = reliability_bins(conf, correct).numpy()
    out = {
        "n": int(len(y)),
        "acc": float(correct.float().mean()),
        "macro_f1": macro_f1(cm),
        "ece": ece_from_bins(bins),
        "nll": float(F.cross_entropy(logits, y)),
    }
    if shared is not None:
        mask = torch.full((num_classes,), float("-inf"))
        mask[torch.as_tensor(shared)] = 0.0
        out["acc_shared"] = float(((logits + mask).argmax(dim=1) == y).float().mean())
    return out, cm, bins


# ----------------------------------------------------------------------------------------
# Corruption robustness summaries
# ----------------------------------------------------------------------------------------
def corruption_summary(df: pd.DataFrame, normalizer: pd.DataFrame | None = None) -> dict:
    """Summary of one model/seed.

    df: rows of one evaluation with columns corruption, severity, acc, macro_f1, ece; the clean test
        set is the row with corruption == 'clean'.
    normalizer: the same kind of table for the reference model (SmallCNN), averaged over its seeds.
        mCE     = 100 * mean_c [ sum_s E(c, s) / sum_s E_ref(c, s) ]
        rel_mce = 100 * mean_c [ sum_s (E(c, s) - E_clean) / sum_s (E_ref(c, s) - E_ref_clean) ]
    (Hendrycks & Dietterich 2019, with SmallCNN in the role of AlexNet.)
    """
    clean = df[df.corruption == "clean"].iloc[0]
    corr = df[df.corruption != "clean"]
    out = {
        "clean_acc": 100 * clean.acc, "clean_f1": 100 * clean.macro_f1, "clean_ece": 100 * clean.ece,
        "corr_acc": 100 * corr.acc.mean(), "corr_f1": 100 * corr.macro_f1.mean(), "corr_ece": 100 * corr.ece.mean(),
    }
    if normalizer is not None:
        ref_clean_err = 1 - float(normalizer[normalizer.corruption == "clean"].acc.iloc[0])
        ref = normalizer[normalizer.corruption != "clean"].groupby("corruption").acc.mean()
        err = 1 - corr.groupby("corruption").acc.mean()
        ref_err = (1 - ref).reindex(err.index)
        ce = err / ref_err.clip(lower=1e-4)
        rel = (err - (1 - clean.acc)) / (ref_err - ref_clean_err).clip(lower=1e-4)
        out["mce"] = 100 * float(ce.mean())
        out["rel_mce"] = 100 * float(rel.mean())
    return out


def mean_std(df: pd.DataFrame, by, cols=None) -> pd.DataFrame:
    """Mean and standard deviation over seeds: one row per group, columns <col>_mean / <col>_std
    (sample std, ddof=1; NaN when there is a single seed) plus n_seeds."""
    by = [by] if isinstance(by, str) else list(by)
    cols = cols or [c for c in df.columns if c not in by and c != "seed" and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(by, sort=False)
    out = g.size().rename("n_seeds").to_frame()
    mean, std = g[cols].mean(), g[cols].std(ddof=1)
    for c in cols:
        out[f"{c}_mean"], out[f"{c}_std"] = mean[c], std[c]
    return out.reset_index()
