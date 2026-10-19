"""Post-hoc temperature scaling (Guo et al. 2017).

One scalar T > 0 per trained model: predictions become softmax(logits / T). The arg-max, and with
it accuracy and macro-F1, never changes; only the confidence does. T is fitted by minimising the
negative log-likelihood on held-out validation data, never on test data.

The models here are trained with label smoothing, which caps their confidence near 0.9 while
their accuracy is far higher, so they need T < 1 (sharpening), the opposite of the usual case.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def nll_at(logits: torch.Tensor, y: torch.Tensor, temperature: float) -> float:
    return float(F.cross_entropy(logits.double() / temperature, y))


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, lo: float = 0.02, hi: float = 50.0, iters: int = 60) -> float:
    """T in [lo, hi] that minimises the NLL of softmax(logits / T) on (logits, y).

    The NLL is convex in 1 / T, hence unimodal in log T: a golden-section search on log T finds
    the minimum to about 1e-6 relative precision without any learning rate to tune.
    """
    logits, y = logits.detach().double().cpu(), y.detach().long().cpu()
    if len(y) == 0:
        raise ValueError("cannot fit a temperature on an empty set")
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = math.log(lo), math.log(hi)
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll_at(logits, y, math.exp(c)), nll_at(logits, y, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll_at(logits, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll_at(logits, y, math.exp(d))
    return math.exp((a + b) / 2.0)
