"""Evaluate a model (optionally with test-time adaptation) on suites of test sets.

A suite is a list of test sets:
    gtsrb_clean   GTSRB train / val / test, clean
    gtsrb_sel     the select half of the validation split, clean and as validation-C (model selection)
    gtsrb_c       GTSRB test: clean + 14 corruptions x 5 severities
    btsd_c, ctsd_c  the same for a cross-dataset test set (labels mapped to GTSRB ids)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data.loading import as_float, load_corrupted_set, load_selection_data, load_split, load_test, u8_to_device
from ..utils import corruption_names
from .metrics import summarize_logits


@dataclass
class EvalSet:
    dataset: str
    split: str
    corruption: str            # "clean" or a corruption name
    severity: int              # 0 for clean
    load: Callable[[], tuple[np.ndarray, np.ndarray]]

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.split}/{self.corruption}/{self.severity}"


def build_sets(cfg: dict, suite: str, severities=None, corruptions=None) -> list[EvalSet]:
    if suite == "gtsrb_clean":
        return [EvalSet("gtsrb", s, "clean", 0, lambda s=s: load_split(cfg, s)) for s in ("train", "val", "test")]
    if suite == "gtsrb_sel":       # the select half of the validation split: the model-selection data
        return [EvalSet("gtsrb", "val_sel", "clean", 0, lambda: load_selection_data(cfg)[:2]),
                EvalSet("gtsrb", "val_sel", "val_c", 0, lambda: load_selection_data(cfg)[2:])]
    if not suite.endswith("_c"):
        raise ValueError(f"unknown evaluation suite: {suite!r}")
    dataset = suite[:-2]
    sets = [EvalSet(dataset, "test", "clean", 0, lambda: load_test(cfg, dataset))]
    for name in corruptions or corruption_names(cfg):
        for sev in severities or cfg["corruptions"]["severities"]:
            sets.append(EvalSet(dataset, "test", name, int(sev),
                                lambda name=name, sev=sev: load_corrupted_set(cfg, dataset, name, sev)))
    return sets


@torch.no_grad()
def predict_logits(model, x: np.ndarray, device, batch_size: int = 512, amp: bool = True) -> torch.Tensor:
    """Logits [N, K] (float32, CPU) of a uint8 NHWC array, in order, without adaptation."""
    model.eval()
    out = []
    for s in range(0, len(x), batch_size):
        xb = as_float(u8_to_device(x[s:s + batch_size], device))
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out.append(model(xb).float().cpu())
    return torch.cat(out)


def predict_stream(tta, x: np.ndarray, device, batch_size: int, seed: int = 0) -> torch.Tensor:
    """Run a test-time-adaptation method over `x` as an online stream of batches.

    The stream is shuffled with a fixed seed (a class-sorted stream would make batch statistics
    meaningless) and the method is reset first. Returns logits in the original order of `x`.
    """
    X = u8_to_device(np.asarray(x), device)
    perm = torch.from_numpy(np.random.RandomState(seed).permutation(len(X))).to(device)
    tta.reset()
    out = None
    for s in range(0, len(X), batch_size):
        idx = perm[s:s + batch_size]
        logits = tta.predict(as_float(X[idx]))
        if out is None:
            out = torch.empty(len(X), logits.shape[1], device=device)
        out[idx] = logits
    return out.cpu()


def evaluate_sets(predict: Callable[[np.ndarray], torch.Tensor], sets: list[EvalSet], num_classes: int,
                  n_per_set: int = 0, subset_seed: int = 0, desc: str = "eval"):
    """Evaluate every set. Returns (DataFrame with one row per set, confusions [S, K, K], bins [S, 15, 3]).

    `n_per_set` > 0 evaluates a fixed random subset of each set (same images for every corruption).
    """
    rows, confs, bins = [], [], []
    for es in tqdm(sets, desc=desc, leave=False):
        x, y = es.load()
        if n_per_set and n_per_set < len(x):
            idx = np.sort(np.random.RandomState(subset_seed).permutation(len(x))[:n_per_set])
            x, y = np.asarray(x[idx]), y[idx]
        yt = torch.from_numpy(np.asarray(y)).long()
        shared = np.unique(y) if es.dataset != "gtsrb" else None
        stats, cm, b = summarize_logits(predict(x), yt, num_classes, shared)
        rows.append({"dataset": es.dataset, "split": es.split, "corruption": es.corruption,
                     "severity": es.severity, **stats})
        confs.append(cm)
        bins.append(b)
    return pd.DataFrame(rows), np.stack(confs), np.stack(bins)
