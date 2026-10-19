"""Config-level access to the cached arrays, plus the GPU batch helpers.

Images stay uint8 NHWC on disk / in RAM and become float NCHW in [0, 1] on the device. The whole
training set lives on the GPU (430 MB), so there are no DataLoader workers: shuffling and all
augmentations run batched on the device and are seeded through torch's global generator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..utils import corruption_names, data_root
from .corrupted import load_corrupted, load_val_c
from .external import load_external
from .gtsrb import halve_by_track, load_gtsrb_cfg


def _subset(cfg: dict, split: str, n_total: int, spread: bool = False) -> np.ndarray | None:
    """Indices of the reduced subset for `split`, or None for the full split.

    train: random. val and class-sorted sets (`spread`): evenly spaced, so every class is covered.
    GTSRB test (already in random order): a prefix.
    """
    n = int(cfg["data"].get(f"{split}_subset", 0) or 0)
    if not n or n >= n_total:
        return None
    if split == "train":
        return np.sort(np.random.RandomState(0).permutation(n_total)[:n])
    if split == "val" or spread:
        return np.linspace(0, n_total - 1, n).astype(np.int64)
    return np.arange(n)


def load_split(cfg: dict, split: str):
    """(x uint8 NHWC, y int64) of GTSRB train / val / test, honouring data.<split>_subset."""
    x, y = load_gtsrb_cfg(cfg, data_root(cfg), split)
    idx = _subset(cfg, split, len(x))
    return (x, y) if idx is None else (x[idx], y[idx])


def load_val_c_split(cfg: dict):
    """(x, y, corruption_id, severity) of validation-C, aligned with load_split(cfg, 'val')."""
    x, y, cid, sev = load_val_c(data_root(cfg), cfg["data"]["image_size"])
    idx = _subset(cfg, "val", len(x))
    return (x, y, cid, sev) if idx is None else (x[idx], y[idx], cid[idx], sev[idx])


def stack_split(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Indices (fit, select) into load_split(cfg, 'val'): two halves that share no physical sign.

    The fit half is for training a combiner on images its base models never saw (stacked
    generalisation); the select half is for choosing and calibrating it. The split depends only on
    the data settings, so every run and every evaluation sees the same two halves.
    """
    _, y, tracks = load_gtsrb_cfg(cfg, data_root(cfg), "val", with_tracks=True)
    idx = _subset(cfg, "val", len(y))
    if idx is not None:
        y, tracks = y[idx], tracks[idx]
    return halve_by_track(y, tracks, seed=int(cfg["data"]["split_seed"]))


def load_stack_data(cfg: dict):
    """(x fit, y fit, x select, y select, x select-C, y select-C) for a run with train.fit_split = val.

    Fit set: the clean fit half plus its validation-C copies. Select sets: the clean select half and
    its validation-C copies. No test image and no test corruption is involved.
    """
    x, y = load_split(cfg, "val")
    xc, yc, _, _ = load_val_c_split(cfg)
    fit, sel = stack_split(cfg)
    return (np.concatenate([x[fit], xc[fit]]), np.concatenate([y[fit], yc[fit]]),
            x[sel], y[sel], xc[sel], yc[sel])


def load_selection_data(cfg: dict):
    """(x, y, x-C, y-C): the select half of the validation split, clean and as validation-C."""
    return load_stack_data(cfg)[2:]


def load_test(cfg: dict, dataset: str = "gtsrb"):
    """Clean test set of `dataset` (labels in GTSRB ids), honouring data.test_subset."""
    if dataset == "gtsrb":
        return load_split(cfg, "test")
    d = cfg["data"]
    x, y, _ = load_external(data_root(cfg), dataset, d["image_size"], d["crop_roi"])
    idx = _subset(cfg, "test", len(x), spread=True)
    return (x, y) if idx is None else (x[idx], y[idx])


def corrupted_dir(cfg: dict, dataset: str = "gtsrb") -> Path:
    """Directory of the pre-generated corrupted test sets of `dataset`."""
    d = cfg["data"]
    name = f"{dataset}_c_s{d['image_size']}"
    n = int(d.get("test_subset", 0) or 0)
    return data_root(cfg) / (name + (f"_sub{n}" if n else ""))


def load_corrupted_set(cfg: dict, dataset: str, corruption: str, severity: int):
    """(x uint8 NHWC memory-mapped, y) of one corrupted test set."""
    return load_corrupted(corrupted_dir(cfg, dataset), corruption, severity)


def u8_to_device(x: np.ndarray, device) -> torch.Tensor:
    """uint8 NHWC numpy (possibly memory-mapped) -> uint8 NCHW tensor on `device`."""
    x = np.ascontiguousarray(x)
    if not x.flags.writeable:          # read-only memory maps: torch wants a writable buffer
        x = x.copy()
    return torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()


def corrupted_sets_missing(cfg: dict, dataset: str = "gtsrb") -> list[str]:
    """Corrupted test sets of `dataset` ('<corruption>/<severity>') that have not been generated yet."""
    needed = [f"{n}/{s}" for n in corruption_names(cfg) for s in cfg["corruptions"]["severities"]]
    manifest = corrupted_dir(cfg, dataset) / "manifest.json"
    done = set(json.loads(manifest.read_text())["done"]) if manifest.exists() else set()
    return [k for k in needed if k not in done]


def corrupted_fingerprint(cfg: dict, dataset: str = "gtsrb") -> str:
    """Fingerprint of the generated corrupted sets (clean data + seed + corruption parameters)."""
    manifest = corrupted_dir(cfg, dataset) / "manifest.json"
    return json.loads(manifest.read_text())["fingerprint"] if manifest.exists() else ""


def make_c_command(cfg: dict, dataset: str = "gtsrb") -> str:
    return ("python scripts/make_gtsrb_c.py" + ("" if dataset == "gtsrb" else f" --dataset {dataset}")
            + (f" --profile {cfg['profile']}" if cfg.get("profile") else ""))


def external_unavailable(cfg: dict, dataset: str) -> str | None:
    """Why a cross-dataset test set cannot be evaluated (None if it is ready)."""
    from .external import _cache_path, load_mapping

    d = cfg["data"]
    if not _cache_path(data_root(cfg), dataset, d["image_size"], d["crop_roi"]).exists():
        return f"not prepared (python scripts/prepare_data.py --datasets {dataset})"
    if not load_mapping(dataset).get("verified", False):
        return (f"its class map is not verified (python scripts/check_mapping.py --dataset {dataset}, "
                f"then set verified: true in roadlens/data/mappings/{dataset}_to_gtsrb.yaml)")
    missing = corrupted_sets_missing(cfg, dataset)
    if missing:
        return f"{len(missing)} corrupted sets are missing or still being generated ({make_c_command(cfg, dataset)})"
    return None


def as_float(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW -> float32 in [0, 1]."""
    return x_u8.float().div_(255.0)
