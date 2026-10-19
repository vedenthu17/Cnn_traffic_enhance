"""Config loading, paths, seeding and logging helpers."""

from __future__ import annotations

import copy
import logging
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike, overrides: Iterable[str] | None = None) -> dict:
    """Load a YAML config.

    `_base_: other.yaml` (path relative to the including file) is loaded first and deep-merged.
    `overrides` are `dotted.key=value` strings (value parsed as YAML), e.g. `data.image_size=32`.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        bases = [base] if isinstance(base, str) else list(base)
        merged: dict = {}
        for b in bases:
            merged = _deep_merge(merged, load_config(path.parent / b))
        cfg = _deep_merge(merged, cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def deep_merge(base: dict, update: dict) -> dict:
    return _deep_merge(base, update)


def _resolve_dir(value: str | os.PathLike) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_root(cfg: dict) -> Path:
    """Data directory: $ROADLENS_DATA if set, else cfg['data_root'] (relative to the repo root)."""
    return _resolve_dir(os.environ.get("ROADLENS_DATA") or cfg.get("data_root", "./datasets"))


def _profile_suffix(cfg: dict) -> str:
    return f"_{cfg['profile']}" if cfg.get("profile") else ""


def runs_root(cfg: dict) -> Path:
    """Checkpoints, logs and cached evaluations: $ROADLENS_RUNS if set, else cfg['runs_root'].
    A profile (smoke / pilot) gets its own directory, e.g. runs_pilot/."""
    root = os.environ.get("ROADLENS_RUNS") or cfg.get("runs_root", "./runs")
    return _resolve_dir(str(Path(root)) + _profile_suffix(cfg))


def scratch_root(cfg: dict) -> Path | None:
    """Optional folder for per-epoch training checkpoints: $ROADLENS_SCRATCH (None if unset).

    Set it to a local-disk folder when the runs directory is on a mounted Google Drive: every
    overwrite of a file there sends the previous version to the Drive trash, which keeps counting
    against the quota, so per-epoch checkpoints would silently fill the Drive. With a scratch folder
    only the final best weights are written to the runs directory, once per run.
    """
    root = os.environ.get("ROADLENS_SCRATCH")
    return _resolve_dir(str(Path(root)) + _profile_suffix(cfg)) if root else None


def results_root(cfg: dict) -> Path:
    """Tables and figures: cfg['results_root'] (relative to the repo root); results_pilot/ etc. for a profile."""
    return _resolve_dir(str(Path(cfg.get("results_root", "./results"))) + _profile_suffix(cfg))


PROFILE_DIR = REPO_ROOT / "configs" / "profiles"


def apply_profile(cfg: dict, profile: str | None) -> dict:
    """Overlay configs/profiles/<profile>.yaml (reduced-scale settings) on a config.

    smoke: tiny subsets, 2 epochs; checks that every script runs (minutes).
    pilot: full training set, one seed, shortened schedules, 2,000 test images; real but rough numbers.
    Runs and results of a profile live in their own *_<profile> directories, so they can never be
    mistaken for (or overwrite) the full experiments.
    """
    if not profile:
        return cfg
    path = PROFILE_DIR / f"{profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown profile {profile!r}: {path} does not exist")
    overlay = load_config(path)
    scale = overlay.get("train", {}).pop("epoch_scale", None)
    out = _deep_merge(cfg, overlay)
    if scale and "epochs" in cfg.get("train", {}):
        out["train"]["epochs"] = max(2, int(round(cfg["train"]["epochs"] * float(scale))))
    out["profile"] = profile
    return out


def corruption_names(cfg: dict) -> list[str]:
    """Flat list of the test corruptions, in category order."""
    return [name for names in cfg["corruptions"]["test"].values() for name in names]


def corruption_categories(cfg: dict) -> dict[str, str]:
    """corruption name -> category (noise / blur / weather / digital)."""
    return {name: cat for cat, names in cfg["corruptions"]["test"].items() for name in names}


def get_device(name: str | None = None):
    import torch

    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA). With `deterministic`, cuDNN is forced onto
    deterministic kernels; runs then repeat exactly on the same hardware and library versions."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "roadlens") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
