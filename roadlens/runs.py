"""Named training runs: configs/runs/<name>.yaml -> <runs_root>/<name>/seed<k>/.

Experiments never train or evaluate directly; they ask for (run, seed) and this module trains it
if needed (resumable), loads it, and caches every evaluation next to the checkpoint. That is what
lets the six experiments share models: e.g. `resnet18_full` is trained once per seed and reused by
the corruption table, the ablation, the leave-category-out study, the fusion and the TTA sweep.
"""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data.loading import corrupted_fingerprint, corrupted_sets_missing, load_selection_data, make_c_command
from .engine import train_run
from .eval.calibration import fit_temperature, nll_at
from .eval.evaluate import build_sets, evaluate_sets, predict_logits, predict_stream
from .models import build_model
from .tta import build_tta
from .utils import REPO_ROOT, apply_profile, get_device, get_logger, load_config, runs_root, scratch_root

log = get_logger()
RUN_CONFIG_DIR = REPO_ROOT / "configs" / "runs"


def load_run_config(name: str, profile: str | None = None) -> dict:
    path = RUN_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no run config {path}")
    cfg = load_config(path)
    cfg["name"] = name
    return apply_profile(cfg, profile)


def run_dir(cfg: dict, seed: int) -> Path:
    return runs_root(cfg) / cfg["name"] / f"seed{seed}"


def fingerprint(cfg: dict) -> str:
    """Hash of everything that determines the trained weights (including the branch runs of a fusion)."""
    train = {k: v for k, v in cfg["train"].items() if k != "eval_batch_size"}
    data = {k: v for k, v in cfg["data"].items() if k != "test_subset"}
    spec = {"model": cfg["model"], "train": train, "augment": cfg["augment"], "data": data,
            "val_corruptions": [cfg["corruptions"]["val"], cfg["corruptions"]["val_severities"], cfg["corruptions"]["seed"]],
            "branches": [fingerprint(load_run_config(b, cfg.get("profile"))) for b in cfg["model"].get("branches", [])]}
    # v3 dependencies enter the hash only when used, so the fingerprints of older runs do not change
    teacher = (cfg["train"].get("distill") or {}).get("teacher")
    if teacher:
        spec["teacher"] = fingerprint(load_run_config(teacher, cfg.get("profile")))
    if cfg.get("ssl"):
        spec["ssl"] = cfg["ssl"]
    if cfg["model"].get("init_from"):
        spec["init_from"] = fingerprint(load_run_config(cfg["model"]["init_from"], cfg.get("profile")))
    return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _check_fingerprint(cfg: dict, rd: Path) -> None:
    fp_file = rd / "fingerprint.txt"
    fp = fingerprint(cfg)
    if fp_file.exists() and fp_file.read_text().strip() != fp:
        raise RuntimeError(
            f"{rd} was trained with a different configuration than configs/runs/{cfg['name']}.yaml "
            f"(or its branches / base configs) now describe. Delete that directory to retrain, or give the "
            f"changed configuration a new run name.")
    rd.mkdir(parents=True, exist_ok=True)
    fp_file.write_text(fp)


def _build(cfg: dict, seed: int, device, for_training: bool):
    branches = [load_model(b, seed, device, cfg.get("profile"))[0] for b in cfg["model"].get("branches", [])] or None
    model = build_model(cfg["model"], cfg["data"]["num_classes"], branches, pretrained=None if for_training else False)
    return model.to(device)


def ensure_trained(name: str, seed: int, profile: str | None = None, device=None, use_wandb: bool = False) -> dict:
    """Train run `name` with `seed` unless it is already finished. Returns its done.json."""
    cfg = load_run_config(name, profile)
    device = device or get_device()
    rd = run_dir(cfg, seed)
    _check_fingerprint(cfg, rd)
    done = rd / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    for b in cfg["model"].get("branches", []):
        ensure_trained(b, seed, profile, device, use_wandb)
    teacher_name = (cfg["train"].get("distill") or {}).get("teacher")
    init_from = cfg["model"].get("init_from")
    for dep in (teacher_name, init_from):
        if dep:
            ensure_trained(dep, seed, profile, device, use_wandb)
    (rd / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    model = _build(cfg, seed, device, for_training=True)
    if init_from:
        _init_from(model, init_from, seed, profile, device)
    scratch = scratch_root(cfg)
    scratch_dir = scratch / cfg["name"] / f"seed{seed}" if scratch else None
    if cfg["train"].get("objective", "supervised") == "simclr":
        from .ssl import train_simclr
        info = train_simclr(cfg, seed, rd, model, device, scratch_dir=scratch_dir, fingerprint=fingerprint(cfg))
    else:
        teacher = load_model(teacher_name, seed, device, profile)[0] if teacher_name else None
        info = train_run(cfg, seed, rd, model, device, use_wandb, scratch_dir=scratch_dir,
                         fingerprint=fingerprint(cfg), teacher=teacher)
        del teacher
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info


def _classifier_prefixes(model) -> list[str]:
    """Parameter-name prefixes of the classification head(s) of a model (not transferred by init_from)."""
    heads = []
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "get_classifier"):
        heads.append(backbone.get_classifier())
    if hasattr(model, "head"):
        heads.append(model.head)
    return [name + "." for name, m in model.named_modules() if any(m is h for h in heads) and name]


def _init_from(model, run: str, seed: int, profile: str | None, device) -> None:
    """Initialise `model` with the weights of a finished run (e.g. self-supervised pre-training), except
    the classification head and any parameter whose shape differs."""
    cfg = load_run_config(run, profile)
    state = torch.load(run_dir(cfg, seed) / "best.pt", map_location=device, weights_only=False)["model"]
    own = model.state_dict()
    skip = tuple(_classifier_prefixes(model))
    keep = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape and not k.startswith(skip)}
    model.load_state_dict(keep, strict=False)
    log.info(f"initialised {len(keep)} of {len(own)} tensors from {run} seed {seed}")


def load_model(name: str, seed: int, device=None, profile: str | None = None):
    """Return (model in eval mode with its best weights, config). The run must be trained."""
    cfg = load_run_config(name, profile)
    device = device or get_device()
    rd = run_dir(cfg, seed)
    if not (rd / "done.json").exists():
        raise FileNotFoundError(f"run {name} seed {seed} is not trained yet ({rd}). "
                                f"Run: python scripts/train.py --run {name} --seeds {seed}")
    _check_fingerprint(cfg, rd)
    model = _build(cfg, seed, device, for_training=False)
    best = rd / "best.pt"
    if best.exists():
        model.load_checkpoint_state(torch.load(best, map_location=device, weights_only=False)["model"])
    return model.eval(), cfg


def eval_tag(suite: str, tta: dict | None = None, severities=None, n_per_set: int = 0, calibrate: bool = False) -> str:
    tag = suite
    if tta and tta.get("method", "none") != "none":
        tag += f"__{tta['method']}_bs{tta['batch_size']}"
        if tta["method"] == "tent":
            tag += f"_lr{float(tta.get('lr', 1e-3)):g}" + ("_fused" if tta.get("objective") == "fused" else "")
        if float(tta.get("prior", 0) or 0) > 0:
            tag += f"_p{float(tta['prior']):g}"
    if severities:
        tag += "__sev" + "".join(str(s) for s in severities)
    if n_per_set:
        tag += f"__n{n_per_set}"
    if calibrate:
        tag += "__cal"
    return tag


def temperature(name: str, seed: int, profile: str | None = None, device=None, force: bool = False) -> float:
    """Post-hoc temperature of (run, seed): fitted once and cached in <run dir>/temperature.json.

    It is fitted on the select half of the validation split, clean and as validation-C in equal
    parts (the half that no stacked gate was trained on), so every model is calibrated on the same
    held-out images and no test image is involved.
    """
    cfg = load_run_config(name, profile)
    device = device or get_device()
    path = run_dir(cfg, seed) / "temperature.json"
    if path.exists() and not force:
        return float(json.loads(path.read_text())["temperature"])
    model, cfg = load_model(name, seed, device, profile)
    kw = dict(device=device, amp=bool(cfg["train"].get("amp", True)), batch_size=int(cfg["train"].get("eval_batch_size", 512)))
    xv, yv, xc, yc = load_selection_data(cfg)
    logits = torch.cat([predict_logits(model, xv, **kw), predict_logits(model, xc, **kw)])
    y = torch.from_numpy(np.concatenate([yv, yc])).long()
    t = fit_temperature(logits, y)
    path.write_text(json.dumps({"temperature": t, "n_images": int(len(y)), "nll_before": nll_at(logits, y, 1.0),
                                "nll_after": nll_at(logits, y, t),
                                "fitted_on": "select half of the validation split: clean + validation-C"}, indent=1))
    log.info(f"[{name} seed {seed}] temperature {t:.3f} (NLL {nll_at(logits, y, 1.0):.4f} -> {nll_at(logits, y, t):.4f} "
             f"on {len(y)} held-out images)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return t


def evaluate_run(name: str, seed: int, suite: str, profile: str | None = None, device=None, tta: dict | None = None,
                 severities=None, n_per_set: int = 0, force: bool = False, calibrate: bool = False):
    """Evaluate (run, seed) on a suite, with caching.

    tta: None, or {"method": "bn" | "tent", "batch_size": int, "lr": float, "objective": "branch" | "fused",
         "prior": float}.
    calibrate: divide the logits by the run's post-hoc temperature (see `temperature`).
    Returns (DataFrame, {"confusion": [S, K, K], "bins": [S, 15, 3]}); rows and arrays share their order.
    """
    cfg = load_run_config(name, profile)
    device = device or get_device()
    out_dir = run_dir(cfg, seed) / "eval"
    tag = eval_tag(suite, tta, severities, n_per_set, calibrate)
    csv_path, npz_path = out_dir / f"{tag}.csv", out_dir / f"{tag}.npz"
    # A cached evaluation of corrupted sets is only valid for the sets it was computed on: if the
    # corruption parameters change, GTSRB-C is regenerated with a new fingerprint and old caches are redone.
    data_fp = corrupted_fingerprint(cfg, suite[:-2]) if suite.endswith("_c") else ""
    if csv_path.exists() and npz_path.exists() and not force:
        with np.load(npz_path) as z:
            cached_fp = str(z["data_fp"]) if "data_fp" in z.files else ""
            if cached_fp == data_fp:
                return pd.read_csv(csv_path), {"confusion": z["confusion"], "bins": z["bins"]}
        log.info(f"{name} seed {seed}: {tag} was computed on other corrupted sets; evaluating again")

    if suite.endswith("_c"):
        missing = corrupted_sets_missing(cfg, suite[:-2])
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of the corrupted {suite[:-2]} test sets are missing or still being generated "
                f"(first: {missing[0]}). Run, or wait for: {make_c_command(cfg, suite[:-2])}")
    temp = temperature(name, seed, profile, device) if calibrate else 1.0
    model, cfg = load_model(name, seed, device, profile)
    amp = bool(cfg["train"].get("amp", True))
    if tta and tta.get("method", "none") != "none":
        adapter = build_tta(tta["method"], model, amp=amp, lr=float(tta.get("lr", 1e-3)),
                            objective=tta.get("objective", "branch"), prior=float(tta.get("prior", 0) or 0))
        predict = partial(predict_stream, adapter, device=device, batch_size=int(tta["batch_size"]))
    else:
        predict = partial(predict_logits, model, device=device, amp=amp,
                          batch_size=int(cfg["train"].get("eval_batch_size", 512)))
    if calibrate:
        uncalibrated = predict

        def predict(x):
            return uncalibrated(x) / temp
    sets = build_sets(cfg, suite, severities)
    df, confs, bins = evaluate_sets(predict, sets, cfg["data"]["num_classes"], n_per_set,
                                    desc=f"{name} seed {seed}: {tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, confusion=confs, bins=bins, data_fp=data_fp)
    df.to_csv(csv_path, index=False)
    del model, predict
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return df, {"confusion": confs, "bins": bins}
