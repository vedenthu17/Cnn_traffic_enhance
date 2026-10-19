"""Generate once / reuse: GTSRB-C (test corruptions x severities) and the small validation-C."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..utils import get_logger
from . import corruptions as C

log = get_logger()


def _fingerprint(x: np.ndarray, names: list[str], seed: int) -> str:
    """Hash of everything that determines the corrupted pixels."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(x).tobytes())
    spec = {"seed": seed, "shape": list(x.shape), "frost_scale": C.FROST_TEXTURE_SCALE,
            "params": {n: C.PARAMS[n] for n in sorted(names)}}
    h.update(json.dumps(spec, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _work(args):
    x, name, severity, seed, start = args
    import cv2
    cv2.setNumThreads(0)
    return start, C.corrupt_batch(x, name, severity, seed, start)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def gtsrb_c_dir(data_root: Path, size: int) -> Path:
    return data_root / f"gtsrb_c_s{size}"


def generate_corrupted_sets(x: np.ndarray, y: np.ndarray, out_dir: Path, names: list[str],
                            severities: list[int], seed: int, workers: int = 0,
                            chunk: int = 256, overwrite: bool = False) -> None:
    """Write out_dir/<corruption>/<severity>.npy (uint8 NHWC) + labels.npy + manifest.json.

    Resumable: finished files are skipped. Refuses to mix files generated from different
    data / seed / parameters unless `overwrite`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    size = x.shape[1]
    if not 0.75 * C.REFERENCE_SIZE <= size <= 1.5 * C.REFERENCE_SIZE:
        log.warning(f"Corruption constants are calibrated for ~{C.REFERENCE_SIZE}px images; at {size}px "
                    f"blur/weather severities will not be comparable. See CORRUPTIONS.md.")
    fp = _fingerprint(x, names, seed)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest and manifest["fingerprint"] != fp:
        if not overwrite:
            raise RuntimeError(f"{out_dir} was generated with different data/seed/parameters. "
                               f"Re-run with --overwrite or delete the directory.")
        manifest = None
    if manifest is None:
        manifest = {"fingerprint": fp, "seed": seed, "n": int(len(x)), "image_size": int(size),
                    "params": {n: C.PARAMS[n] for n in names}, "done": []}
        for p in out_dir.glob("*/*.npy"):
            p.unlink()
    np.save(out_dir / "labels.npy", y)

    jobs = [(n, s) for n in names for s in severities if f"{n}/{s}" not in manifest["done"]]
    if not jobs:
        log.info(f"{out_dir}: all {len(names) * len(severities)} sets already present.")
        return
    workers = workers or min(os.cpu_count() or 1, 8)
    starts = list(range(0, len(x), chunk))
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        with tqdm(total=len(jobs) * len(x), unit="img", desc="corrupting") as bar:
            for name, sev in jobs:
                bar.set_postfix_str(f"{name} s{sev}")
                tasks = [(x[s:s + chunk], name, sev, seed, s) for s in starts]
                out = np.empty_like(x)
                for start, res in (pool.map(_work, tasks) if pool else map(_work, tasks)):
                    out[start:start + len(res)] = res
                    bar.update(len(res))
                _save_npy(out_dir / name / f"{sev}.npy", out)
                manifest["done"].append(f"{name}/{sev}")
                manifest_path.write_text(json.dumps(manifest, indent=1, default=str))
    finally:
        if pool:
            pool.shutdown()


def load_corrupted(out_dir: Path, name: str, severity: int, mmap: bool = True):
    """Return (x uint8 NHWC, y) of one pre-generated corrupted set."""
    path = out_dir / name / f"{severity}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/make_gtsrb_c.py")
    return np.load(path, mmap_mode="r" if mmap else None), np.load(out_dir / "labels.npy")


def val_c_path(data_root: Path, size: int) -> Path:
    return data_root / f"gtsrb_val_c_s{size}.npz"


def generate_val_c(x: np.ndarray, y: np.ndarray, path: Path, names: list[str],
                   severities: list[int], seed: int, overwrite: bool = False) -> None:
    """Validation-C: every validation image gets exactly one (held-out corruption, severity),
    balanced over all combinations. Same size as the clean validation split."""
    fp = _fingerprint(x, names, seed) + ":" + ",".join(map(str, severities))
    if path.exists() and not overwrite:
        with np.load(path) as d:
            if str(d["fingerprint"]) == fp:
                log.info(f"{path}: already present.")
                return
        raise RuntimeError(f"{path} was generated with different data/seed/parameters; use --overwrite.")
    combos = [(n, s) for n in names for s in severities]
    order = np.random.RandomState(seed).permutation(len(x))
    cid = np.empty(len(x), dtype=np.int64)
    sev = np.empty(len(x), dtype=np.int64)
    out = np.empty_like(x)
    for k, i in enumerate(tqdm(order, desc="validation-C")):
        name, s = combos[k % len(combos)]
        out[i] = C.corrupt(x[i], name, s, C.derive_seed(seed, name, s, int(i)))
        cid[i], sev[i] = C.CORRUPTION_IDS[name], s
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, x=out, y=y, corruption_id=cid, severity=sev, fingerprint=fp)
    os.replace(tmp, path)
    log.info(f"Saved {path} {out.shape}")


def load_val_c(data_root: Path, size: int):
    path = val_c_path(data_root, size)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/make_gtsrb_c.py")
    with np.load(path) as d:
        return d["x"], d["y"], d["corruption_id"], d["severity"]
