"""Cross-dataset test sets: BTSD (BelgiumTSC) and CTSD (Chinese traffic signs, TSRD release).

Only classes that also exist in GTSRB are evaluated; the class maps live in
roadlens/data/mappings/<name>_to_gtsrb.yaml.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import yaml

from ..utils import get_logger
from .gtsrb import download, load_resized, read_gt_csv, roi_of

log = get_logger()

MAPPING_DIR = Path(__file__).parent / "mappings"
BTSD_URL = "https://btsd.ethz.ch/shareddata/BelgiumTSC/"
BTSD_FILES = {"Training": "BelgiumTSC_Training.zip", "Testing": "BelgiumTSC_Testing.zip"}
IMG_EXT = {".ppm", ".png", ".jpg", ".jpeg", ".bmp"}

CTSD_HELP = """CTSD needs a manual download (no reliable scripted source).
Get the Chinese Traffic Sign Recognition Database (TSRD, 58 classes) from either
  - http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html  (tsrd-train.zip, TSRD-Test.zip + annotation .txt)
  - https://www.kaggle.com/datasets/dmitryyemelyanov/chinese-traffic-signs  (images/ + annotations.csv)
and unzip it anywhere under: {dst}
File names must keep their original form <class:3 digits>_<...>.png."""


def _cache_path(data_root: Path, name: str, size: int, crop_roi: bool) -> Path:
    return data_root / "processed" / f"{name}_s{size}{'_roi' if crop_roi else ''}.npz"


def _index_btsd(zf: zipfile.ZipFile):
    """Members look like Testing/00017/01234_00001.ppm with a GT-00017.csv per class folder."""
    names = zf.namelist()
    rois = {}
    for gt in (n for n in names if n.endswith(".csv")):
        cdir = gt.rsplit("/", 1)[0]
        for r in read_gt_csv(zf, gt):
            rois[f"{cdir}/{r['Filename']}"] = roi_of(r)
    items, labels = [], []
    for n in sorted(names):
        parts = n.split("/")
        if len(parts) >= 2 and parts[-2].isdigit() and Path(n).suffix.lower() in IMG_EXT:
            items.append((n, rois.get(n)))
            labels.append(int(parts[-2]))
    return items, labels


def _index_ctsd(raw: Path):
    """Layout-agnostic: class id = first three digits of the file name; ROIs from any annotation file."""
    rois = {}
    for ann in list(raw.rglob("*.txt")) + list(raw.rglob("*.csv")):
        for line in ann.read_text(errors="ignore").splitlines():
            parts = [p for p in re.split(r"[;,]", line.strip()) if p != ""]
            if len(parts) >= 8 and parts[1].isdigit():
                rois[parts[0]] = tuple(int(v) for v in parts[3:7])
    items, labels = [], []
    for p in sorted(raw.rglob("*")):
        m = re.match(r"^(\d{3})_", p.name)
        if p.suffix.lower() in IMG_EXT and m:
            items.append((p, rois.get(p.name)))
            labels.append(int(m.group(1)))
    return items, labels


def prepare_external(data_root: Path, name: str, size: int = 64, crop_roi: bool = False) -> None:
    """Cache every image of the dataset at the working resolution with its SOURCE class id."""
    path = _cache_path(data_root, name, size, crop_roi)
    if path.exists():
        return
    raw = data_root / "raw" / name
    if name == "btsd":
        xs, labels = [], []
        for split, fname in BTSD_FILES.items():
            download(BTSD_URL + fname, raw / fname)
            with zipfile.ZipFile(raw / fname) as zf:
                items, lab = _index_btsd(zf)
                if not crop_roi:
                    items = [(p, None) for p, _ in items]
                xs.append(load_resized(items, size, desc=f"btsd {split} -> {size}px", zf=zf))
                labels += lab
        x = np.concatenate(xs)
    elif name == "ctsd":
        raw.mkdir(parents=True, exist_ok=True)
        items, labels = _index_ctsd(raw)
        if not items:
            raise FileNotFoundError(CTSD_HELP.format(dst=raw))
        if not crop_roi:
            items = [(p, None) for p, _ in items]
        x = load_resized(items, size, desc=f"ctsd -> {size}px")
    else:
        raise ValueError(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, x=x, y_src=np.array(labels, dtype=np.int64))
    tmp.replace(path)
    log.info(f"Saved {path} {x.shape}")


def load_mapping(name: str) -> dict:
    with open(MAPPING_DIR / f"{name}_to_gtsrb.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_external(data_root: Path, name: str, size: int = 64, crop_roi: bool = False,
                  mapped_only: bool = True, allow_unverified: bool = False):
    """Return (x uint8 NHWC, y in GTSRB ids, y_src). With mapped_only=False, unmapped images
    are kept with y = -1."""
    path = _cache_path(data_root, name, size, crop_roi)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/prepare_data.py --datasets {name}")
    with np.load(path) as d:
        x, y_src = d["x"], d["y_src"]
    mapping = load_mapping(name)
    if mapped_only and not mapping.get("verified", False) and not allow_unverified:
        raise RuntimeError(
            f"The {name}->GTSRB class map is not marked verified. Inspect it with "
            f"`python scripts/check_mapping.py --dataset {name}`, fix "
            f"{MAPPING_DIR / (name + '_to_gtsrb.yaml')} and set `verified: true`.")
    lut = np.full(int(y_src.max()) + 1, -1, dtype=np.int64)
    for src, dst in mapping["map"].items():
        if int(src) < len(lut):
            lut[int(src)] = int(dst)
    y = lut[y_src]
    if mapped_only:
        keep = y >= 0
        return x[keep], y[keep], y_src[keep]
    return x, y, y_src
