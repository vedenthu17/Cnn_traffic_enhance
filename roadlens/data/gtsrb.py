"""GTSRB (official 39,209 train / 12,630 test): download, resize-once cache, track-wise val split."""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..utils import get_logger

log = get_logger()

_BASE = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
GTSRB_FILES = {
    "train": "GTSRB_Final_Training_Images.zip",
    "test": "GTSRB_Final_Test_Images.zip",
    "test_gt": "GTSRB_Final_Test_GT.zip",
}
N_TRAIN, N_TEST = 39209, 12630


def download(url: str, dst: Path) -> None:
    """Download with a progress bar; writes to `.part` first so interrupted downloads are not reused."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    log.info(f"Downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dst)


def load_resized(items: list[tuple[str, tuple[int, int, int, int] | None]], size: int,
                 desc: str = "resize", zf: zipfile.ZipFile | None = None) -> np.ndarray:
    """Read images (members of `zf`, or file paths), optionally crop to ROI (x1, y1, x2, y2) and
    resize to size x size. Returns uint8 NHWC.

    Reading straight from the archive avoids unpacking tens of thousands of small files.
    """
    out = np.empty((len(items), size, size, 3), dtype=np.uint8)
    for i, (path, roi) in enumerate(tqdm(items, desc=desc)):
        src = io.BytesIO(zf.read(path)) if zf is not None else path
        with Image.open(src) as im:
            im = im.convert("RGB")
            if roi is not None:
                im = im.crop((roi[0], roi[1], roi[2] + 1, roi[3] + 1))
            out[i] = np.asarray(im.resize((size, size), Image.BILINEAR))
    return out


def read_gt_csv(zf: zipfile.ZipFile, member: str) -> list[dict]:
    with zf.open(member) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter=";"))


def roi_of(row: dict) -> tuple[int, int, int, int]:
    return tuple(int(row[k]) for k in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"))


def _index_train(zf: zipfile.ZipFile):
    items, labels, tracks = [], [], []
    gts = sorted(n for n in zf.namelist() if "Final_Training/Images/" in n and n.endswith(".csv"))
    for gt in gts:
        cdir = gt.rsplit("/", 1)[0]
        c = int(cdir.rsplit("/", 1)[1])
        for row in read_gt_csv(zf, gt):
            # File names are <track>_<frame>.ppm: each track is ~30 frames of one physical sign.
            track = int(row["Filename"].split("_")[0])
            items.append((f"{cdir}/{row['Filename']}", roi_of(row)))
            labels.append(c)
            tracks.append(c * 10000 + track)
    return items, np.array(labels, dtype=np.int64), np.array(tracks, dtype=np.int64)


def _index_test(zf_gt: zipfile.ZipFile):
    rows = read_gt_csv(zf_gt, "GT-final_test.csv")
    items = [(f"GTSRB/Final_Test/Images/{r['Filename']}", roi_of(r)) for r in rows]
    labels = np.array([int(r["ClassId"]) for r in rows], dtype=np.int64)
    return items, labels


def _cache_path(data_root: Path, split: str, size: int, crop_roi: bool) -> Path:
    return data_root / "processed" / f"gtsrb_{split}_s{size}{'_roi' if crop_roi else ''}.npz"


def prepare_gtsrb(data_root: Path, size: int = 64, crop_roi: bool = False) -> None:
    """Download GTSRB and cache both official splits as uint8 arrays at the working resolution."""
    raw = data_root / "raw" / "gtsrb"
    todo = [s for s in ("train", "test") if not _cache_path(data_root, s, size, crop_roi).exists()]
    if not todo:
        return
    for name in GTSRB_FILES.values():
        download(_BASE + name, raw / name)

    for split in todo:
        zf = zipfile.ZipFile(raw / GTSRB_FILES[split])
        if split == "train":
            items, y, tracks = _index_train(zf)
            assert len(items) == N_TRAIN, f"expected {N_TRAIN} train images, found {len(items)}"
        else:
            with zipfile.ZipFile(raw / GTSRB_FILES["test_gt"]) as zf_gt:
                items, y = _index_test(zf_gt)
            tracks = np.full(len(items), -1, dtype=np.int64)
            assert len(items) == N_TEST, f"expected {N_TEST} test images, found {len(items)}"
        if not crop_roi:
            items = [(p, None) for p, _ in items]
        x = load_resized(items, size, desc=f"gtsrb {split} -> {size}px", zf=zf)
        zf.close()
        path = _cache_path(data_root, split, size, crop_roi)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, x=x, y=y, track=tracks)
        tmp.replace(path)
        log.info(f"Saved {path} {x.shape}")


def trackwise_split(y: np.ndarray, tracks: np.ndarray, val_fraction: float, seed: int):
    """Stratified split that keeps every track (frames of one physical sign) on one side.

    A random per-image split would put near-duplicate frames of the same sign in train and val,
    making validation accuracy meaningless for early stopping.
    """
    rng = np.random.RandomState(seed)
    val_mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        ctracks = np.unique(tracks[y == c])
        rng.shuffle(ctracks)
        n_val = max(1, int(round(val_fraction * len(ctracks))))
        val_mask |= np.isin(tracks, ctracks[:n_val])
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def load_gtsrb(data_root: Path, split: str, size: int = 64, crop_roi: bool = False,
               val_fraction: float = 0.1, split_seed: int = 0, with_tracks: bool = False):
    """Return (x uint8 NHWC, y int64) for split in {train, val, trainval, test}.

    with_tracks: also return the track id (class * 10000 + track) of every image; -1 for test images.
    """
    official = "test" if split == "test" else "train"
    path = _cache_path(data_root, official, size, crop_roi)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/prepare_data.py")
    with np.load(path) as d:
        x, y, tracks = d["x"], d["y"], d["track"]
    if split in ("test", "trainval"):
        return (x, y, tracks) if with_tracks else (x, y)

    split_file = data_root / "processed" / f"gtsrb_split_seed{split_seed}_val{val_fraction}.json"
    if split_file.exists():
        idx = json.loads(split_file.read_text())
    else:
        tr, va = trackwise_split(y, tracks, val_fraction, split_seed)
        idx = {"train": tr.tolist(), "val": va.tolist()}
        split_file.write_text(json.dumps(idx))
    sel = np.asarray(idx[split])
    return (x[sel], y[sel], tracks[sel]) if with_tracks else (x[sel], y[sel])


def halve_by_track(y: np.ndarray, tracks: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Indices of two halves that share no track, with the tracks of every class dealt out alternately.

    Used to fit a model on one half of the validation split and select it on the other: a random
    per-image split would put near-duplicate frames of one physical sign on both sides.
    """
    rng = np.random.RandomState(seed)
    first = np.zeros(len(y), dtype=bool)
    turn = 0
    for c in np.unique(y):
        ctracks = np.unique(tracks[y == c])
        rng.shuffle(ctracks)
        first |= np.isin(tracks, ctracks[turn % 2::2]) & (y == c)
        turn += len(ctracks)            # a class with an odd number of tracks hands the extra one over
    return np.flatnonzero(first), np.flatnonzero(~first)


def load_gtsrb_cfg(cfg: dict, data_root: Path, split: str, with_tracks: bool = False):
    d = cfg["data"]
    return load_gtsrb(data_root, split, d["image_size"], d["crop_roi"], d["val_fraction"], d["split_seed"], with_tracks)
