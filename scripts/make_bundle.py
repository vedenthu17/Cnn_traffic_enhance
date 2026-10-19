"""Build the two files to upload to Google Drive for Colab.

    python scripts/make_bundle.py                 # -> ~/RoadLens_gdrive/RoadLens.zip + RoadLens_colab.ipynb
    python scripts/make_bundle.py --out D:/upload

RoadLens.zip contains
    RoadLens/         the code, configs, tests and the stage-1 figures
    roadlens_data/    the cached 64 px datasets (GTSRB train/test, BTSD, the train/val split, validation-C),
                      so that Colab neither downloads nor re-processes them
Checkpoints (runs*/), reduced-profile results and the corrupted test sets are not included; the
corrupted sets are regenerated on Colab.
"""

import argparse
import shutil
import time
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from roadlens.utils import REPO_ROOT, data_root, load_config

CODE = ["configs", "roadlens", "scripts", "tests", "notebooks", "colab_files", "results/corruptions", "results/mappings",
        "README.md", "CORRUPTIONS.md", "requirements.txt", ".gitignore"]
UPDATE = ["configs", "roadlens", "scripts", "tests", "notebooks", "colab_files", "README.md", "CORRUPTIONS.md", "requirements.txt"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path.home() / "RoadLens_gdrive"))
    ap.add_argument("--no-data", action="store_true", help="code only (Colab then downloads the datasets itself)")
    ap.add_argument("--update", action="store_true",
                    help="write RoadLens_update.zip instead: the code alone (well under 1 MB), to put next to an "
                         "already uploaded RoadLens.zip; the notebook installs it on top")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.update:
        files = []
        for item in UPDATE:
            path = REPO_ROOT / item
            if path.is_file():
                files.append((path, f"RoadLens/{item}"))
            elif path.is_dir():
                files += [(p, f"RoadLens/{p.relative_to(REPO_ROOT).as_posix()}") for p in sorted(path.rglob("*"))
                          if p.is_file() and not SKIP_DIRS & set(p.parts) and p.suffix != ".pyc"]
        target = out / "RoadLens_update.zip"
        tmp = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("RoadLens/VERSION", time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            for src, name in files:
                z.write(src, name)
        tmp.replace(target)
        print(f"{target}  ({target.stat().st_size / 2 ** 10:.0f} KB: {len(files)} code files)")
        print("Upload it to the top level of My Drive, next to RoadLens.zip; the notebook's first cell installs it.")
        return

    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    root, size = data_root(cfg), cfg["data"]["image_size"]
    files = []
    for item in CODE:
        path = REPO_ROOT / item
        if path.is_file():
            files.append((path, f"RoadLens/{item}"))
        elif path.is_dir():
            files += [(p, f"RoadLens/{p.relative_to(REPO_ROOT).as_posix()}") for p in sorted(path.rglob("*"))
                      if p.is_file() and not SKIP_DIRS & set(p.parts) and p.suffix != ".pyc"]
    data = []
    if not args.no_data:
        wanted = [root / "processed" / f"gtsrb_train_s{size}.npz", root / "processed" / f"gtsrb_test_s{size}.npz",
                  root / "processed" / f"btsd_s{size}.npz", root / f"gtsrb_val_c_s{size}.npz"]
        wanted += sorted((root / "processed").glob("gtsrb_split_*.json"))
        for p in wanted:
            if p.exists():
                data.append((p, f"roadlens_data/{p.relative_to(root).as_posix()}"))
            else:
                print(f"note: {p} is missing and will not be in the bundle (Colab will build it)")

    target = out / "RoadLens.zip"
    tmp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("RoadLens/VERSION", time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        for src, name in files + data:
            z.write(src, name)
    tmp.replace(target)
    shutil.copyfile(REPO_ROOT / "notebooks" / "RoadLens_colab.ipynb", out / "RoadLens_colab.ipynb")
    mb = target.stat().st_size / 2 ** 20
    print(f"{target}  ({mb:.0f} MB: {len(files)} code files, {len(data)} data files)")
    print(f"{out / 'RoadLens_colab.ipynb'}")
    print("Upload both to the top level of My Drive, open the notebook in Colab, choose a T4 GPU, Run all.")


if __name__ == "__main__":
    main()
