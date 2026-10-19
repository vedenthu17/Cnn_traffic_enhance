"""Write colab_files/RoadLens_standalone.ipynb: ONE notebook that contains the whole project.

    python scripts/make_colab_notebook.py

The notebook needs no zip and no .py file: every source file of the project (the roadlens package,
the scripts, the configs, the tests) is embedded as a `%%writefile` cell, so the notebook recreates
the project in /content/RoadLens when it runs. The datasets are downloaded from their official
sites. Checkpoints, results and the corrupted test sets are kept on Google Drive (optional) so that
an interrupted run continues in the next session. Edit the project files and re-run this script;
do not edit the notebook by hand.
"""

import ast
import json

import _bootstrap  # noqa: F401

from roadlens.utils import REPO_ROOT

OUT = REPO_ROOT / "colab_files"
NOTEBOOK = "RoadLens_standalone.ipynb"
SKIP_SCRIPTS = {"make_bundle.py", "make_colab_notebook.py", "make_test_notebook.py"}
KEEP_NOTEBOOKS = {NOTEBOOK, "RoadLens_test.ipynb"}       # written by scripts/make_test_notebook.py
CODE_ROOT = "/content/RoadLens"


def project_files():
    files = []
    for pattern in ("roadlens/**/*.py", "roadlens/**/*.yaml", "configs/**/*.yaml", "scripts/*.py", "tests/*.py"):
        for p in sorted(REPO_ROOT.glob(pattern)):
            if "__pycache__" in p.parts or p.name in SKIP_SCRIPTS:
                continue
            files.append(p)
    files.append(REPO_ROOT / "requirements.txt")
    return files


SETTINGS = r'''
# ---- Settings: change these, then Runtime > Run all ----
PROFILE = 'pilot'        # 'smoke' (minutes, meaningless numbers) | 'pilot' (1 seed, quick) | '' (full: seeds 0, 1, 2)
SEEDS   = ''             # optional, e.g. '0' to run only seed 0 of the full experiments first
KEEP_ON_DRIVE = True     # keep checkpoints, results and the corrupted test sets on Google Drive (survive the session)
SAVE_C_TO_DRIVE = True   # keep the corrupted test sets on Drive too (~3.4 GB pilot, ~13.3 GB full)
RUN_CHECKS = False       # also run the unit tests and print a speed estimate for this GPU
RUN_V3 = True            # also train and evaluate RoadLens v3 (experiments 8 and 9); finished runs are skipped

import glob, os, shutil, subprocess, sys, time
CODE   = '/content/RoadLens'                 # the project is written here by the cells below
DATA   = '/content/roadlens_data'            # datasets on the fast local disk (downloaded every session)
if KEEP_ON_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    STORE = '/content/drive/MyDrive/RoadLens_standalone'   # checkpoints, results, cached corrupted sets
else:
    STORE = '/content/roadlens_store'
CACHE = f'{STORE}/corrupted_cache'
os.makedirs(STORE, exist_ok=True)
for d in DIRS:
    os.makedirs(f'{CODE}/{d}', exist_ok=True)

PROF = f'--profile {PROFILE}' if PROFILE else ''
P = PROF + (f' --seeds {SEEDS}' if SEEDS else '')
RESULTS = 'results' + (f'_{PROFILE}' if PROFILE else '')
print('profile:', PROFILE or 'full', '| outputs ->', STORE)
'''

SETUP = r'''
# ---- Install requirements, link outputs to the store, check the GPU ----
%cd {CODE}
!pip install -q -r requirements.txt

for name in ('results', 'results_pilot', 'results_smoke'):     # results*/ live in the store
    target, link = f'{STORE}/{name}', f'{CODE}/{name}'
    os.makedirs(target, exist_ok=True)
    if not os.path.islink(link):
        shutil.rmtree(link, ignore_errors=True)
        os.symlink(target, link)
os.environ['ROADLENS_DATA'] = DATA
os.environ['ROADLENS_RUNS'] = f'{STORE}/runs'                    # runs/, runs_pilot/, ... in the store
os.environ['ROADLENS_SCRATCH'] = '/content/roadlens_scratch'     # per-epoch checkpoints stay off Drive

def run(cmd):
    """Run a shell command with live output; stop the notebook if it fails."""
    print('$', cmd)
    get_ipython().system(cmd)
    code = get_ipython().user_ns.get('_exit_code', 0)
    if code:
        raise RuntimeError(f'command failed (exit code {code}): {cmd}')

import torch
if not torch.cuda.is_available():
    raise RuntimeError('No GPU: Runtime > Change runtime type > T4 GPU, then Runtime > Run all again')
print('GPU:', torch.cuda.get_device_name(0), f'({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)')

# corrupted test sets of this profile, and how to keep them on Drive
SUB = {'smoke': '_sub200', 'pilot': '_sub2000', '': ''}[PROFILE]
C_DIRS = [f'gtsrb_c_s64{SUB}', f'btsd_c_s64{SUB}']
C_FILTER = ' '.join(f'--include="{d}/***"' for d in C_DIRS) + ' --exclude="*"'

def save_corrupted_to_drive():
    if not (KEEP_ON_DRIVE and SAVE_C_TO_DRIVE):
        return
    need = sum(os.path.getsize(f) for d in C_DIRS for f in glob.glob(f'{DATA}/{d}/**', recursive=True)
               if os.path.isfile(f))
    free = shutil.disk_usage(STORE).free
    if free < 1.1 * need:
        print(f'not saving the corrupted sets to Drive: needs {need / 2**30:.1f} GB, {free / 2**30:.1f} GB free')
        return
    run(f'mkdir -p "{CACHE}" && rsync -a {C_FILTER} "{DATA}/" "{CACHE}/"')
'''

DATA_CELL = r'''
# Download and cache GTSRB and BTSD at 64 px (about 620 MB from the official sites)
run('python scripts/prepare_data.py --datasets gtsrb btsd')
# validation-C (needed for model selection before any training)
run(f'python scripts/make_gtsrb_c.py --val-c-only {PROF}')

# corrupted test sets: restore from Drive if saved there, generate the rest in the background
os.makedirs(DATA, exist_ok=True)
if os.path.isdir(CACHE):
    run(f'rsync -a {C_FILTER} "{CACHE}/" "{DATA}/"')
if 'gen' in globals() and gen.poll() is None:
    print('the corrupted test sets are already being generated')
else:
    cmd = (f'{sys.executable} scripts/make_gtsrb_c.py --workers 1 --overwrite --skip-val-c {PROF} && '
           f'{sys.executable} scripts/make_gtsrb_c.py --dataset btsd --workers 1 --overwrite {PROF}')
    gen = subprocess.Popen(cmd, shell=True, stdout=open('/content/make_c.log', 'w'), stderr=subprocess.STDOUT)
    print('generating the corrupted test sets in the background (log: /content/make_c.log)')

def wait_for_corrupted_sets():
    while gen.poll() is None:
        lines = open('/content/make_c.log').read().replace('\r', '\n').strip().splitlines()
        print(time.strftime('%H:%M:%S'), 'still generating:', lines[-1][-100:] if lines else '')
        time.sleep(120)
    assert gen.returncode == 0, open('/content/make_c.log').read()[-3000:]
    print('corrupted test sets are ready')
    save_corrupted_to_drive()
'''

SHOW = r'''
TABLES = ['exp1_clean/clean.txt', 'exp2_corruption/main.txt', 'exp2_corruption/per_corruption.txt',
          'exp3_ablation/ablation.txt', 'exp4_leave_category_out/candidates.txt',
          'exp4_leave_category_out/selection.txt', 'exp5_cross_dataset/btsd.txt',
          'exp6_tent_batchsize/sweep.txt', 'exp7_improved/improved.txt',
          'exp7_improved/severity5.txt', 'exp7_improved/adaptation.txt',
          'exp8_v3/v3.txt', 'exp8_v3/per_family.txt', 'exp8_v3/efficiency.txt',
          'exp9_classical_unsupervised/classical.txt', 'exp9_classical_unsupervised/clusters.txt']
for path in TABLES:
    full = os.path.join(RESULTS, path)
    print(f'\n===== {path} =====')
    print(open(full).read() if os.path.exists(full) else '(missing)')
'''

ALL_RUNS = ["smallcnn_base", "resnet18_base", "deit_s_base", "resnet18_mix", "resnet18_mix_jsd", "resnet18_full",
            "deit_s_full", "fusion_full", "resnet18_jsd_prime", "resnet18_jsd_nogeom", "resnet18_in", "deit_s_in",
            "fusion_v2"]
# RoadLens v3 (experiment 8); the SimCLR run comes before the run that starts from it
V3_RUNS = ["resnet18_ln", "resnet18_ln_apr", "resnet18_v3", "resnet18_v3_64", "resnet18_simclr", "resnet18_v3_ssl",
           "convnext_atto_v3", "efficientnet_b0_v3", "mobilenetv3_v3", "deit_s_v3", "student_v3", "student_mbv3"]


def main():
    files = project_files()
    rels = [p.relative_to(REPO_ROOT).as_posix() for p in files]
    dirs = sorted({r.rsplit("/", 1)[0] for r in rels if "/" in r})
    cells = [("md", "# RoadLens - the complete project in one notebook\n\n"
                    "Traffic sign classification robust to image corruptions. **This notebook needs nothing else:** "
                    "all project code is inside it (section 0 writes it to `/content/RoadLens`), and the datasets "
                    "are downloaded from their official sites.\n\n"
                    "**How to run:** `Runtime > Change runtime type > T4 GPU`, then `Runtime > Run all`, and approve "
                    "the Google Drive prompt (Drive keeps checkpoints and results between sessions; set "
                    "`KEEP_ON_DRIVE = False` to skip it).\n\n"
                    "**Scale** (first code cell): `PROFILE = 'pilot'` = 1 seed, shortened training, 2,000 test images "
                    "(roughly 1-1.5 h, not measured); `''` = the full experiments with seeds 0, 1, 2 (roughly 10-12 GPU "
                    "hours, not measured, so several sessions). If a session ends, open the notebook and Run all "
                    "again: finished training runs and evaluations are skipped.\n\n"
                    "| Section | What it does |\n|---|---|\n"
                    "| 0. Project code | writes every source file |\n"
                    "| 1. Data | downloads GTSRB + BTSD, builds validation-C, generates the corrupted test sets in the background |\n"
                    "| 2. Checks | optional unit tests and speed estimate |\n"
                    "| 3. Training | every model of every experiment |\n"
                    "| 4. Experiments 1-9 | clean, corruptions (main table), ablation, leave-category-out, BTSD, TENT batch size, "
                    "v2, v3 (new stem, APR, EMA, other architectures, SimCLR, distillation), classical / unsupervised learners |\n"
                    "| 5. Figures | severity curves, confusion matrices, reliability, Fourier heat map, Grad-CAM (PDF) |\n"
                    "| 6. Results | prints every table and downloads them as a zip |"),
             ("code", f"DIRS = {dirs!r}\n" + SETTINGS.strip()),
             ("md", f"## 0. Project code\n\nEvery file of the project, {len(files)} in total. Running these cells "
                    f"writes them to `{CODE_ROOT}`. To change the code, edit a cell and run it again.")]
    cells += code_cells(files, rels)
    cells += [
        ("md", "## Setup"),
        ("code", SETUP.strip()),
        ("md", "## 1. Data"),
        ("code", DATA_CELL.strip()),
        ("code", "run('python scripts/plot_corruptions.py --stats-n 150')\nfrom IPython.display import Image, display\n"
                 "display(Image('results/corruptions/grid_0.png', width=480))"),
        ("md", "## 2. Checks\nWith `RUN_V3`, the v3 unit tests always run first (about 1-2 minutes on the CPU), so that a "
               "fault in new code stops the notebook before any GPU time is spent."),
        ("code", "if RUN_CHECKS:\n    run('python -m pytest tests -q')\n    run('python scripts/benchmark.py')\n"
                 "elif RUN_V3:\n    run('python -m pytest tests/test_v3.py -q')"),
        ("md", "## 3. Training\nEvery model, one at a time; finished runs are skipped. The corrupted test sets "
               "are generated meanwhile."),
        ("code", f"RUNS = {ALL_RUNS!r}\nV3_RUNS = {V3_RUNS!r}\nSEED_LIST = SEEDS or ('0' if PROFILE else '0 1 2')\n"
                 "for name in RUNS + (V3_RUNS if RUN_V3 else []):\n"
                 "    run(f'python scripts/train.py --run {name} --seeds {SEED_LIST} {PROF}')"),
        ("md", "## 4. Experiments\nTables (CSV, TXT, LaTeX) go to `results*/<experiment>/`."),
        ("code", "# Experiment 1 - clean accuracy and macro-F1\nrun(f'python scripts/exp1_clean.py {P}')"),
        ("code", "# Experiment 2 - every corruption x severity, mCE, relative mCE, ECE (main table)\n"
                 "wait_for_corrupted_sets()\nrun(f'python scripts/exp2_corruption.py {P}')"),
        ("code", "# Experiment 3 - ablation: baseline -> +mixing -> +JSD -> +Fourier/PRIME ops -> +fusion -> +TENT\n"
                 "run(f'python scripts/exp3_ablation.py {P}')"),
        ("code", "# Experiment 4 - leave-category-out\nrun(f'python scripts/exp4_leave_category_out.py {P}')"),
        ("code", "# Experiment 5 - cross-dataset: Belgian signs (BTSD)\nrun(f'python scripts/exp5_cross_dataset.py {P}')"),
        ("code", "# Experiment 6 - TENT / test-time BN vs. batch size 1, 16, 64, 200\n"
                 "run(f'python scripts/exp6_tent_batchsize.py {P}')"),
        ("code", "# Experiment 7 - the v2 approach\nrun(f'python scripts/exp7_improved.py {P}')"),
        ("code", "# Experiment 8 - the v3 approach: local-global norm stem, APR, EMA, other architectures,\n"
                 "# SimCLR pre-training, distilled single-network students, GPU throughput\n"
                 "if RUN_V3:\n    run(f'python scripts/exp8_v3.py {P}')"),
        ("code", "# Experiment 9 - classical supervised learners (HOG + SVM / random forest / kNN), classifiers on\n"
                 "# the self-supervised SimCLR features, and an unsupervised k-means stability analysis\n"
                 "if RUN_V3:\n    run(f'python scripts/exp9_classical_unsupervised.py {P}')"),
        ("md", "## 5. Figures"),
        ("code", "run(f'python scripts/make_figures.py {PROF}')\n!pip install -q pymupdf\nimport fitz\n"
                 "for pdf in sorted(glob.glob(f'{RESULTS}/figures/*.pdf')) + glob.glob(f'{RESULTS}/exp6_tent_batchsize/*.pdf'):\n"
                 "    print(pdf)\n    display(Image(data=fitz.open(pdf)[0].get_pixmap(dpi=90).tobytes('png')))"),
        ("md", "## 6. Results"),
        ("code", SHOW.strip()),
        ("code", "shutil.make_archive(f'/content/{RESULTS}', 'zip', STORE, RESULTS)\n"
                 "from google.colab import files\nfiles.download(f'/content/{RESULTS}.zip')\n"
                 "print('Everything is also in', f'{STORE}/{RESULTS}')"),
    ]
    target = write_notebook(cells, NOTEBOOK)
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB, {len(cells)} cells, {len(files)} embedded files)")


def code_cells(files, rels):
    """One `%%writefile` cell per project file, grouped under a heading per top-level directory."""
    cells, current = [], None
    for p, rel in zip(files, rels):
        top = rel.split("/")[0] if "/" in rel else "other"
        if top != current:
            current = top
            cells.append(("md", f"### `{top}/`" if top != "other" else "### requirements"))
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".py":
            ast.parse(text)
        cells.append(("raw_code", f"%%writefile {CODE_ROOT}/{rel}\n{text.rstrip()}\n"))
    return cells


def write_notebook(cells, name):
    """Write [(kind, source)] cells as colab_files/<name>; kind is md, code (syntax-checked) or raw_code."""
    nb_cells = []
    for kind, src in cells:
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})
            continue
        if kind == "code":
            py = "\n".join(ln[:len(ln) - len(ln.lstrip())] + "pass" if ln.lstrip().startswith(("!", "%")) else ln
                           for ln in src.splitlines())
            ast.parse(py)
        nb_cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                         "outputs": [], "source": src.splitlines(keepends=True)})
    doc = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 0,
           "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                        "kernelspec": {"display_name": "Python 3", "name": "python3"},
                        "language_info": {"name": "python"}}}
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.ipynb"):
        if old.name not in KEEP_NOTEBOOKS:
            old.unlink()
    target = OUT / name
    target.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    main()
