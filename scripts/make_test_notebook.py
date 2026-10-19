"""Write colab_files/RoadLens_test.ipynb: test the models trained by RoadLens_standalone.ipynb.

    python scripts/make_test_notebook.py

The notebook trains nothing. It embeds the same project code as the training notebook (needed to
rebuild the networks and to check that each checkpoint matches its configuration), loads every
finished best.pt from the training notebook's Google Drive folder, and evaluates it: clean test
accuracy, the corrupted test sets (accuracy, mCE), Belgian signs, a confusion matrix, example
predictions and the user's own photos. Evaluations already done by the experiments are read from
their cache, so they are instant.
"""

import _bootstrap  # noqa: F401

from make_colab_notebook import ALL_RUNS, REPO_ROOT, V3_RUNS, code_cells, project_files, write_notebook

NOTEBOOK = "RoadLens_test.ipynb"
ENSEMBLES = ["ensemble_avg", "ensemble_conf", "ensemble_views", "ensemble_avg_in", "ensemble_views_in", "ensemble_v3"]

SETTINGS = r'''
# ---- Settings: change these, then Runtime > Run all ----
PROFILE = 'pilot'      # must match the training notebook: 'pilot' | 'smoke' | '' (full)
SEED    = 0            # which trained seed to test (0, 1 or 2 for the full run)
MODELS  = []           # [] = every trained model found, or e.g. ['resnet18_full', 'fusion_full']
STORE   = '/content/drive/MyDrive/RoadLens_standalone'   # the training notebook's Drive folder
GENERATE_MISSING_C = True   # build corrupted test sets that are not saved on Drive (can take a while)
TEST_BTSD = True            # also test on Belgian signs (BTSD)
FORCE = False               # evaluate again even when a cached result exists

import glob, io, json, os, shutil, subprocess, sys, time
CODE = '/content/RoadLens'                 # the project is written here by the cells below
DATA = '/content/roadlens_data'            # datasets on the fast local disk
from google.colab import drive
drive.mount('/content/drive')
if not os.path.isdir(STORE):
    raise FileNotFoundError(f'{STORE} not found: run RoadLens_standalone.ipynb first, or fix STORE')
CACHE = f'{STORE}/corrupted_cache'
for d in DIRS:
    os.makedirs(f'{CODE}/{d}', exist_ok=True)
PROF = f'--profile {PROFILE}' if PROFILE else ''
SUFFIX = f'_{PROFILE}' if PROFILE else ''
RUNS_DIR = f'{STORE}/runs{SUFFIX}'
OUT = f'{STORE}/test_results{SUFFIX}'
os.makedirs(OUT, exist_ok=True)
print('profile:', PROFILE or 'full', '| seed', SEED, '| checkpoints:', RUNS_DIR, '| results ->', OUT)
'''

SETUP = r'''
# ---- Install requirements, import the project, pick the device ----
%cd {CODE}
!pip install -q -r requirements.txt
os.environ['ROADLENS_DATA'] = DATA
os.environ['ROADLENS_RUNS'] = f'{STORE}/runs'           # runs_pilot/ etc. is added from the profile
sys.path.insert(0, CODE)

def run(cmd):
    """Run a shell command with live output; stop the notebook if it fails."""
    print('$', cmd)
    get_ipython().system(cmd)
    code = get_ipython().user_ns.get('_exit_code', 0)
    if code:
        raise RuntimeError(f'command failed (exit code {code}): {cmd}')

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display
from roadlens.runs import ensure_trained, evaluate_run, load_model, load_run_config
from roadlens.eval.evaluate import predict_logits
from roadlens.eval.metrics import corruption_summary
from roadlens.data.loading import corrupted_sets_missing, external_unavailable, load_corrupted_set, load_test
from roadlens.data.classes import GTSRB_CLASSES, GTSRB_SHORT
from roadlens.utils import corruption_names

PROFILE_ARG = PROFILE or None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('WARNING: no GPU. Testing works on the CPU but is very slow; Runtime > Change runtime type > T4 GPU')
cfg0 = load_run_config('resnet18_base', PROFILE_ARG)    # data settings shared by every run
CORRUPTIONS = corruption_names(cfg0)
pd.set_option('display.width', 200)
'''

FIND = r'''
# ---- Which trained models are on Drive? ----
rows = []
for done in sorted(glob.glob(f'{RUNS_DIR}/*/seed{SEED}/done.json')):
    info, best = json.load(open(done)), os.path.join(os.path.dirname(done), 'best.pt')
    rows.append({'run': info['name'],
                 'kind': 'ensemble (no weights of its own)' if info.get('skipped') else 'trained',
                 'best epoch': info.get('best_epoch'),
                 'best val-C acc %': round(100 * info['best_val_c_acc'], 2) if info.get('best_val_c_acc') else None,
                 'train min': round(info.get('train_time_s', 0) / 60, 1),
                 'best.pt MB': round(os.path.getsize(best) / 2**20, 1) if os.path.exists(best) else None})
found = pd.DataFrame(rows)
display(found)
trained = set(found.run) if len(found) else set()

# ensembles only combine trained models: set them up (nothing is trained) once their parts exist
for name in ENSEMBLES:
    parts = load_run_config(name, PROFILE_ARG)['model'].get('branches', [])
    if name not in trained and parts and all(p in trained for p in parts):
        ensure_trained(name, SEED, PROFILE_ARG, DEVICE)
        trained.add(name)

TEST = [m for m in (MODELS or ORDER) if m in trained]
not_ready = [m for m in (MODELS or []) if m not in trained]
print('testing:', ', '.join(TEST))
if not_ready:
    print('not trained yet, skipped:', ', '.join(not_ready))
if not TEST:
    raise RuntimeError(f'no finished model in {RUNS_DIR} for seed {SEED}: check PROFILE, SEED and STORE')
'''

DATA_CELL = r'''
# ---- Test data: GTSRB (+ BTSD), and the corrupted test sets ----
run('python scripts/prepare_data.py --datasets gtsrb' + (' btsd' if TEST_BTSD else ''))

DATASETS = ['gtsrb'] + (['btsd'] if TEST_BTSD else [])
SUB = {'smoke': '_sub200', 'pilot': '_sub2000', '': ''}[PROFILE]
C_DIRS = [f'{d}_c_s64{SUB}' for d in DATASETS]
C_FILTER = ' '.join(f'--include="{d}/***"' for d in C_DIRS) + ' --exclude="*"'
os.makedirs(DATA, exist_ok=True)
if os.path.isdir(CACHE):              # saved on Drive by the training notebook
    run(f'rsync -a {C_FILTER} "{CACHE}/" "{DATA}/"')

generated = False
for ds in DATASETS:
    missing = corrupted_sets_missing(cfg0, ds)
    if not missing:
        print(f'{ds}: all corrupted test sets are ready')
    elif GENERATE_MISSING_C:
        print(f'{ds}: building {len(missing)} missing corrupted sets')
        run(f'python scripts/make_gtsrb_c.py --dataset {ds} {PROF}' + (' --skip-val-c' if ds == 'gtsrb' else ''))
        generated = True
    else:
        print(f'{ds}: {len(missing)} corrupted sets missing; its corrupted tests are skipped')
if generated:                          # keep them for next time
    run(f'mkdir -p "{CACHE}" && rsync -a {C_FILTER} "{DATA}/" "{CACHE}/"')
C_READY = {ds: not corrupted_sets_missing(cfg0, ds) for ds in DATASETS}
'''

CLEAN = r'''
# ---- Test 1: clean GTSRB test photos ----
rows = []
for m in TEST:
    df, _ = evaluate_run(m, SEED, 'gtsrb_clean', profile=PROFILE_ARG, device=DEVICE, force=FORCE)
    t = df[df.split == 'test'].iloc[0]
    rows.append({'model': m, 'test images': int(t.n), 'accuracy %': 100 * t.acc,
                 'macro-F1 %': 100 * t.macro_f1, 'ECE %': 100 * t.ece})
clean = pd.DataFrame(rows).set_index('model')
clean.to_csv(f'{OUT}/clean_seed{SEED}.csv', float_format='%.4f')
display(clean.round(2))
'''

CORRUPTED = r'''
# ---- Test 2: the 14 corruptions x 5 severities (GTSRB-C) ----
long = {}
if not C_READY['gtsrb']:
    print('skipped: the corrupted GTSRB test sets are missing (set GENERATE_MISSING_C = True)')
else:
    for m in TEST:
        long[m], _ = evaluate_run(m, SEED, 'gtsrb_c', profile=PROFILE_ARG, device=DEVICE, force=FORCE)
    norm = long.get('smallcnn_base')            # mCE reference (= 100)
    if norm is None and 'smallcnn_base' in trained:
        norm, _ = evaluate_run('smallcnn_base', SEED, 'gtsrb_c', profile=PROFILE_ARG, device=DEVICE)
    cols = {'clean_acc': 'clean acc %', 'corr_acc': 'corrupted acc %', 'mce': 'mCE',
            'rel_mce': 'relative mCE', 'corr_f1': 'corrupted macro-F1 %', 'corr_ece': 'corrupted ECE %'}
    summary = pd.DataFrame({m: corruption_summary(df, norm) for m, df in long.items()}).T
    summary = summary[[c for c in cols if c in summary]].rename(columns=cols)
    summary.index.name = 'model'
    summary.to_csv(f'{OUT}/corrupted_summary_seed{SEED}.csv', float_format='%.4f')
    print('mCE: mistakes on corrupted photos relative to SmallCNN (= 100); lower is better')
    display(summary.round(2))

    per = pd.DataFrame({m: 100 * df[df.corruption != 'clean'].groupby('corruption').acc.mean()
                        for m, df in long.items()}).reindex(CORRUPTIONS)
    per.to_csv(f'{OUT}/per_corruption_seed{SEED}.csv', float_format='%.3f')
    print('accuracy % per corruption (mean over the 5 severities)')
    display(per.round(1).style.background_gradient(axis=1, cmap='RdYlGn'))
'''

CURVES = r'''
# ---- Accuracy as the damage gets stronger ----
if long:
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in long.items():
        sev = 100 * df[df.corruption != 'clean'].groupby('severity').acc.mean()
        ax.plot([0, *sev.index], [100 * df[df.corruption == 'clean'].acc.iloc[0], *sev.values], marker='o', label=m)
    ax.set(xlabel='severity (0 = clean)', ylabel='accuracy % (mean over 14 corruptions)',
           title=f'GTSRB-C, seed {SEED}', xticks=range(6))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(f'{OUT}/severity_curves_seed{SEED}.png', dpi=150, bbox_inches='tight')
    plt.show()
'''

BTSD = r'''
# ---- Test 3: Belgian signs (models trained on German signs only) ----
if TEST_BTSD:
    reason = external_unavailable(cfg0, 'btsd')
    if reason:
        print('BTSD skipped:', reason)
    else:
        rows = []
        for m in TEST:
            df, _ = evaluate_run(m, SEED, 'btsd_c', profile=PROFILE_ARG, device=DEVICE, force=FORCE)
            c, k = df[df.corruption == 'clean'].iloc[0], df[df.corruption != 'clean']
            rows.append({'model': m, 'images': int(c.n), 'clean acc %': 100 * c.acc,
                         'clean acc, shared classes %': 100 * c.acc_shared,
                         'corrupted acc %': 100 * k.acc.mean(), 'corrupted acc, shared classes %': 100 * k.acc_shared.mean()})
        btsd = pd.DataFrame(rows).set_index('model')
        btsd.to_csv(f'{OUT}/btsd_seed{SEED}.csv', float_format='%.4f')
        print("'shared classes': the answer may only be one of the 22 sign types that exist in both countries")
        display(btsd.round(2))
'''

CONFUSION = r'''
# ---- Which signs get mixed up? (one model, clean test set) ----
SHOW_MODEL = 'fusion_full' if 'fusion_full' in TEST else TEST[-1]     # change to any name in TEST
_, arrays = evaluate_run(SHOW_MODEL, SEED, 'gtsrb_clean', profile=PROFILE_ARG, device=DEVICE)
cm = arrays['confusion'][2]                     # sets are train, val, test
rate = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(11, 10))
ax.imshow(np.sqrt(rate), cmap='Blues')
ax.set_xticks(range(43), GTSRB_SHORT, rotation=90, fontsize=6)
ax.set_yticks(range(43), GTSRB_SHORT, fontsize=6)
ax.set(xlabel='predicted', ylabel='true', title=f'{SHOW_MODEL}: confusion matrix, clean test (sqrt scale)')
plt.show()

off = cm.copy()
np.fill_diagonal(off, 0)
top = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:10], off.shape))[0]
display(pd.DataFrame([{'true sign': GTSRB_CLASSES[t], 'predicted as': GTSRB_CLASSES[p], 'times': int(off[t, p]),
                       'share of that sign %': round(100 * off[t, p] / cm[t].sum(), 2)}
                      for t, p in top if off[t, p] > 0]))
'''

EXAMPLES = r'''
# ---- Example predictions: the same photos clean and corrupted ----
CORRUPTION, SEVERITY, N = 'fog', 3, 12          # CORRUPTION: any name printed below
print('corruptions:', ', '.join(CORRUPTIONS))
model, cfg = load_model(SHOW_MODEL, SEED, DEVICE, PROFILE_ARG)
x, y = load_test(cfg, 'gtsrb')
idx = np.sort(np.random.RandomState(1).choice(len(x), N, replace=False))
views = [('clean', x[idx])]
if C_READY['gtsrb']:
    xc, yc = load_corrupted_set(cfg, 'gtsrb', CORRUPTION, SEVERITY)
    assert (yc[idx] == y[idx]).all()
    views.append((f'{CORRUPTION}, severity {SEVERITY}', np.asarray(xc[idx])))

fig, axes = plt.subplots(len(views), N, figsize=(1.6 * N, 2.3 * len(views)), squeeze=False)
for r, (title, imgs) in enumerate(views):
    probs = torch.softmax(predict_logits(model, imgs, DEVICE), 1)
    conf, pred = probs.max(1)
    for i, ax in enumerate(axes[r]):
        ax.imshow(imgs[i])
        ax.axis('off')
        ok = int(pred[i]) == int(y[idx[i]])
        ax.set_title(f'{GTSRB_SHORT[int(pred[i])]}\n{100 * float(conf[i]):.0f}%', fontsize=8,
                     color='green' if ok else 'red')
    axes[r][0].text(-0.15, 0.5, title, transform=axes[r][0].transAxes, rotation=90, va='center', ha='right', fontsize=9)
fig.suptitle(f'{SHOW_MODEL}: prediction and confidence (green = right, red = wrong)', fontsize=10)
plt.tight_layout()
plt.show()
print('true labels:', ', '.join(GTSRB_SHORT[int(t)] for t in y[idx]))
'''

UPLOAD = r'''
# ---- Try your own photos ----
# Crop the photo close around one sign (the training photos have a small border), then upload it.
from google.colab import files
from PIL import Image
size = cfg['data']['image_size']
for fname, data in files.upload().items():
    im = Image.open(io.BytesIO(data)).convert('RGB')
    arr = np.asarray(im.resize((size, size), Image.BILINEAR))[None]       # same resizing as the datasets
    p = torch.softmax(predict_logits(model, arr, DEVICE), 1)[0]
    conf, cls = p.topk(5)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 2.8), gridspec_kw={'width_ratios': [1, 2.5]})
    a.imshow(arr[0])
    a.axis('off')
    a.set_title(fname, fontsize=8)
    b.barh([GTSRB_CLASSES[int(c)] for c in cls][::-1], 100 * conf.numpy()[::-1], color='#2a78d6')
    b.set(xlim=(0, 100), xlabel='confidence %', title=f'{SHOW_MODEL}: top 5')
    plt.tight_layout()
    plt.show()
'''

SAVE = r'''
# ---- Download every table and figure of this notebook ----
print('saved in', OUT)
for f in sorted(os.listdir(OUT)):
    print('  ', f)
shutil.make_archive(f'/content/test_results{SUFFIX}', 'zip', OUT)
from google.colab import files
files.download(f'/content/test_results{SUFFIX}.zip')
'''


def main():
    files = project_files()
    rels = [p.relative_to(REPO_ROOT).as_posix() for p in files]
    dirs = sorted({r.rsplit("/", 1)[0] for r in rels if "/" in r})
    order = (ALL_RUNS[:7] + ENSEMBLES[:3] + ALL_RUNS[7:] + ENSEMBLES[3:5]
             + [r for r in V3_RUNS if r != "resnet18_simclr"] + ENSEMBLES[5:])
    cells = [
        ("md", "# RoadLens - test the trained models\n\n"
               "This notebook **trains nothing**. It loads the `best.pt` checkpoints that `RoadLens_standalone.ipynb` "
               "saved on Google Drive and tests them. Tests that the experiments already ran are read from their "
               "cache, so they finish at once.\n\n"
               "**How to run:** `Runtime > Change runtime type > T4 GPU`, check `PROFILE` and `SEED` in the first code "
               "cell (they must match the training run), then `Runtime > Run all` and approve the Drive prompt.\n\n"
               "| Section | What it does |\n|---|---|\n"
               "| 0. Project code | writes the project files (needed to rebuild the networks) |\n"
               "| 1. Setup and checkpoints | lists every finished model on Drive |\n"
               "| 2. Data | GTSRB test set, Belgian signs, corrupted test sets (restored from Drive or built) |\n"
               "| 3. Tests | clean accuracy; 14 corruptions x 5 severities with mCE; Belgian signs |\n"
               "| 4. Look inside | confusion matrix, example predictions, your own photos |\n"
               "| 5. Save | tables and figures to `MyDrive/RoadLens_standalone/test_results*/` and a zip |\n\n"
               "Use the code generated together with the training notebook: a checkpoint is only loaded if its "
               "configuration has not changed since training."),
        ("code", f"DIRS = {dirs!r}\nORDER = {order!r}\nENSEMBLES = {ENSEMBLES!r}\n" + SETTINGS.strip()),
        ("md", f"## 0. Project code\n\nThe same {len(files)} files as in the training notebook."),
        *code_cells(files, rels),
        ("md", "## 1. Setup and checkpoints"),
        ("code", SETUP.strip()),
        ("code", FIND.strip()),
        ("md", "## 2. Data"),
        ("code", DATA_CELL.strip()),
        ("md", "## 3. Tests\nTables are also saved as CSV in `test_results*/`."),
        ("code", CLEAN.strip()),
        ("code", CORRUPTED.strip()),
        ("code", CURVES.strip()),
        ("code", BTSD.strip()),
        ("md", "## 4. Look inside one model"),
        ("code", CONFUSION.strip()),
        ("code", EXAMPLES.strip()),
        ("code", UPLOAD.strip()),
        ("md", "## 5. Save"),
        ("code", SAVE.strip()),
    ]
    target = write_notebook(cells, NOTEBOOK)
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB, {len(cells)} cells, {len(files)} embedded files)")


if __name__ == "__main__":
    main()
