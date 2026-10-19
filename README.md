# RoadLens

Traffic sign classification that stays accurate under image corruptions (PyTorch): data and
corruption benchmark, four models, an AugMix/PRIME-style training method with a JSD consistency
loss, a gated ResNet-18 + DeiT-S fusion, test-time adaptation, six experiments and all figures.

The code is complete and tested. **The full three-seed result tables have not been produced yet**;
they come from running the commands below on a GPU (Colab). What has been run so far is listed in
[Status](#status).

## Layout

```
configs/data.yaml, base.yaml     data + corruption settings, training defaults
configs/runs/<name>.yaml         one file per trained model (inherits base.yaml)
configs/experiments/*.yaml       one file per experiment / the figures
configs/profiles/*.yaml          reduced-scale overlays: smoke, pilot, pilot_4gb
roadlens/data/                   datasets, class maps, corruptions, GTSRB-C generation, GPU batch helpers
roadlens/augment/                batched GPU augmentation ops, AugMix mixing, JSD loss
roadlens/models/                 SmallCNN, timm backbones (ResNet-18, DeiT-S, ...), gated fusion
roadlens/tta/                    test-time BN statistics, TENT
roadlens/eval/                   metrics, evaluation suites, tables, plots, Fourier heat map, Grad-CAM
roadlens/engine.py, runs.py      training loop; named runs with checkpoint / evaluation caching
scripts/                         one entry point per step and per experiment
notebooks/RoadLens_colab.ipynb   everything, in order, for Colab
tests/                           unit tests (python -m pytest tests)
CORRUPTIONS.md                   what was changed in the corruptions for 64 px, and why
```

## Run on Colab (T4)

1. `python scripts/make_bundle.py` writes two files to `~/RoadLens_gdrive/`: `RoadLens.zip` (the code
   plus the cached 64 px datasets, about 0.6 GB) and `RoadLens_colab.ipynb`.
2. Upload both to the top level of My Drive.
3. Open the notebook in Colab, `Runtime > Change runtime type > T4 GPU`, `Runtime > Run all`.

The notebook unpacks the code to `MyDrive/RoadLens/` and the datasets to the local Colab disk,
generates the corrupted test sets in the background while the first models train, runs the six
experiments and the figures, and prints the tables. It starts at the pilot scale; set
`PROFILE = ''` for the full experiments.

- Checkpoints (`runs*/`) and results (`results*/`) are written to `MyDrive/RoadLens/` and survive
  the session. Finished runs and evaluations are skipped when you run the notebook again.
- Per-epoch checkpoints go to the local disk (`ROADLENS_SCRATCH`), not to Drive: every overwrite
  of a file on a mounted Drive leaves the old version in the Drive trash, which would fill the
  quota. So on Colab a training run that is cut off by the end of a session starts again from its
  first epoch (at most about 40 minutes of work); only finished runs persist.
- GTSRB-C (10.9 GB) lives on the local Colab disk and is regenerated in each session, from a fixed
  seed. Pixels are identical as long as the library versions are; to rule out any drift between
  sessions, keep a copy on Drive (`SAVE_C_TO_DRIVE` in the notebook, 13.3 GB).

### Scale: profiles

Every script accepts `--profile`:

| profile | what | outputs |
|---|---|---|
| `--profile smoke` | every script on tiny subsets, 2 epochs, 2 seeds. Checks that the pipeline runs; numbers are meaningless | `runs_smoke/`, `results_smoke/` |
| `--profile pilot` | full training set, **1 seed**, schedules shortened to 40 %, first 2,000 test images. Real but rough numbers | `runs_pilot/`, `results_pilot/` |
| `--profile pilot_4gb` | the pilot with batch size 64, for a GPU with about 4 GB | `runs_pilot_4gb/`, `results_pilot_4gb/` |
| (none) | the full experiments: seeds 0/1/2, full schedules, full test sets | `runs/`, `results/` |

Profiles never overwrite each other. Start with the pilot.

## Run locally

```bash
pip install -r requirements.txt
export ROADLENS_DATA=/path/with/15GB/free     # optional; default ./datasets
export ROADLENS_RUNS=/path/for/checkpoints    # optional; default ./runs
python -m pytest tests -q                     # 55 unit tests, about 2 minutes
python scripts/benchmark.py                   # speed of every run on this GPU + estimated experiment times
```

## Reproducing every table and figure

Run in this order (add `--profile pilot` to any command for the reduced version). Each experiment
trains what it still needs, so `train.py` is optional.

| # | Output | Command | Files |
|---|---|---|---|
| data | GTSRB / BTSD caches, train/val split | `python scripts/prepare_data.py --datasets gtsrb btsd` | `$ROADLENS_DATA/processed/` |
| data | GTSRB-C (70 sets) + validation-C | `python scripts/make_gtsrb_c.py` | `$ROADLENS_DATA/gtsrb_c_s64/`, `gtsrb_val_c_s64.npz` |
| data | corrupted BTSD (for experiment 5) | `python scripts/make_gtsrb_c.py --dataset btsd` | `$ROADLENS_DATA/btsd_c_s64/` |
| fig | 14 x 5 corruption grids, package-vs-calibrated comparison, PSNR table | `python scripts/plot_corruptions.py` | `results/corruptions/` |
| - | train one model (optional) | `python scripts/train.py --run resnet18_full --seeds 0 1 2` | `runs/<run>/seed<k>/` |
| 1 | clean train / test accuracy, macro-F1, ECE | `python scripts/exp1_clean.py` | `results/exp1_clean/clean.{csv,txt,tex}`, `per_seed.csv` |
| 2 | accuracy / macro-F1 per corruption and severity; clean acc, mean corruption acc, mCE, relative mCE, ECE (**main table**) | `python scripts/exp2_corruption.py` | `results/exp2_corruption/main.{csv,txt,tex}`, `per_seed.csv`, `per_severity.csv`, `per_corruption.{csv,txt,tex}`, `per_category.csv` |
| 3 | ablation baseline -> +mixing -> +JSD -> +Fourier ops -> +fusion -> +TENT (**ablation table**) | `python scripts/exp3_ablation.py` | `results/exp3_ablation/ablation.{csv,txt,tex}`, `per_seed.csv`, `per_severity.csv` |
| 4 | leave-category-out | `python scripts/exp4_leave_category_out.py` | `results/exp4_leave_category_out/candidates.*`, `selection.*` |
| 5 | cross-dataset, clean and corrupted | `python scripts/exp5_cross_dataset.py` | `results/exp5_cross_dataset/btsd.{csv,txt,tex}`, `btsd_per_seed.csv` |
| 6 | TENT / test-time BN vs. batch size 1, 16, 64, 200 | `python scripts/exp6_tent_batchsize.py` | `results/exp6_tent_batchsize/sweep.{csv,txt,tex}`, `tent_batchsize.pdf` |
| 7 | the v2 approach, one mechanism per row; test-time BN with a statistics prior | `python scripts/exp7_improved.py` | `results/exp7_improved/improved.{csv,txt,tex}`, `per_corruption.*`, `severity5.*`, `selection.csv`, `adaptation.{csv,txt,tex}` |
| fig | accuracy-vs-severity curves per corruption, per model | `python scripts/make_figures.py --only curves` | `results/figures/severity_curves.pdf` |
| fig | confusion matrices: clean, and the baseline's worst corruption at severity 5, baseline and proposed | `python scripts/make_figures.py --only confusion` | `results/figures/confusion_<run>_<condition>.pdf` |
| fig | reliability diagrams (clean, GTSRB-C) | `python scripts/make_figures.py --only reliability` | `results/figures/reliability.pdf` |
| fig | Fourier heat map of error for ResNet-18, DeiT-S, fusion | `python scripts/make_figures.py --only fourier` | `results/figures/fourier_heatmap.pdf` |
| fig | Grad-CAM, clean vs. corrupted samples | `python scripts/make_figures.py --only gradcam` | `results/figures/gradcam.pdf` |

Common options: `--seeds 0 1 2`, `--device cuda:0`, `--wandb` (also log training to Weights & Biases),
`-o key=value` (override a config value, repeatable, e.g. `-o n_per_set=2000`). Every `.tex` file is a
booktabs table; every figure is a vector PDF with its numbers in a CSV next to it (Grad-CAM panels
and the corruption grids are photographs, so those PDFs embed raster images).

To change what an experiment covers, edit its YAML in `configs/experiments/`. To add a model or a
training variant, add a file to `configs/runs/` and name it in an experiment. If you change a run's
config after it was trained, the code refuses to reuse the old checkpoint: delete
`runs/<run>/` or give the variant a new name.

## What the code does

**Models** (`configs/runs/`). All take 64 px images in [0, 1]; normalisation and resizing happen inside the model.

| run | model | training |
|---|---|---|
| `smallcnn_base` | SmallCNN, 1.2 M parameters, from scratch | standard; also the mCE normaliser |
| `resnet18_base`, `deit_s_base` | ImageNet-pretrained (timm), input upsampled to 128 px | standard (mild colour jitter only) |
| `resnet18_mix` / `resnet18_mix_jsd` | ResNet-18 | AugMix mixing / + JSD (ablation steps) |
| `resnet18_full`, `deit_s_full` | ResNet-18, DeiT-S | proposed method: AugMix + PRIME-style ops, JSD |
| `fusion_full` | **proposed**: gated fusion of `resnet18_full` + `deit_s_full` | only the gate is trained (branches frozen) |
| `ensemble_conf`, `ensemble_avg` | same two branches, confidence-weighted / plain average | nothing to train |
| `resnet18_jsd_prime`, `resnet18_jsd_nogeom` | ResNet-18 | extra op sets for experiment 4 |

- **Fusion**: `p(y|x) = sum_k w_k(x) softmax(z_k(x))`. The gate is an MLP over the branches' pooled
  features and probabilities; it starts as the plain average. The model returns `log p`, so it is
  trained, calibrated and adapted like any classifier.
- **Training method**: AugMix-style mixing (width 3, depth 1-3, Dirichlet/Beta weights) of ops drawn
  from `autocontrast, equalize, posterize, solarize, rotate, shear_x/y, translate_x/y` plus PRIME-style
  `fourier_noise, spectral_filter, spatial_warp, color_curve`; loss = cross-entropy on the clean view
  + 12 x JSD(clean, view 1, view 2). AdamW, warm-up + cosine, label smoothing 0.1, mixed precision,
  early stopping on validation-C accuracy. The op set is `augment.ops` / `augment.exclude_ops`.
  Augmentation runs batched on the GPU (`roadlens/augment/`), so no DataLoader workers are needed.
- **Test-time adaptation** (`roadlens/tta/`): test-time BN statistics, and TENT (one Adam step per
  batch on the BatchNorm scale/shift parameters, entropy objective). Both act on BatchNorm layers,
  i.e. SmallCNN, ResNet-18 and the ResNet branch of the fusion.
- **Metrics** (`roadlens/eval/metrics.py`): accuracy, macro-F1, ECE (15 equal-width bins), NLL.
  `mCE = 100 x mean_c [sum_s E(c,s) / sum_s E_SmallCNN(c,s)]` and relative mCE with the clean error
  subtracted on both sides; SmallCNN's errors are averaged over its seeds, so SmallCNN scores 100.
  Tables report mean +- sample standard deviation over seeds.

## The v2 approach (experiment 7)

The pilot showed five concrete weaknesses of the models above. v2 answers each with one mechanism;
experiment 7 measures them one row at a time. It ran in the 15 Oct 2026 pilot (one seed, 2,000 test
images; `RoadLens_pilot_results.pdf`): view experts were the strongest idea (GTSRB-C 90.8 % -> 92.9 %,
mCE 73 -> 65), the stacked gate did not beat averaging them, and temperature scaling cut the clean ECE
from 9.0 % to 0.5 %.

| weakness measured in the pilot | mechanism | where |
|---|---|---|
| contrast, fog and brightness at high severity (contrast 38-81 % at severity 5) | **instance input normalisation**: every image is standardised with its own per-channel mean and std, so a per-channel gain and offset, which is exactly what the contrast corruption is, never reaches the network | `model.input_norm: instance`; runs `resnet18_in`, `deit_s_in` |
| the three noise corruptions are 35 % of the fusion's remaining error, while each blur costs under 1 point | **view experts**: every branch also sees a 3x3-median and a Gaussian-blurred copy of the image, trading pixel noise for a little blur | `model.fusion.views`; runs `ensemble_views`, `ensemble_views_in` |
| the learned gate was worse than the plain average | **stacked gate**: the gate is fitted on one half of the validation split (images its branches never trained on) and selected on the other half; the untrained gate, i.e. the average, is a candidate, so selection cannot do worse than averaging on that data. Its inputs are confidence statistics and five image statistics, not raw features | `train.fit_split: val`, `train.eval_at_init`; run `fusion_v2` |
| ECE of 8-17 % (underconfidence from label smoothing) | **temperature scaling** fitted on held-out validation data | `evaluate_run(..., calibrate=True)`, `roadlens/eval/calibration.py` |
| test-time BN and TENT collapse at batch size 1 | **statistics prior**: batch statistics are mixed with the source statistics, n : N | `tta: {prior: 16}`, `PriorBatchNorm` |

Read the experiment 7 table with these in mind:

- **The mechanisms were chosen after looking at per-corruption test results.** No test image or test
  corruption is used for training, fitting or selection, but the *design* is informed by the test
  categories, so GTSRB-C is no longer a blind test of v2. BTSD-C uses the same corruption types and
  does not fix that; an independent check needs corruptions that were never looked at.
- **Use "Val-C (sel.)" to choose between rows**, not the test columns. It is accuracy under the four
  validation corruptions on the select half of the validation split.
- **The gate may select the average.** `selection.csv` records the epoch that was kept; epoch 0 means
  training did not beat the untrained gate and `fusion_v2` equals `ensemble_views_in`.
- **The low-pass views assume blur robustness.** They can cost accuracy on already-blurred images;
  the per-corruption table shows whether they do.
- **View experts triple the inference cost** (three forward passes per branch).
- **Instance normalisation discards the absolute colour level** of an image (not its colour pattern).
- Existing runs are untouched: v2 only adds run configs, and every new option defaults to the old behaviour.

## The v3 approach (experiments 8 and 9)

The 15 Oct pilot left four measured problems: **fog** (34 % at severity 5 for the final v2 model, and
no method moved it; instance normalisation did not help fog, 69 % -> 68 %), **pixel noise** (gaussian
65 %, shot 74 % at S5), **a gate that does not beat averaging**, and **cost**: the best model needs six
forward passes per image. v3 adds one mechanism per problem, a study of other architectures and
learning algorithms, and GPU speed-ups. **None of v3 has been run on a GPU yet**; there are no v3
numbers until experiment 8 has run.

| problem | mechanism | where |
|---|---|---|
| fog: a spatially varying veil that a per-image normalisation cannot remove | **local-global contrast normalisation stem**: each pixel is also standardised by its Gaussian neighbourhood (sigma 4 px), with a floor relative to the image's own contrast so that the result stays exactly invariant to gain and offset and flat regions are not amplified; a learnable per-channel blend with instance normalisation keeps absolute colour | `model.input_norm: local` (`roadlens/models/base.py`); run `resnet18_ln` |
| noise and haze disturb the amplitude spectrum, shape lives in the phase | **amplitude-phase recombination (APR)**: an AugMix op that keeps an image's Fourier phase and moves its amplitude towards another sign's | op set `apr` (`roadlens/augment/ops.py`); run `resnet18_ln_apr` |
| short schedules, noisy final weights | **EMA of the weights**, validated and saved instead of the raw weights | `train.ema`; run `resnet18_v3` |
| six forward passes per image | **consistent distillation**: one network is trained to match the six-expert teacher on exactly the same augmented views (plus the usual CE + JSD); one forward pass at test time | `train.distill`; runs `student_v3` (ResNet-18), `student_mbv3` (MobileNetV3) |
| labels only teach what the label needs | **SimCLR self-supervised pre-training** on the GTSRB training images (labels unused), with the v3 ops as views, then supervised fine-tuning | `train.objective: simclr` (`roadlens/ssl.py`), `model.init_from`; runs `resnet18_simclr`, `resnet18_v3_ssl` |
| is ResNet-18 / DeiT-S the right backbone? | **architecture study** with the v3 recipe: ConvNeXt-Atto (LayerNorm, large kernels), EfficientNet-B0, MobileNetV3-Large, and ResNet-18 at the native 64 px | runs `convnext_atto_v3`, `efficientnet_b0_v3`, `mobilenetv3_v3`, `resnet18_v3_64` |
| training speed on the T4 | channels-last memory layout, fused AdamW, cuDNN autotuning (optional `torch.compile`) | `train.channels_last`, `fused_optimizer`, `cudnn_benchmark`, `compile` |

**Experiment 8** (`scripts/exp8_v3.py`) reports every row like experiment 7 (Val-C (sel.), clean,
corrupted, S5, mCE, ECE, BTSD), a per-family table with fog at S5, and an efficiency table:
parameters, forward passes per image, training minutes including dependencies, GPU images/s at batch
256 and milliseconds per image at batch 1.

**Experiment 9** (`scripts/exp9_classical_unsupervised.py`) compares other learning algorithms on the
same test sets: classical supervised learners on HOG features (linear SVM, random forest, PCA + kNN),
kNN and logistic regression on the label-free SimCLR features, and an unsupervised analysis: k-means
(k = 43) on each model's clean test features, then the share of corrupted images that land in the
cluster of their clean original (a robustness score computed without labels) and the NMI with the classes.

Caveats, as for v2: the v3 mechanisms were designed after looking at the pilot's per-corruption
results, so GTSRB-C is not a blind test of them; choose rows with Val-C (sel.). APR mixes real sign
spectra and adds no noise or blur, but it does change contrast, which is close to the contrast and fog
corruptions; its effect on those two must be read with that in mind. Older runs keep their
fingerprints (`tests/test_v3.py` checks the pilot's values), so finished checkpoints are reused.

## Things to know when reading the results

- **Validation split and model selection.** 10 % of train is held out by *track* (the ~30 frames of
  one physical sign), so no near-duplicate frames leak. Early stopping uses validation-C, built from
  four corruptions that are not test corruptions (speckle noise, Gaussian blur, spatter, saturate).
  They are relatives of test corruptions (as in ImageNet-C), so selection is not fully independent of
  the noise and blur categories.
- **Augmentation / test-corruption overlap.** No training op is one of the 14 test corruptions, but
  some are related, and you should know which before claiming generalisation:
  `rotate, shear, translate, spatial_warp` vs. `elastic_transform` (which is a random affine plus a
  smooth warp); `spectral_filter` (random 3x3 filter) vs. mild blur; `color_curve` and the baseline
  colour jitter vs. `brightness` / `contrast`. `resnet18_jsd_nogeom` removes the geometric group, and
  experiment 4 measures how an op set chosen on some categories transfers to the others.
- **Fourier heat map.** The proposed training uses `fourier_noise` (single Fourier-basis waves). The
  heat map perturbs images along exactly such bases, so for models trained with that op (the fusion
  and the `*_full` runs) the map is not an independent test; compare it with `resnet18_jsd_nogeom`
  -style variants, or drop the op, if you need one.
- **The gate may end up close to an ensemble.** Its branches are near-perfect on the training data,
  so the gate only learns from augmented views on which they disagree. `ensemble_avg` and
  `ensemble_conf` are in the corruption table precisely so that the gate's contribution is visible.
- **"+ TENT" in the ablation** adapts the ResNet branch of the fusion online (batch size 64, full
  test sets, reset before every corruption set, stream shuffled with a fixed seed). TENT minimises
  the entropy of that branch's own prediction (`objective: branch`); `objective: fused` is available.
- **Experiment 6** uses a fixed random subset per corruption (1,000 images; 250 for batch size 1,
  where each image is its own optimisation step) at severity 5. TENT's gain grows with stream
  length, so its numbers there are lower bounds; all methods in a column see the same images.
- **Cross-dataset.** BTSD: 22 of its 62 classes are the same sign as a GTSRB class (2,734 images,
  class map checked by eye, `roadlens/data/mappings/btsd_to_gtsrb.yaml`). "Acc." keeps all 43 outputs;
  "Acc. (shared)" restricts the arg-max to the 22 shared classes. Some shared classes have very few
  images, so macro-F1 is noisy. **CTSD is skipped** until you download it and verify its draft class
  map (`python scripts/prepare_data.py --datasets ctsd` prints the instructions).
- **Resolution.** Images are resized to 64 px first, then corrupted. The pretrained models upsample
  to 128 px internally (`model.input_size`); this is a compute choice, not a tuned value.
- **Determinism.** Same seed, same hardware and library versions give the same weights (cuDNN is set
  to deterministic kernels; a unit test checks that an interrupted run resumes bit-identically when
  its per-epoch checkpoint is still there). Results differ slightly between GPU models and between
  GPU and CPU.
- **Corruption severities** were strengthened on 2026-10-15 (RoadLens "strong" calibration: ImageNet-C
  noise constants, own ladders for blur, zoom, pixelate, JPEG, fog, frost and elastic). They no longer
  follow Tiny-ImageNet-C, and pilot results from 2026-10-10 used the old, milder sets. See `CORRUPTIONS.md`.

## Status

What has actually been run (on a laptop RTX 3050, 4 GB). **Nothing has been run on Colab yet.**

- **Unit tests**: 39 pass (`python -m pytest tests`): augmentation ops against PIL / torchvision
  references, metrics, models and fusion, Grad-CAM, test-time adaptation, exact resume.
- **`--profile smoke`**: every script (experiments 1-6 and all figures) ran end to end on tiny
  subsets. That checks the plumbing, nothing else.
- **A one-seed pilot was started and stopped early.** Two runs finished: SmallCNN (12 epochs:
  99.9 % clean validation accuracy, 95.1 % on validation-C) and the ResNet-18 baseline (6 epochs:
  99.5 % / 87.5 %). DeiT-S reached 98.2 % / 92.1 % after its first epoch.
- **Not yet run at real scale**: the proposed training (AugMix / PRIME ops + JSD), the gated fusion,
  TENT, and every experiment table and figure. Their hyper-parameters are standard choices that
  have only been exercised at smoke scale. The pilot profile on Colab is their first real test, so
  run it before the full experiments.
- **Full three-seed results**: not produced yet (`results/` holds only the stage-1 corruption figures).
- **CTSD**: not available; experiment 5 covers BTSD only until CTSD is downloaded and its class map checked.

## Run times

Not measured on a T4. Rough planning numbers, from laptop measurements and typical T4 throughput:

| profile | GPU time on a T4 (estimate) |
|---|---|
| `smoke` | about 15 minutes |
| `pilot` | about 1-1.5 hours |
| full (3 seeds) | about 10-12 hours (training ~7 h, evaluation ~4 h): two or three Colab sessions |

`python scripts/benchmark.py` (about 3 minutes; `RUN_CHECKS` in the notebook) pushes random images
through the real training step of every run and prints a measured estimate per experiment for the
GPU it runs on. To split the full run, pass `--seeds 0` first and `--seeds 0 1 2` later: finished
seeds are reused.

Measured on the laptop (RTX 3050 Laptop, batch 64, plugged in): SmallCNN about 1,800 images/s,
ResNet-18 baseline about 850, DeiT-S baseline about 280. The proposed training processes three
views of every image, so an epoch costs 3-4 times the baseline epoch of the same model.

Other costs: generating the corrupted sets is CPU-only (4 minutes for the 2,000-image pilot subset
with 6 processes on the laptop; expect about 45 minutes for the full sets on one Colab core, in the
background). Disk: 10.9 GB GTSRB-C + 2.4 GB corrupted BTSD on the Colab disk; all checkpoints of the
full run together take about 1.4 GB on Drive.
