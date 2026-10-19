# Corruptions at 64 px

GTSRB-C = the 12,630 GTSRB test images, resized to 64x64 **first**, then corrupted with
14 corruptions x 5 severities (70 test sets). Implementation: `roadlens/data/corruptions.py`.
Reproduce the figures and numbers below with `python scripts/plot_corruptions.py`
(outputs in `results/corruptions/`).

## Current ladders (since 2026-10-15): RoadLens "strong" calibration

At the user's request every test corruption was made stronger on 2026-10-15, after reviewing the
corruption grids (no model results were used to choose the values). The Tiny-ImageNet-C constants
described further down were judged too mild: at severity 5 most signs were still easy to read, and
several ladders barely changed between severities. The new rule:

- **Noise** (gaussian, shot, impulse): the original ImageNet-C constants (they act on pixel
  intensities, so they do not depend on image size).
- **Blur, zoom, pixelate, JPEG, fog, frost**: RoadLens's own values, chosen so that severity 1 is
  clearly visible, severity 5 is about as hard as ImageNet-C is on 224 px images, and the damage
  grows evenly in between.
- **Elastic**: rescaled warp, 0.75 to 4 px average displacement (section below).
- **Snow, brightness, contrast**: unchanged (already strong; same values as ImageNet-C for
  brightness and contrast).
- **Validation corruptions** (speckle, gaussian blur, spatter, saturate): unchanged.

| corruption | severity 1 -> 5 (current) | Tiny-ImageNet-C (before) |
|---|---|---|
| gaussian_noise (sigma) | 0.08, 0.12, 0.18, 0.26, 0.38 | 0.04 ... 0.18 |
| shot_noise (photons, lower = worse) | 60, 25, 12, 5, 3 | 250 ... 15 |
| impulse_noise (fraction) | 0.03, 0.06, 0.09, 0.17, 0.27 | 0.01 ... 0.14 |
| defocus_blur (radius px, alias) | (1, 0.1), (1.5, 0.1), (2, 0.1), (3, 0.5), (4, 0.5) | radius 0.5 ... 3 |
| motion_blur (radius, sigma px) | (4, 1.2), (6, 2), (6, 3.2), (6, 4.8), (8, 6) | sigma 1 ... 3 |
| zoom_blur (max zoom, step) | 1.16/0.02, 1.24/0.02, 1.32/0.03, 1.40/0.03, 1.50/0.04 | 1.06 ... 1.26 |
| fog (amplitude, decay) | (0.7, 2.5), (1.1, 2.2), (1.5, 2), (2.2, 1.7), (3, 1.4) | 0.4 ... 2 |
| frost (image, frost weight) | (1, 0.35), (0.9, 0.5), (0.8, 0.6), (0.7, 0.7), (0.6, 0.8) | frost 0.3 ... 0.55 |
| pixelate (scale) | 0.65, 0.52, 0.43, 0.36, 0.30 | 0.9 ... 0.5 |
| jpeg_compression (quality) | 45, 30, 20, 12, 8 | 65 ... 25 |
| elastic_transform (RMS warp px) | 0.75, 1.5, 2.25, 3, 4 | 0 ... 1.2 |
| snow, brightness, contrast | unchanged | |

Mean PSNR against the clean image (150 test images), severity 1 -> 5; every ladder now falls
monotonically:

| corruption | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| gaussian_noise | 22.4 | 19.2 | 16.1 | 13.4 | 10.8 |
| shot_noise | 24.1 | 20.4 | 17.3 | 13.8 | 11.8 |
| impulse_noise | 19.8 | 16.8 | 15.0 | 12.3 | 10.3 |
| motion_blur | 33.3 | 29.3 | 26.2 | 24.2 | 23.2 |
| zoom_blur | 27.4 | 24.8 | 23.6 | 22.4 | 21.8 |
| defocus_blur | 40.2 | 36.7 | 34.5 | 30.0 | 27.9 |
| snow | 17.5 | 16.3 | 13.2 | 11.2 | 10.2 |
| fog | 19.9 | 18.1 | 17.2 | 15.9 | 15.3 |
| frost | 13.5 | 11.0 | 10.0 | 9.0 | 8.6 |
| brightness | 21.7 | 15.6 | 12.2 | 9.8 | 8.1 |
| contrast | 22.6 | 21.2 | 20.1 | 19.0 | 18.6 |
| pixelate | 34.1 | 32.8 | 30.6 | 29.6 | 28.2 |
| elastic_transform | 28.8 | 25.2 | 23.3 | 21.9 | 20.8 |
| jpeg_compression | 33.5 | 32.1 | 30.6 | 28.5 | 26.9 |

Consequences: GTSRB-C is no longer the published Tiny-ImageNet-C protocol (say so in the paper);
noise at severities 4-5 can make dark images unreadable, as in ImageNet-C; and all results from the
2026-10-10 pilot were measured on the old, milder sets, so they are not comparable with new runs.
The rest of this document describes the original Tiny-ImageNet-C calibration and is kept as history.

## Why the `imagecorruptions` defaults are not used as-is

`imagecorruptions` hard-codes the ImageNet-C severity constants, which were tuned for 224+ px
images. Several of them are expressed in pixels, so at 64 px they are ~3.5x too strong.
`results/corruptions/package_vs_roadlens.pdf` shows both side by side on the same sign:

| corruption | package default at 64 px | problem |
|---|---|---|
| motion_blur | kernel radius 10-20 px, sigma 3-15 px | streak up to 2/3 of the image; sign gone from severity 3 |
| defocus_blur | disk radius 3-10 px | uniform blob from severity 3 |
| elastic_transform | displacement amplitude fixed at 12.5-30 (not scaled with size) | pixel "dust" at every severity, nothing like a warp |
| frost | crops 64x64 from a full-resolution (300-900 px) texture | the crop is one flat patch, not frost crystals |
| fog | plasma amplitude 1.5-3.0 | sign already washed out at severity 1, barely visible by 5 |
| pixelate | down to 25% (a 16x16 grid) | digits unreadable to a human at severity 4-5 |
| jpeg_compression | quality 25 down to 7 | 8x8 blocks are 1/8 of the image side; colour garbage at severity 5 |
| gaussian/shot/impulse noise | sigma up to 0.38, etc. | each pixel carries far more of the sign at 64 px than at 224 px |

Two further defects, independent of resolution, on current library versions
(numpy 2.x, scikit-image >= 0.19; this is what Colab ships):

- `fog` crashes (`np.float_` was removed) and `gaussian_blur` / `glass_blur` crash
  (`multichannel=` argument was removed).
- `impulse_noise` is **not reproducible**: `skimage.util.random_noise` now draws from a fresh
  generator and ignores `np.random.seed`, so a fixed seed does not give a fixed test set.

## What was used before 2026-10-15 (history)

**Severity constants:** those of Tiny-ImageNet-C (`make_tinyimagenet_c.py` in
[hendrycks/robustness](https://github.com/hendrycks/robustness)), i.e. the constants the
ImageNet-C authors themselves calibrated for 64x64 images. They were copied verbatim for all
14 test corruptions and the 4 validation corruptions; nothing was tuned by hand or against any
model.

**Code:** the `imagecorruptions` algorithms and helpers (`disk` kernel, `_motion_blur`, frost
textures), with these changes:

| item | change | reason |
|---|---|---|
| all severity constants | Tiny-ImageNet-C values (table below) | calibrated for 64 px |
| elastic_transform | Tiny-ImageNet-C structure (random affine + smooth displacement field, all magnitudes proportional to image side), but the field is rescaled to a set RMS displacement and the ladder is RoadLens's own (see below) | the package variant has a size-independent amplitude; the Tiny-ImageNet-C constants barely move a 64 px sign |
| frost | texture shrunk to 0.3x (x size/64) before cropping, as in Tiny-ImageNet-C | visible crystal structure at 64 px |
| zoom_blur / snow | zoomed layer is trimmed symmetrically (Tiny-ImageNet-C `clipped_zoom`) | the package trims from the top-left corner instead (a sub-pixel to 1 px shift at this size) |
| pixelate | BOX down, BOX up (Tiny-ImageNet-C) instead of BOX down, NEAREST up | follows the reference |
| impulse_noise | salt & pepper written with `np.random` | reproducibility (see above) |
| fog, gaussian_blur | `np.float64`, `channel_axis=-1` | crash fixes, same maths |
| motion_blur, snow | use the package's pure-numpy `_motion_blur` with the Tiny-ImageNet-C (radius, sigma) | the reference uses ImageMagick (Wand); not identical pixel-for-pixel, same parameters |

Unchanged from the package: brightness, contrast, saturate (identical constants, resolution-independent).
Kept upstream quirk: frost draws from the first five of the six textures.

### Constants (severity 1 -> 5)

| corruption | RoadLens (64 px) | `imagecorruptions` default |
|---|---|---|
| gaussian_noise (sigma) | 0.04, 0.08, 0.12, 0.15, 0.18 | 0.08, 0.12, 0.18, 0.26, 0.38 |
| shot_noise (photons) | 250, 100, 50, 30, 15 | 60, 25, 12, 5, 3 |
| impulse_noise (fraction) | 0.01, 0.02, 0.05, 0.08, 0.14 | 0.03, 0.06, 0.09, 0.17, 0.27 |
| defocus_blur (radius px, alias) | (0.5, 0.6), (1, 0.1), (1.5, 0.1), (2.5, 0.01), (3, 0.1) | (3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5) |
| motion_blur (radius, sigma px) | (10, 1), (10, 1.5), (10, 2), (10, 2.5), (12, 3) | (10, 3), (15, 5), (15, 8), (15, 12), (20, 15) |
| zoom_blur (max zoom, step) | 1.06, 1.11, 1.16, 1.21, 1.26, step 0.01 | 1.11/0.01, 1.16/0.01, 1.21/0.02, 1.26/0.02, 1.31/0.03 |
| snow | see `PARAMS["snow"]` | see package |
| fog (amplitude, decay) | (0.4, 3), (0.7, 3), (1, 2.5), (1.5, 2), (2, 1.75) | (1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4) |
| frost (image, frost weight) | (1, 0.3), (0.9, 0.4), (0.8, 0.45), (0.75, 0.5), (0.7, 0.55) | (1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75) |
| brightness | 0.1, 0.2, 0.3, 0.4, 0.5 | same |
| contrast | 0.4, 0.3, 0.2, 0.1, 0.05 | same |
| pixelate (scale) | 0.9, 0.8, 0.7, 0.6, 0.5 | 0.6, 0.5, 0.4, 0.3, 0.25 |
| elastic_transform (RMS warp, smoothness sigma, affine; x side) | (0.012, 0.0625, 0.02), (0.023, 0.0625, 0.03), (0.035, 0.0625, 0.04), (0.047, 0.0625, 0.05), (0.0625, 0.0625, 0.06), i.e. 0.75 / 1.5 / 2.25 / 3 / 4 px average warp | different algorithm |
| jpeg_compression (quality) | 65, 58, 50, 40, 25 | 25, 18, 15, 10, 7 |
| speckle_noise (val) | 0.15, 0.2, 0.25, 0.3, 0.35 | 0.15, 0.2, 0.35, 0.45, 0.6 |
| gaussian_blur (val, sigma px) | 0.5, 0.75, 1, 1.25, 1.5 | 1, 2, 3, 4, 6 |
| spatter (val) | see `PARAMS["spatter"]` | see package |
| saturate (val) | (0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (30, 0.2) | (0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2) |

### Measured strength

Mean PSNR (dB) against the clean image over 300 random GTSRB test images
(`results/corruptions/corruption_stats.csv`; lower = stronger):

| corruption | package s1 / s3 / s5 | RoadLens s1 / s3 / s5 |
|---|---|---|
| gaussian_noise | 22.4 / 16.1 / 10.8 | 28.1 / 19.2 / 16.1 |
| shot_noise | 23.9 / 17.2 / 11.7 | 30.0 / 23.2 / 18.1 |
| impulse_noise | 19.8 / 15.0 / 10.2 | 24.6 / 17.6 / 13.1 |
| motion_blur | 25.9 / 21.6 / 19.5 | 34.5 / 28.6 / 26.0 |
| zoom_blur | 27.9 / 24.9 / 23.1 | 30.9 / 26.1 / 23.5 |
| defocus_blur | 29.8 / 24.5 / 22.2 | 41.4 / 36.2 / 29.8 |
| snow | 15.7 / 11.0 / 8.2 | 17.5 / 13.2 / 10.3 |
| fog | 16.6 / 15.2 / 14.7 | 22.3 / 18.1 / 15.7 |
| frost | 12.3 / 9.1 / 9.0 | 14.9 / 13.2 / 11.9 |
| brightness | 21.8 / 12.4 / 8.3 | 21.8 / 12.4 / 8.3 |
| contrast | 22.0 / 19.5 / 18.0 | 22.0 / 19.5 / 18.0 |
| pixelate | 33.0 / 29.6 / 26.6 | 38.9 / 34.1 / 32.1 |
| elastic_transform | 26.6 / 23.1 / 21.2 | 21.2 / 22.5 / 26.2 |
| jpeg_compression | 31.2 / 29.2 / 26.1 | 34.6 / 33.5 / 31.2 |

PSNR is only a rough guide; it over-penalises geometric changes (a small shift moves many pixels
but barely hurts legibility). The elastic row above is from the original Tiny-ImageNet-C ladder; the
current numbers are in `results/corruptions/corruption_stats.csv`.

### Elastic transform: the one ladder that is not Tiny-ImageNet-C

Changed on 2026-10-15 at the user's request, after looking at the corruption grids (no model
results were used to choose the new values). With the Tiny-ImageNet-C constants the smooth warp is
multiplied by a fixed amplitude *after* Gaussian smoothing, and the smoothing shrinks it: at 64 px
the average warp was 0 / 0.05 / 0.32 / 0.8 / 1.24 px for severities 1-5, so severities 1-3 were
little more than a small tilt and the ladder was inverted in PSNR.

Now the smoothed field is rescaled so that its root-mean-square displacement is exactly
0.75 / 1.5 / 2.25 / 3 / 4 px (1.2 % to 6.25 % of the side), with a fixed smoothness of 4 px
(6.25 % of the side) so it stays a warp and never turns into pixel noise, and the affine jitter
grows with severity (up to 1.3 / 1.9 / 2.6 / 3.2 / 3.8 px on the corner points). At severity 5 the
sign outline is strongly wavy but the symbol is still recognisable. The pilot results (2026-10-10)
were computed with the old elastic ladder and are not comparable on that corruption.

## Things to know when reading results

- **Ladders were strengthened on 2026-10-15** (see the top of this file), on visual grounds and at
  the user's request. That decision came after the pilot had been run, so be explicit about it.
- **contrast 4-5** leaves an almost uniform image (factor 0.1 / 0.05) - same as ImageNet-C.
- **elastic_transform includes a random affine** (shear/rotation/shift of up to 8% of the side at
  severity 1) followed by a smooth random warp. The geometric training augmentations (rotate, shear,
  translate, spatial_warp) therefore overlap with this test corruption. The run
  `resnet18_jsd_nogeom` trains without them (`augment.exclude_ops: [geometric]`), and experiment 4
  compares it with the full op set. The README lists the other related op / corruption pairs.
- **Validation corruptions are relatives of test corruptions** (speckle ~ gaussian/shot noise,
  gaussian blur ~ defocus blur), as in ImageNet-C. Model selection on validation-C is not fully
  independent of the noise and blur test categories.
- **Other resolutions.** The constants are calibrated for 64 px. `data.image_size` can be changed,
  but pixel-denominated blur parameters are not rescaled (frost and elastic are); the generator
  warns outside 48-96 px.

## Reproducibility

Each image is corrupted with its own seed derived from
`(corruptions.seed, corruption id, severity, image index)`, so results do not depend on the number
of worker processes, on generation order, or on which other corruptions are generated.
`manifest.json` stores a fingerprint of the clean data, seed and constants; the generator refuses
to mix sets produced under different settings.
