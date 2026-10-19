"""Image corruptions calibrated for ~64 px traffic-sign crops.

The algorithms and helper kernels come from the `imagecorruptions` package (itself a port of
Hendrycks & Dietterich's ImageNet-C code). The package hard-codes severity constants tuned for
224+ px images, so every corruption here is re-parameterised; see CORRUPTIONS.md for the table
of changes and the reasons. All functions take a uint8 HxWx3 RGB array and return uint8.

Randomness goes through the global `np.random` state only, so `corrupt(..., seed=...)` is
fully reproducible.
"""

from __future__ import annotations

import os
import sys
import types
import warnings
from functools import lru_cache
from io import BytesIO

import cv2
import numpy as np
import skimage.color
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian


def _import_package_helpers():
    """Import imagecorruptions.corruptions, shimming modules it imports that newer stacks dropped."""
    try:
        import pkg_resources  # noqa: F401
    except Exception:  # removed from recent setuptools; the package only needs resource_filename
        shim = types.ModuleType("pkg_resources")
        shim.resource_filename = lambda pkg, rel: os.path.join(
            os.path.dirname(sys.modules[pkg].__file__), rel)
        sys.modules["pkg_resources"] = shim
    try:
        import scipy.ndimage.interpolation  # noqa: F401
    except Exception:
        shim = types.ModuleType("scipy.ndimage.interpolation")
        shim.map_coordinates = map_coordinates
        sys.modules["scipy.ndimage.interpolation"] = shim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import imagecorruptions.corruptions as pkg
    return pkg


_pkg = _import_package_helpers()
_FROST_DIR = os.path.join(os.path.dirname(_pkg.__file__), "frost")
_FROST_FILES = ["frost1.png", "frost2.png", "frost3.png", "frost4.jpg", "frost5.jpg", "frost6.jpg"]

# Resolution the constants below were calibrated for.
REFERENCE_SIZE = 64

# --------------------------------------------------------------------------------------
# Severity constants (index = severity - 1). See CORRUPTIONS.md.
# --------------------------------------------------------------------------------------
PARAMS = {
    # noise: std of additive noise on [0, 1] / photon count / fraction of flipped values
    "gaussian_noise": [0.08, 0.12, 0.18, 0.26, 0.38],
    "shot_noise": [60, 25, 12, 5, 3],
    "impulse_noise": [0.03, 0.06, 0.09, 0.17, 0.27],
    # blur: (disk radius px, anti-alias sigma) / (kernel radius, sigma) px / zoom factors
    "defocus_blur": [(1, 0.1), (1.5, 0.1), (2, 0.1), (3, 0.5), (4, 0.5)],
    "motion_blur": [(4, 1.2), (6, 2), (6, 3.2), (6, 4.8), (8, 6)],
    "zoom_blur": [(1.16, 0.02), (1.24, 0.02), (1.32, 0.03), (1.40, 0.03), (1.50, 0.04)],
    # weather
    "snow": [(0.1, 0.2, 1, 0.6, 8, 3, 0.8), (0.1, 0.2, 1, 0.5, 10, 4, 0.8),
             (0.15, 0.3, 1.75, 0.55, 10, 4, 0.7), (0.25, 0.3, 2.25, 0.6, 12, 6, 0.65),
             (0.3, 0.3, 1.25, 0.65, 14, 12, 0.6)],
    "fog": [(0.7, 2.5), (1.1, 2.2), (1.5, 2), (2.2, 1.7), (3, 1.4)],
    "frost": [(1, 0.35), (0.9, 0.5), (0.8, 0.6), (0.7, 0.7), (0.6, 0.8)],
    "brightness": [0.1, 0.2, 0.3, 0.4, 0.5],
    # digital
    "contrast": [0.4, 0.3, 0.2, 0.1, 0.05],
    "pixelate": [0.65, 0.52, 0.43, 0.36, 0.3],
    # (RMS warp displacement, warp smoothness sigma, affine jitter), each as a fraction of the image side.
    # RoadLens ladder (not Tiny-ImageNet-C): 0.75 -> 4 px average warp at 64 px; see CORRUPTIONS.md.
    "elastic_transform": [(0.012, 0.0625, 0.02), (0.023, 0.0625, 0.03), (0.035, 0.0625, 0.04),
                          (0.047, 0.0625, 0.05), (0.0625, 0.0625, 0.06)],
    "jpeg_compression": [45, 30, 20, 12, 8],
    # held-out (validation-C only)
    "speckle_noise": [0.15, 0.2, 0.25, 0.3, 0.35],
    "gaussian_blur": [0.5, 0.75, 1, 1.25, 1.5],
    "spatter": [(0.62, 0.1, 0.7, 0.7, 0.6, 0), (0.65, 0.1, 0.8, 0.7, 0.6, 0),
                (0.65, 0.3, 1, 0.69, 0.6, 0), (0.65, 0.1, 0.7, 0.68, 0.6, 1),
                (0.65, 0.1, 0.5, 0.67, 0.6, 1)],
    "saturate": [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (30, 0.2)],
}

# Fraction by which frost textures are shrunk before cropping, at REFERENCE_SIZE.
FROST_TEXTURE_SCALE = 0.3

# Stable ids used for seeding and for validation-C metadata. Append only; never reorder.
CORRUPTION_IDS = {name: i for i, name in enumerate(PARAMS)}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _clipped_zoom(img: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Centre-crop then zoom back to (at least) the original size, trimmed symmetrically."""
    h, w = img.shape[:2]
    ch, cw = int(np.ceil(h / zoom_factor)), int(np.ceil(w / zoom_factor))
    top, left = (h - ch) // 2, (w - cw) // 2
    out = scizoom(img[top:top + ch, left:left + cw], (zoom_factor, zoom_factor, 1), order=1)
    tt, tl = (out.shape[0] - h) // 2, (out.shape[1] - w) // 2
    return out[tt:tt + h, tl:tl + w]


def _plasma_fractal(mapsize: int, wibbledecay: float) -> np.ndarray:
    """Diamond-square heightmap in [0, 1] (package version uses the removed `np.float_`)."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    while stepsize >= 2:
        half = stepsize // 2
        # squares
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[half:mapsize:stepsize, half:mapsize:stepsize] = wibbledmean(squareaccum)
        # diamonds
        drgrid = maparray[half:mapsize:stepsize, half:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:mapsize:stepsize, half:mapsize:stepsize] = wibbledmean(ldrsum + lulsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[half:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(tdrsum + tulsum)
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


@lru_cache(maxsize=None)
def _frost_texture(idx: int, size: int) -> np.ndarray:
    """Frost texture (RGB) shrunk so that a size x size crop shows crystal structure, not one blob."""
    tex = cv2.imread(os.path.join(_FROST_DIR, _FROST_FILES[idx]))[..., ::-1]
    scale = FROST_TEXTURE_SCALE * size / REFERENCE_SIZE
    scale = max(scale, (size + 1) / min(tex.shape[:2]))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return np.ascontiguousarray(cv2.resize(tex, (0, 0), fx=scale, fy=scale, interpolation=interp))


def _f(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


# --------------------------------------------------------------------------------------
# Corruptions: f(x uint8 HxWx3, c) -> float array in [0, 255] (clipped/cast by `corrupt`)
# --------------------------------------------------------------------------------------
def gaussian_noise(x, c):
    x = _f(x)
    return (x + np.random.normal(size=x.shape, scale=c)) * 255


def shot_noise(x, c):
    return np.random.poisson(_f(x) * c) / float(c) * 255


def impulse_noise(x, c):
    # Salt & pepper on a fraction c of values. Written out by hand because
    # skimage.util.random_noise draws from its own generator and ignores np.random.seed.
    x = _f(x)
    flip = np.random.uniform(size=x.shape) < c
    salt = np.random.uniform(size=x.shape) < 0.5
    x[flip & salt] = 1.0
    x[flip & ~salt] = 0.0
    return x * 255


def speckle_noise(x, c):
    x = _f(x)
    return (x + x * np.random.normal(size=x.shape, scale=c)) * 255


def gaussian_blur(x, c):
    return gaussian(_f(x), sigma=c, channel_axis=-1) * 255


def defocus_blur(x, c):
    kernel = _pkg.disk(radius=c[0], alias_blur=c[1])
    return cv2.filter2D(_f(x), -1, kernel) * 255


def motion_blur(x, c):
    angle = np.random.uniform(-45, 45)
    return _pkg._motion_blur(x.astype(np.float32), radius=c[0], sigma=c[1], angle=angle)


def zoom_blur(x, c):
    factors = np.arange(1, c[0], c[1])
    x = _f(x)
    out = np.zeros_like(x)
    for z in factors:
        out += _clipped_zoom(x, z)
    return (x + out) / (len(factors) + 1) * 255


def snow(x, c):
    x = _f(x)
    layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    layer = _clipped_zoom(layer[..., np.newaxis], c[2])
    layer[layer < c[3]] = 0
    layer = np.clip(layer.squeeze(-1), 0, 1)
    layer = _pkg._motion_blur(layer, radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    layer = (np.round(layer * 255).astype(np.uint8) / 255.0)[..., np.newaxis]
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
    x = c[6] * x + (1 - c[6]) * np.maximum(x, gray * 1.5 + 0.5)
    return (x + layer + np.rot90(layer, k=2)) * 255


def fog(x, c):
    x = _f(x)
    h, w = x.shape[:2]
    mapsize = 1 << (max(h, w) - 1).bit_length()
    max_val = x.max()
    x = x + c[0] * _plasma_fractal(mapsize, c[1])[:h, :w, np.newaxis]
    return x * max_val / (max_val + c[0]) * 255


def frost(x, c):
    h, w = x.shape[:2]
    idx = np.random.randint(5)  # upstream draws from the first five of six textures
    tex = _frost_texture(idx, max(h, w))
    top = np.random.randint(0, tex.shape[0] - h)
    left = np.random.randint(0, tex.shape[1] - w)
    return c[0] * x.astype(np.float32) + c[1] * tex[top:top + h, left:left + w]


def brightness(x, c):
    x = skimage.color.rgb2hsv(_f(x))
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    return skimage.color.hsv2rgb(x) * 255


def contrast(x, c):
    x = _f(x)
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return ((x - means) * c + means) * 255


def saturate(x, c):
    x = skimage.color.rgb2hsv(_f(x))
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    return skimage.color.hsv2rgb(x) * 255


def jpeg_compression(x, c):
    buf = BytesIO()
    Image.fromarray(x).save(buf, "JPEG", quality=int(c))
    return np.asarray(Image.open(buf).convert("RGB"))


def pixelate(x, c):
    h, w = x.shape[:2]
    im = Image.fromarray(x).resize((int(w * c), int(h * c)), Image.BOX)
    return np.asarray(im.resize((w, h), Image.BOX))


def elastic_transform(x, c):
    image = _f(x)
    shape = image.shape
    h, w = shape[:2]
    side = min(h, w)
    disp, sigma, affine = c[0] * side, c[1] * side, c[2] * side

    # random affine
    center = np.float32([h, w]) // 2
    sq = side // 3
    pts1 = np.float32([center + sq, [center[0] + sq, center[1] - sq], center - sq])
    pts2 = pts1 + np.random.uniform(-affine, affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # smooth random displacement field, rescaled so that its RMS length is exactly `disp` pixels.
    # (Tiny-ImageNet-C multiplies the smoothed noise by a fixed amplitude instead; the smoothing shrinks
    # it so much that its average warp at 64 px is only 0-1.2 px, which barely changes a sign.)
    rx = np.random.uniform(-1, 1, size=(h, w))
    ry = np.random.uniform(-1, 1, size=(h, w))
    if disp > 0:
        dx = gaussian(rx, sigma, mode="reflect", truncate=3)
        dy = gaussian(ry, sigma, mode="reflect", truncate=3)
        scale = disp / max(float(np.sqrt(np.mean(dx ** 2 + dy ** 2))), 1e-8)
        dx, dy = (dx * scale).astype(np.float32), (dy * scale).astype(np.float32)
        yy, xx, zz = np.meshgrid(np.arange(h), np.arange(w), np.arange(shape[2]), indexing="ij")
        idx = (yy + dy[..., None]).reshape(-1, 1), (xx + dx[..., None]).reshape(-1, 1), zz.reshape(-1, 1)
        image = map_coordinates(image, idx, order=1, mode="reflect").reshape(shape)
    return image * 255


def spatter(x, c):
    x = _f(x)
    liquid = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    liquid = gaussian(liquid, sigma=c[2])
    liquid[liquid < c[3]] = 0
    if c[5] == 0:  # water
        liquid = (liquid * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        m = (liquid * dist)[..., np.newaxis].astype(np.float32)
        m = m / max(float(m.max()), 1e-8) * c[4]
        color = np.array([175, 238, 238], dtype=np.float32) / 255.0  # pale turquoise, RGB
        return (x + m * color) * 255
    # mud
    m = np.where(liquid > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0
    color = np.array([63, 42, 20], dtype=np.float32) / 255.0
    return (x * (1 - m[..., np.newaxis]) + color * m[..., np.newaxis]) * 255


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise, "shot_noise": shot_noise, "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur, "motion_blur": motion_blur, "zoom_blur": zoom_blur,
    "snow": snow, "fog": fog, "frost": frost, "brightness": brightness,
    "contrast": contrast, "pixelate": pixelate, "elastic_transform": elastic_transform,
    "jpeg_compression": jpeg_compression,
    "speckle_noise": speckle_noise, "gaussian_blur": gaussian_blur, "spatter": spatter,
    "saturate": saturate,
}
assert set(CORRUPTIONS) == set(PARAMS)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def derive_seed(base_seed: int, name: str, severity: int, index: int) -> int:
    """Per-image seed, independent of processing order and of which other corruptions are generated."""
    ss = np.random.SeedSequence([int(base_seed), CORRUPTION_IDS[name], int(severity), int(index)])
    return int(ss.generate_state(1)[0])


def corrupt(image: np.ndarray, name: str, severity: int, seed: int | None = None) -> np.ndarray:
    """Corrupt one uint8 HxWx3 RGB image. With `seed`, the result is deterministic and the global
    numpy random state is left untouched."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3, got {image.dtype} {image.shape}")
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError("severity must be in 1..5")
    c = PARAMS[name][severity - 1]
    if seed is None:
        out = CORRUPTIONS[name](image, c)
    else:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            out = CORRUPTIONS[name](image, c)
        finally:
            np.random.set_state(state)
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_batch(x: np.ndarray, name: str, severity: int, base_seed: int,
                  start_index: int = 0) -> np.ndarray:
    """Corrupt x[i] with seed derive_seed(base_seed, name, severity, start_index + i)."""
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = corrupt(x[i], name, severity, derive_seed(base_seed, name, severity, start_index + i))
    return out


def package_default(image: np.ndarray, name: str, severity: int, seed: int | None = None) -> np.ndarray:
    """The unmodified `imagecorruptions` corruption (ImageNet-scale constants), for comparison only.

    Patches the two things that crash on current numpy/scikit-image.
    """
    _pkg.plasma_fractal = lambda mapsize=256, wibbledecay=3: _plasma_fractal(mapsize, wibbledecay)

    def _gaussian(img, sigma=1, multichannel=None, **kw):
        if multichannel:
            kw["channel_axis"] = -1
        return gaussian(img, sigma=sigma, **kw)

    _pkg.gaussian = _gaussian
    if seed is not None:
        np.random.seed(seed)
    out = getattr(_pkg, name)(Image.fromarray(image), severity)
    return np.clip(np.asarray(out), 0, 255).astype(np.uint8)
