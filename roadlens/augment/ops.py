"""Batched GPU image operations used as AugMix-style training augmentations.

Every op has the signature ``op(x, level) -> x'`` with

    x      float tensor [B, 3, H, W] in [0, 1]
    level  float tensor [B] in [0, 1]: per-sample magnitude (0 = weakest)

and draws its randomness from torch's global generator on ``x.device``.

Two families:

* AugMix ops (Hendrycks et al. 2020) minus the ones that reproduce a test corruption
  (brightness, contrast, colour, sharpness are left out): autocontrast, equalize, posterize,
  solarize, rotate, shear_x/y, translate_x/y. Magnitude ranges follow the AugMix reference code.
* PRIME-style primitives (Modas et al. 2022), re-implemented here with their own documented
  parameterisation: fourier_noise, spectral_filter, spatial_warp, color_curve.

None of the ops is one of the 14 test corruptions. Related pairs are listed in the README
("Augmentation / test-corruption overlap") so they can be ablated with `augment.exclude_ops`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF


def _sign(level: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.rand_like(level) < 0.5, -1.0, 1.0)


def _to_u8(x: torch.Tensor) -> torch.Tensor:
    return (x * 255.0).round().clamp_(0, 255).to(torch.uint8)


def _from_u8(x8: torch.Tensor) -> torch.Tensor:
    return x8.float().div_(255.0)


def _affine(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """theta [B, 2, 3] maps output coordinates to input coordinates (normalised, centre origin).
    Pixels that fall outside the source image are black, as in AugMix."""
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _theta(a, b, c, d, e, f) -> torch.Tensor:
    return torch.stack([torch.stack([a, b, c], dim=1), torch.stack([d, e, f], dim=1)], dim=1)


# ----------------------------------------------------------------------------------------
# AugMix ops
# ----------------------------------------------------------------------------------------
def autocontrast(x, level):
    """Stretch every channel of every image to the full [0, 1] range (level unused)."""
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    span = hi - lo
    return torch.where(span > 1e-6, (x - lo) / span.clamp_min(1e-6), x)


def equalize(x, level):
    """Per-channel histogram equalisation (level unused)."""
    return _from_u8(TF.equalize(_to_u8(x)))


def posterize(x, level):
    """Keep the 4 - floor(4 * level) most significant bits (at least 1)."""
    bits = (4 - torch.floor(level * 4)).clamp(1, 8)
    shift = (8 - bits).to(torch.uint8).view(-1, 1, 1, 1)
    x8 = _to_u8(x)
    return _from_u8(torch.bitwise_left_shift(torch.bitwise_right_shift(x8, shift), shift))


def solarize(x, level):
    """Invert pixel values at or above 256 - floor(256 * level) (on the 0..255 scale)."""
    thr = (256 - torch.floor(level * 256)).view(-1, 1, 1, 1)
    x8 = _to_u8(x).float()
    return torch.where(x8 >= thr, 255.0 - x8, x8) / 255.0


def rotate(x, level):
    """Rotate by up to +-30 degrees."""
    ang = level * math.radians(30.0) * _sign(level)
    cos, sin, zero = torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)
    return _affine(x, _theta(cos, -sin, zero, sin, cos, zero))


def shear_x(x, level):
    """Horizontal shear by up to +-0.3."""
    s, one, zero = level * 0.3 * _sign(level), torch.ones_like(level), torch.zeros_like(level)
    return _affine(x, _theta(one, s, zero, zero, one, zero))


def shear_y(x, level):
    """Vertical shear by up to +-0.3."""
    s, one, zero = level * 0.3 * _sign(level), torch.ones_like(level), torch.zeros_like(level)
    return _affine(x, _theta(one, zero, zero, s, one, zero))


def translate_x(x, level):
    """Horizontal shift by up to +-1/3 of the image width."""
    t, one, zero = level * (2.0 / 3.0) * _sign(level), torch.ones_like(level), torch.zeros_like(level)
    return _affine(x, _theta(one, zero, t, zero, one, zero))


def translate_y(x, level):
    """Vertical shift by up to +-1/3 of the image height."""
    t, one, zero = level * (2.0 / 3.0) * _sign(level), torch.ones_like(level), torch.zeros_like(level)
    return _affine(x, _theta(one, zero, zero, zero, one, t))


AFFINE_KIND = {"rotate": 0, "shear_x": 1, "shear_y": 2, "translate_x": 3, "translate_y": 4}


def affine(x, level, kind):
    """rotate / shear_x / shear_y / translate_x / translate_y in one resampling call.

    kind: int tensor [B] with the AFFINE_KIND code of each row. Same result as calling the five ops
    separately; AugMix uses this so that a batch needs one grid_sample instead of five.
    """
    s = _sign(level)
    ang, sh, tr = level * math.radians(30.0) * s, level * 0.3 * s, level * (2.0 / 3.0) * s
    one, zero = torch.ones_like(level), torch.zeros_like(level)
    rot = kind == 0
    cos, sin = torch.where(rot, torch.cos(ang), one), torch.where(rot, torch.sin(ang), zero)
    return _affine(x, _theta(cos, torch.where(kind == 1, sh, -sin), torch.where(kind == 3, tr, zero),
                             torch.where(kind == 2, sh, sin), cos, torch.where(kind == 4, tr, zero)))


# ----------------------------------------------------------------------------------------
# PRIME-style primitives
# ----------------------------------------------------------------------------------------
def fourier_noise(x, level, max_amp: float = 0.5):
    """Add one random planar wave (a single Fourier basis) to every channel.

    Per sample and channel: spatial frequency with radius ~ U[1, H/2] cycles/image and random
    orientation, random phase, amplitude = level * max_amp. This is structured, band-limited
    noise; it is not i.i.d. pixel noise (gaussian / shot / impulse are test corruptions).
    """
    b, c, h, w = x.shape
    dev = x.device
    radius = torch.empty(b, c, 1, 1, device=dev).uniform_(1.0, min(h, w) / 2.0)
    angle = torch.empty(b, c, 1, 1, device=dev).uniform_(0.0, math.pi)
    fy, fx = torch.round(radius * torch.sin(angle)), torch.round(radius * torch.cos(angle))
    fx = torch.where((fy == 0) & (fx == 0), torch.ones_like(fx), fx)
    phase = torch.empty(b, c, 1, 1, device=dev).uniform_(0.0, 2 * math.pi)
    yy = torch.arange(h, device=dev, dtype=x.dtype).view(1, 1, h, 1) / h
    xx = torch.arange(w, device=dev, dtype=x.dtype).view(1, 1, 1, w) / w
    wave = torch.cos(2 * math.pi * (fy * yy + fx * xx) + phase)
    return (x + (level * max_amp).view(-1, 1, 1, 1) * wave).clamp_(0, 1)


def spectral_filter(x, level, max_sigma: float = 1.0, k: int = 3):
    """Convolve with a random k x k kernel: identity + sigma * zero-sum Gaussian noise,
    sigma = level * max_sigma, one kernel per sample shared by its channels.

    The perturbation sums to zero, so the DC gain stays 1: the op reshapes the spatial spectrum
    (sharpen / emboss / slight smoothing) without changing mean brightness.
    """
    b, c, h, w = x.shape
    noise = torch.randn(b, 1, k, k, device=x.device, dtype=x.dtype)
    noise = noise - noise.mean(dim=(2, 3), keepdim=True)
    kernel = noise * (level * max_sigma).view(-1, 1, 1, 1)
    kernel[:, :, k // 2, k // 2] += 1.0
    weight = kernel.repeat_interleave(c, dim=0)                       # [B*C, 1, k, k]
    xp = F.pad(x.reshape(1, b * c, h, w), [k // 2] * 4, mode="reflect")
    return F.conv2d(xp, weight, groups=b * c).reshape(b, c, h, w).clamp_(0, 1)


def spatial_warp(x, level, max_disp: float = 0.1, modes: int = 4):
    """Smooth random diffeomorphism-like warp.

    Displacement field = sum over i, j <= modes of c_ij sin(i pi u) sin(j pi v) with
    c_ij ~ N(0, 1 / (i^2 + j^2)), rescaled so that the largest displacement is
    level * max_disp of the image side. The field vanishes on the border.
    """
    b, c, h, w = x.shape
    dev, dt = x.device, x.dtype
    freqs = torch.arange(1, modes + 1, device=dev, dtype=dt)
    u = (torch.arange(h, device=dev, dtype=dt) + 0.5) / h
    v = (torch.arange(w, device=dev, dtype=dt) + 0.5) / w
    su = torch.sin(math.pi * freqs[:, None] * u[None, :])             # [M, H]
    sv = torch.sin(math.pi * freqs[:, None] * v[None, :])             # [M, W]
    std = 1.0 / torch.sqrt(freqs[:, None] ** 2 + freqs[None, :] ** 2)
    coef = torch.randn(b, 2, modes, modes, device=dev, dtype=dt) * std
    field = torch.einsum("bdij,ih,jw->bdhw", coef, su, sv)            # [B, 2, H, W] (x, y)
    peak = field.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    field = field / peak * (level * max_disp * 2.0).view(-1, 1, 1, 1)  # grid units span [-1, 1]
    eye = torch.eye(2, 3, device=dev, dtype=dt).unsqueeze(0).expand(b, 2, 3)
    grid = F.affine_grid(eye, [b, c, h, w], align_corners=False) + field.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def color_curve(x, level, scale: float = 0.5, n_freq: int = 4):
    """Random smooth tone curve per channel: x + sum_n b_n sin(n pi x), b_n ~ N(0, (level*scale/n)^2).

    The curve fixes 0 and 1, so black and white stay put while mid-tones move.
    """
    b, c, _, _ = x.shape
    n = torch.arange(1, n_freq + 1, device=x.device, dtype=x.dtype)
    coef = torch.randn(b, c, n_freq, device=x.device, dtype=x.dtype) * (level * scale).view(-1, 1, 1) / n
    out = x.clone()
    for i in range(n_freq):
        out += coef[:, :, i, None, None] * torch.sin(math.pi * n[i] * x)
    return out.clamp_(0, 1)


def amplitude_mix(x, level, max_mix: float = 2.0):
    """Amplitude-phase recombination (APR, Chen et al., ICCV 2021), RoadLens v3.

    Keep every image's Fourier phase and move its amplitude spectrum towards that of another
    image of the batch: A' = (1 - lam) A_x + lam A_other, lam = min(1, level * max_mix).
    Phase carries the shape (edges, symbols); amplitude carries contrast and texture energy, which
    is what noise, haze and contrast loss disturb. Training on images whose amplitude no longer
    matches their content pushes the network to rely on phase. It is not one of the test
    corruptions: no noise is added and nothing is blurred; the spectrum comes from a real sign.
    """
    b = x.shape[0]
    if b < 2:
        return x
    f = torch.fft.fft2(x.float())
    lam = (level * max_mix).clamp(0, 1).view(-1, 1, 1, 1)
    amp = (1 - lam) * f.abs() + lam * f.abs()[torch.randperm(b, device=x.device)]
    out = torch.fft.ifft2(torch.polar(amp, torch.angle(f))).real
    return out.clamp_(0, 1).to(x.dtype)


OPS = {
    "autocontrast": autocontrast, "equalize": equalize, "posterize": posterize, "solarize": solarize,
    "rotate": rotate, "shear_x": shear_x, "shear_y": shear_y,
    "translate_x": translate_x, "translate_y": translate_y,
    "fourier_noise": fourier_noise, "spectral_filter": spectral_filter,
    "spatial_warp": spatial_warp, "color_curve": color_curve,
    "amplitude_mix": amplitude_mix,
}

OP_SETS = {
    "augmix": ["autocontrast", "equalize", "posterize", "solarize", "rotate",
               "shear_x", "shear_y", "translate_x", "translate_y"],
    "prime": ["fourier_noise", "spectral_filter", "spatial_warp", "color_curve"],
    "apr": ["amplitude_mix"],                     # RoadLens v3
    # ops that move pixels; they are related to the affine / warp parts of elastic_transform
    "geometric": ["rotate", "shear_x", "shear_y", "translate_x", "translate_y", "spatial_warp"],
}


def resolve_ops(spec, exclude=None) -> list[str]:
    """Expand a list of op names / set names (or one name) into op names, minus `exclude`."""
    spec = [spec] if isinstance(spec, str) else list(spec)

    def expand(items):
        out = []
        for item in items:
            names = OP_SETS.get(item, [item])
            for name in names:
                if name not in OPS:
                    raise KeyError(f"unknown augmentation op or set: {name!r}")
                if name not in out:
                    out.append(name)
        return out

    drop = set(expand(exclude or []))
    ops = [name for name in expand(spec) if name not in drop]
    if not ops:
        raise ValueError("augmentation op set is empty")
    return ops
