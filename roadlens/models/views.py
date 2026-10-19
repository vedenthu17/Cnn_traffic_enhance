"""Deterministic input views and per-image statistics for view-expert fusion.

A view is a fixed, parameter-free transform of the input image. The fusion evaluates every branch on
every view and mixes the resulting "experts". The two low-pass views trade pixel noise for a little
blur: the models here are measurably robust to blur and fragile to noise, so a smoothed copy of a
noisy image is an easier input than the original.

Nothing here is learned and no view is a training augmentation. Which views a model uses is a
hyper-parameter, to be chosen on validation-C like any other.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def raw(x: torch.Tensor) -> torch.Tensor:
    return x


def median3(x: torch.Tensor) -> torch.Tensor:
    """3 x 3 median filter per channel (replicated border). Removes isolated outliers, keeps edges."""
    b, c, h, w = x.shape
    patches = F.unfold(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel_size=3)       # [B, C * 9, H * W]
    return patches.view(b, c, 9, h * w).median(dim=2).values.view(b, c, h, w)


def gauss(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Separable Gaussian blur with standard deviation `sigma` pixels (reflected border)."""
    radius = max(1, int(math.ceil(2.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


VIEWS = {"raw": raw, "median3": median3, "gauss": gauss}

N_IMAGE_STATS = 5


def image_stats(x: torch.Tensor) -> torch.Tensor:
    """[B, 5] cheap descriptors of how degraded an image looks, all of order 1:

    mean luminance, luminance contrast (std x 4), high-frequency energy (mean |L - blur(L)| x 10),
    the same energy relative to the contrast, and mean chroma (max - min over channels, x 2).
    Pixel noise raises the high-frequency terms, blur lowers them, haze lowers the contrast.
    """
    x = x.float()
    lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    mean = lum.mean(dim=(1, 2, 3))
    std = lum.std(dim=(1, 2, 3), correction=0)
    smooth = F.avg_pool2d(F.pad(lum, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    hf = (lum - smooth).abs().mean(dim=(1, 2, 3))
    chroma = (x.amax(dim=1) - x.amin(dim=1)).mean(dim=(1, 2))
    return torch.stack([mean, 4.0 * std, 10.0 * hf, hf / (std + 0.02), 2.0 * chroma], dim=1)


def expert_stats(probs: list[torch.Tensor]) -> torch.Tensor:
    """[B, 3 E + 1] confidence descriptors of E expert distributions, all in [0, 1]:

    per expert its top probability, top-1 minus top-2 margin and normalised entropy, then the share
    of expert pairs that predict the same class.
    """
    out = []
    for p in probs:
        top2 = p.topk(2, dim=1).values
        ent = -(p * p.clamp_min(1e-8).log()).sum(dim=1) / math.log(p.shape[1])
        out += [top2[:, 0], top2[:, 0] - top2[:, 1], ent]
    pred = torch.stack([p.argmax(dim=1) for p in probs], dim=1)                       # [B, E]
    out.append((pred[:, :, None] == pred[:, None, :]).float().mean(dim=(1, 2)))
    return torch.stack(out, dim=1)
