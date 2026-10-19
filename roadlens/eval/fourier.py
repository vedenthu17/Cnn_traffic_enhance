"""Fourier heat map of model error (Yin et al., "A Fourier Perspective on Model Robustness", 2019).

For every 2-D Fourier basis U_{u,v} (a real planar wave with unit L2 norm) each test image is
perturbed to clip(x + r * eps * U_{u,v}), with an independent random sign r per image and channel,
and the error rate is recorded at position (u, v). Low frequencies sit in the centre of the map.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from tqdm import tqdm

from ..data.loading import as_float, u8_to_device


@torch.no_grad()
def fourier_heatmaps(models: dict, x: np.ndarray, y: np.ndarray, device, eps: float = 4.0, stride: int = 1,
                     batch_size: int = 1000, amp: bool = True, seed: int = 0):
    """Returns ({name: [H/stride, W/stride] error map}, {name: clean error}) for several models at once
    (all models see exactly the same perturbed images)."""
    X = as_float(u8_to_device(x, device))
    Y = torch.from_numpy(np.asarray(y)).to(device)
    n, c, h, w = X.shape
    use_amp = amp and device.type == "cuda"
    for m in models.values():
        m.eval()

    def errors(images):
        out = {}
        with torch.autocast(device.type, enabled=use_amp):
            for name, m in models.items():
                wrong = 0
                for s in range(0, n, batch_size):
                    wrong += int((m(images[s:s + batch_size]).argmax(dim=1) != Y[s:s + batch_size]).sum())
                out[name] = wrong / n
        return out

    clean = errors(X)
    us = list(range(-h // 2, h // 2, stride))
    vs = list(range(-w // 2, w // 2, stride))
    maps = {name: np.full((len(us), len(vs)), np.nan) for name in models}
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ii = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
    jj = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
    index = {(u % h, v % w): (a, b) for a, u in enumerate(us) for b, v in enumerate(vs)}
    todo = [(u, v) for u in us for v in vs]
    for u, v in tqdm(todo, desc="fourier heat map", leave=False):
        a, b = index[(u % h, v % w)]
        if not np.isnan(maps[next(iter(maps))][a, b]):
            continue                                   # filled by its mirror frequency
        basis = torch.cos(2 * math.pi * (u * ii / h + v * jj / w))
        basis = basis / basis.norm()
        sign = (torch.randint(0, 2, (n, c, 1, 1), generator=gen) * 2 - 1).to(device=device, dtype=torch.float32)
        err = errors((X + eps * sign * basis).clamp_(0, 1))
        mirror = index.get(((-u) % h, (-v) % w))       # U_{-u,-v} is the same real basis
        for name in models:
            maps[name][a, b] = err[name]
            if mirror is not None:
                maps[name][mirror] = err[name]
    return maps, clean
