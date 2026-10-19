"""Grad-CAM (Selvaraju et al. 2017) for CNN feature maps and for ViT patch tokens."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..models.base import CamTarget


def grad_cam(model, targets: list[CamTarget], x: torch.Tensor, class_idx: torch.Tensor | None = None):
    """Class-activation maps of `model` at each target layer.

    x: float images [B, 3, H, W] in [0, 1]. class_idx: classes to explain (default: the predictions).
    Returns (cams: list of [B, H, W] tensors in [0, 1], one per target; logits [B, K]).
    """
    model.eval()
    acts = {}
    hooks = [t.module.register_forward_hook(lambda m, i, o, k=k: acts.__setitem__(k, o))
             for k, t in enumerate(targets)]
    try:
        with torch.enable_grad():
            # the input must require grad: frozen models (the fusion's branches) build no graph otherwise
            logits = model(x.detach().clone().requires_grad_(True)).float()
            cls = logits.argmax(dim=1) if class_idx is None else class_idx
            score = logits.gather(1, cls.view(-1, 1)).sum()
            grads = torch.autograd.grad(score, [acts[k] for k in range(len(targets))], allow_unused=True)
    finally:
        for hook in hooks:
            hook.remove()

    cams = []
    for k, t in enumerate(targets):
        a = acts[k].detach().float()
        g = torch.zeros_like(a) if grads[k] is None else grads[k].detach().float()
        if t.kind == "tokens":                        # [B, prefix + h*w, C] -> [B, C, h, w]
            a, g = a[:, t.prefix_tokens:], g[:, t.prefix_tokens:]
            side = int(math.isqrt(a.shape[1]))
            a = a.transpose(1, 2).reshape(a.shape[0], -1, side, side)
            g = g.transpose(1, 2).reshape(g.shape[0], -1, side, side)
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        cams.append(cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8))
    return cams, logits.detach()
