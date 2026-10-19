"""AugMix-style mixing, colour jitter and the JSD consistency loss (all batched, on the GPU)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .ops import AFFINE_KIND, OPS, affine, resolve_ops


class AugMix:
    """x_aug = (1 - m) * x + m * sum_k w_k * chain_k(x)

    with `width` chains of `depth` randomly chosen ops each, w ~ Dirichlet(alpha, ..., alpha) and
    m ~ Beta(alpha, alpha), all sampled per image. Op magnitudes are U(0.1, severity) / 10 as in
    the AugMix reference implementation.
    """

    def __init__(self, ops, width: int = 3, depth=(1, 3), alpha: float = 1.0, severity: float = 3.0):
        self.names = list(ops)
        self.width = int(width)
        self.depth = (int(depth), int(depth)) if isinstance(depth, (int, float)) else (int(depth[0]), int(depth[1]))
        self.alpha = float(alpha)
        self.severity = float(severity)
        # Ops are applied group by group. Every op is its own group, except the affine ones, which
        # share a single resampling call (GPU time here is dominated by the number of calls).
        self.groups, group_of, kind_of = [], [], []
        affine_group = None
        for name in self.names:
            kind_of.append(AFFINE_KIND.get(name, -1))
            if name in AFFINE_KIND:
                if affine_group is None:
                    affine_group = len(self.groups)
                    self.groups.append(None)                         # filled in below
                group_of.append(affine_group)
            else:
                group_of.append(len(self.groups))
                self.groups.append(lambda x, level, idx, op=OPS[name]: op(x, level))
        if affine_group is not None:
            self.groups[affine_group] = lambda x, level, idx: affine(x, level, self._kind_of[idx])
        self._group_of, self._kind_of = torch.tensor(group_of), torch.tensor(kind_of)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, n_views: int = 1) -> list[torch.Tensor]:
        """Return `n_views` independently augmented versions of the batch (computed in one pass)."""
        b, dev = x.shape[0], x.device
        if self._group_of.device != dev:
            self._group_of, self._kind_of = self._group_of.to(dev), self._kind_of.to(dev)
        rows = n_views * self.width
        n, n_groups = rows * b, len(self.groups)
        chains = x.repeat(rows, 1, 1, 1)                             # [views * width * B, C, H, W]
        depth = torch.randint(self.depth[0], self.depth[1] + 1, (n,), device=dev)
        for step in range(self.depth[1]):
            op_idx = torch.randint(len(self.names), (n,), device=dev)
            level = torch.empty(n, device=dev).uniform_(0.1, self.severity) / 10.0
            # sort rows by group so that each group is one contiguous slice; finished chains go last
            key = torch.where(depth > step, self._group_of[op_idx], n_groups)
            order = torch.argsort(key)
            counts = torch.bincount(key, minlength=n_groups + 1).tolist()
            xs, lv, oi = chains[order], level[order], op_idx[order]
            start = 0
            for g in range(n_groups):
                end = start + counts[g]
                if end > start:
                    xs[start:end] = self.groups[g](xs[start:end], lv[start:end], oi[start:end]).to(xs.dtype)
                start = end
            chains[order] = xs
        conc = torch.full((n_views, b, self.width), self.alpha, device=dev)
        w = torch.distributions.Dirichlet(conc).sample()             # [views, B, width]
        a = torch.full((n_views, b), self.alpha, device=dev)
        m = torch.distributions.Beta(a, a).sample().view(n_views, b, 1, 1, 1)
        chains = chains.view(n_views, self.width, *x.shape)
        mixed = (chains * w.permute(0, 2, 1).reshape(n_views, self.width, b, 1, 1, 1)).sum(1)
        out = ((1 - m) * x + m * mixed).clamp_(0, 1)
        return list(out.unbind(0))


class ColorJitter:
    """Mild per-image brightness / contrast / saturation / hue jitter (the only baseline augmentation).

    No horizontal flip anywhere: it would turn 'keep left' into 'keep right'.
    """

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.02):
        self.b, self.c, self.s, self.h = float(brightness), float(contrast), float(saturation), float(hue)

    @staticmethod
    def _factor(n, amount, dev):
        return torch.empty(n, 1, 1, 1, device=dev).uniform_(1 - amount, 1 + amount)

    @staticmethod
    def _gray(x):
        return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n, dev = x.shape[0], x.device
        if self.b > 0:
            x = (x * self._factor(n, self.b, dev)).clamp_(0, 1)
        if self.c > 0:
            mean = self._gray(x).mean(dim=(2, 3), keepdim=True)
            x = ((x - mean) * self._factor(n, self.c, dev) + mean).clamp_(0, 1)
        if self.s > 0:
            gray = self._gray(x)
            x = ((x - gray) * self._factor(n, self.s, dev) + gray).clamp_(0, 1)
        if self.h > 0:
            # hue rotation = rotation of the chroma plane in YIQ space
            ang = torch.empty(n, device=dev).uniform_(-self.h, self.h) * 2 * math.pi
            cos, sin = torch.cos(ang), torch.sin(ang)
            to_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]],
                                  device=dev, dtype=x.dtype)
            rot = torch.zeros(n, 3, 3, device=dev, dtype=x.dtype)
            rot[:, 0, 0] = 1
            rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = cos, -sin, sin, cos
            m = torch.linalg.inv(to_yiq) @ rot @ to_yiq                # [B, 3, 3]
            x = torch.einsum("bij,bjhw->bihw", m, x).clamp_(0, 1)
        return x


class TrainAugment:
    """Turns a clean batch into the views the loss needs.

    mode 'none': [jitter(x)]                          standard training
    mode 'mix' : [augmix(jitter(x))]                  AugMix without the consistency loss
    mode 'jsd' : [jitter(x), augmix(.), augmix(.)]    clean + two mixed views for the JSD loss
    """

    def __init__(self, cfg: dict):
        self.mode = cfg.get("mode", "none")
        if self.mode not in ("none", "mix", "jsd"):
            raise ValueError(f"augment.mode must be none | mix | jsd, got {self.mode!r}")
        cj = cfg.get("color_jitter") or {}
        self.jitter = ColorJitter(**cj) if any(v > 0 for v in cj.values()) else None
        self.augmix = None
        if self.mode != "none":
            self.op_names = resolve_ops(cfg["ops"], cfg.get("exclude_ops"))
            self.augmix = AugMix(self.op_names, cfg.get("width", 3), cfg.get("depth", (1, 3)),
                                 cfg.get("alpha", 1.0), cfg.get("severity", 3.0))

    @property
    def n_views(self) -> int:
        return 3 if self.mode == "jsd" else 1

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.jitter is not None:
            x = self.jitter(x)
        if self.mode == "none":
            return [x]
        if self.mode == "mix":
            return self.augmix(x, 1)
        return [x] + self.augmix(x, 2)


def jsd_loss(logits_views: list[torch.Tensor]) -> torch.Tensor:
    """Jensen-Shannon divergence between the predictive distributions of the views."""
    probs = [F.softmax(z.float(), dim=1) for z in logits_views]
    log_mix = torch.clamp(torch.stack(probs).mean(0), 1e-7, 1.0).log()
    return sum(F.kl_div(log_mix, p, reduction="batchmean") for p in probs) / len(probs)
