"""Fusion: a mixture of expert predictions with per-image weights.

    p(y | x) = sum_e w_e(x) * softmax(z_e(x))

An expert is one branch evaluated on one input view (roadlens/models/views.py). With the default
single view 'raw' the experts are simply the branches; with views (raw, median3, gauss) every
branch also sees two low-pass copies of the image, and the experts are ordered view by view
(all branches on the first view, then all branches on the second, ...).

mode 'gate'       w = softmax(MLP(gate inputs))                          (learned)
mode 'confidence' w_e proportional to the expert's max softmax probability (fallback comparison)
mode 'average'    w_e = 1 / E                                             (plain ensemble)

Gate inputs (any combination):
    features  the branches' pooled features (large; needs a lot of data)
    probs     the experts' class probabilities
    stats     three confidence numbers per expert plus their agreement (views.expert_stats)
    image     five statistics of the input image (views.image_stats), so that the gate can tell
              a noisy or hazy image from a clean one

The output is log p, which is a valid logit vector (softmax(log p) = p), so losses, metrics,
calibration and TENT treat the fusion like any other classifier. Because the result is a convex
combination, the gate can only re-weight the experts; it cannot invent a class no expert supports.
A gate starts as the plain average (zero-initialised output layer).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Classifier
from .views import N_IMAGE_STATS, VIEWS, expert_stats, image_stats

GATE_INPUTS = ("features", "probs", "stats", "image")


class Fusion(Classifier):
    def __init__(self, branches: list[Classifier], num_classes: int = 43, mode: str = "gate",
                 gate_inputs=("features", "probs"), hidden: int = 256, dropout: float = 0.1,
                 freeze_branches: bool = True, views=("raw",)):
        super().__init__()
        if mode not in ("gate", "confidence", "average"):
            raise ValueError(f"fusion mode must be gate | confidence | average, got {mode!r}")
        self.views = tuple(views)
        unknown = [v for v in self.views if v not in VIEWS]
        if unknown or not self.views:
            raise ValueError(f"fusion views must be a non-empty subset of {sorted(VIEWS)}, got {list(views)}")
        self.branches = nn.ModuleList(branches)
        self.mode = mode
        self.gate_inputs = tuple(gate_inputs)
        self.freeze_branches = bool(freeze_branches) or mode != "gate"
        self.feature_dim = sum(b.feature_dim for b in branches)
        self.n_experts = len(branches) * len(self.views)
        if self.freeze_branches:
            for p in self.branches.parameters():
                p.requires_grad_(False)
        if mode == "gate":
            bad = [g for g in self.gate_inputs if g not in GATE_INPUTS]
            if bad or not self.gate_inputs:
                raise ValueError(f"gate_inputs must be a non-empty subset of {list(GATE_INPUTS)}, got {list(gate_inputs)}")
            in_dim = 0
            if "features" in self.gate_inputs:
                self.feat_norms = nn.ModuleList([nn.LayerNorm(b.feature_dim) for b in branches])
                in_dim += self.feature_dim * len(self.views)
            if "probs" in self.gate_inputs:
                in_dim += num_classes * self.n_experts
            if "stats" in self.gate_inputs:
                in_dim += 3 * self.n_experts + 1
            if "image" in self.gate_inputs:
                in_dim += N_IMAGE_STATS
            self.gate = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, self.n_experts))
            nn.init.zeros_(self.gate[-1].weight)        # start as the plain average ensemble
            nn.init.zeros_(self.gate[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_branches:                        # frozen experts keep their BN statistics / no dropout
            self.branches.eval()
        return self

    def branch_weights(self, feats, logits, x=None) -> torch.Tensor:
        """[B, E] mixture weights. `x` (the input batch) is needed only for the 'image' gate input."""
        probs = [F.softmax(z.float(), dim=1) for z in logits]
        if self.mode == "average":
            return torch.full((probs[0].shape[0], len(probs)), 1.0 / len(probs), device=probs[0].device)
        if self.mode == "confidence":
            conf = torch.stack([p.amax(dim=1) for p in probs], dim=1)
            return conf / conf.sum(dim=1, keepdim=True)
        parts = []
        if "features" in self.gate_inputs:
            k = len(self.branches)
            parts += [self.feat_norms[i % k](f.float()) for i, f in enumerate(feats)]
        if "probs" in self.gate_inputs:
            parts += probs
        if "stats" in self.gate_inputs:
            parts.append(expert_stats(probs))
        if "image" in self.gate_inputs:
            if x is None:
                raise ValueError("this gate uses image statistics: pass the input batch to branch_weights")
            parts.append(image_stats(x))
        return F.softmax(self.gate(torch.cat(parts, dim=1)), dim=1)

    def forward_detailed(self, x) -> dict:
        """logits: fused log p. weights: [B, E]. branch_logits: the E expert logits (view by view).
        features: the branches' pooled features on the first view."""
        outs = []
        for name in self.views:
            xv = VIEWS[name](x)
            outs += [b.features_logits(xv) for b in self.branches]
        feats, logits = [o[0] for o in outs], [o[1] for o in outs]
        w = self.branch_weights(feats, logits, x)
        logp = torch.stack([F.log_softmax(z.float(), dim=1) for z in logits], dim=1)      # [B, E, C]
        fused = torch.logsumexp(logp + torch.log(w.clamp_min(1e-8)).unsqueeze(-1), dim=1)
        return {"logits": fused, "weights": w, "branch_logits": logits, "features": torch.cat(
            [f.float() for f in feats[:len(self.branches)]], dim=1)}

    def features_logits(self, x):
        out = self.forward_detailed(x)
        return out["features"], out["logits"]

    def cam_targets(self):
        return [t for b in self.branches for t in b.cam_targets()]

    # Frozen branches are restored from their own runs, so a fusion checkpoint holds only the gate.
    def checkpoint_state(self) -> dict:
        state = self.state_dict()
        if self.freeze_branches:
            state = {k: v for k, v in state.items() if not k.startswith("branches.")}
        return state

    def load_checkpoint_state(self, state: dict) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad = [k for k in missing if not (self.freeze_branches and k.startswith("branches."))]
        if bad or unexpected:
            raise RuntimeError(f"fusion checkpoint mismatch: missing={bad} unexpected={list(unexpected)}")
