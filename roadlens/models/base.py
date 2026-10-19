"""Common interface of every RoadLens classifier.

All models take float images [B, 3, H, W] in [0, 1] at the working resolution (64 px).
Normalisation and any upsampling to the backbone's input size happen inside the model, so
augmentations, corruptions, Fourier perturbations and Grad-CAM all work in plain pixel space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Preprocess(nn.Module):
    """Normalise and (optionally) bilinearly resize to the backbone's input size.

    norm 'dataset'   subtract the dataset mean and divide by its std, per channel (the default).
    norm 'instance'  standardise every image with its own per-channel mean and std. The network then
                     cannot see a per-channel affine change of intensity (gain and offset), which is
                     what exposure, haze and a loss of contrast mostly are. Contrast normalisation of
                     this kind is classical for traffic signs (Sermanet & LeCun 2011, Ciresan et al. 2012).
    norm 'local'     local-global contrast normalisation (RoadLens v3). Instance normalisation removes
                     one gain and offset per image; haze that varies across the image (fog is a
                     low-frequency veil) survives it. Here every pixel is also standardised by the
                     mean and std of its Gaussian neighbourhood (sigma = LOCAL_SIGMA px at 64 px):

                         local(x) = (x - G*x) / max(sqrt(G*x^2 - (G*x)^2), LOCAL_FLOOR * std(x))

                     The floor is relative to the image's own std, so local(a*x + b) = local(x) still
                     holds exactly, while flat regions (a sign's interior) are not amplified into
                     noise. The output is a learnable per-channel blend of the two,
                     (1 - s) * instance(x) + s * local(x) with s = sigmoid(mix), starting at s = 0.5,
                     so training decides how much local detail to use; absolute colour, which local
                     normalisation discards, stays available through the instance part.
    """

    EPS = 1e-4
    LOCAL_SIGMA = 4.0
    LOCAL_FLOOR = 0.5
    LOCAL_CLIP = 6.0

    def __init__(self, input_size: int | None = None, mean=IMAGENET_MEAN, std=IMAGENET_STD, norm: str = "dataset"):
        super().__init__()
        if norm not in ("dataset", "instance", "local"):
            raise ValueError(f"input_norm must be dataset | instance | local, got {norm!r}")
        self.input_size, self.norm = input_size, norm
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        if norm == "local":
            self.mix = nn.Parameter(torch.zeros(1, 3, 1, 1))

    @classmethod
    def local_norm(cls, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel standardisation by the Gaussian-weighted neighbourhood (see the class docstring)."""
        from .views import gauss

        x = x.float()
        mu = gauss(x, cls.LOCAL_SIGMA)
        sd = (gauss(x * x, cls.LOCAL_SIGMA) - mu * mu).clamp_min(0).sqrt()
        floor = cls.LOCAL_FLOOR * x.std(dim=(2, 3), keepdim=True, correction=0) + cls.EPS
        return ((x - mu) / torch.maximum(sd, floor)).clamp(-cls.LOCAL_CLIP, cls.LOCAL_CLIP)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm in ("instance", "local"):
            x = x.float()
            inst = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True, correction=0) + self.EPS)
            if self.norm == "local":
                s = torch.sigmoid(self.mix)
                x = (1 - s) * inst + s * self.local_norm(x)
            else:
                x = inst
        else:
            x = (x - self.mean) / self.std
        if self.input_size and x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return x


@dataclass
class CamTarget:
    """A layer whose activations Grad-CAM can use."""
    name: str
    module: nn.Module
    kind: str                 # "conv": [B, C, h, w]   "tokens": [B, prefix + h*w, C]
    prefix_tokens: int = 0


class Classifier(nn.Module):
    feature_dim: int = 0

    def features_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pooled features [B, D], logits [B, K])."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features_logits(x)[1]

    def cam_targets(self) -> list[CamTarget]:
        return []

    # Checkpoints store only what a run actually trained (see Fusion).
    def checkpoint_state(self) -> dict:
        return self.state_dict()

    def load_checkpoint_state(self, state: dict) -> None:
        self.load_state_dict(state)
