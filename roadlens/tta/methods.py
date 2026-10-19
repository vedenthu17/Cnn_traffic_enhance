"""Test-time adaptation on an unlabeled test stream.

  NoAdapt  the source model, unchanged.
  BNAdapt  plain test-time BatchNorm: every batch is normalised with its own statistics instead of
           the training-set running statistics (Nado et al. 2020).
  Tent     BNAdapt + one entropy-minimisation step per batch on the BatchNorm scale/shift
           parameters only (Wang et al. 2021).

Both adaptive methods take `prior` (default 0): with prior = N > 0 the batch statistics are mixed
with the source statistics, weighted n : N for a batch of n images (Schneider et al. 2020). That
keeps the methods usable at batch sizes where the statistics of one batch alone are too noisy.

Both adaptive methods touch BatchNorm layers only, i.e. the CNN parts of a model: all of SmallCNN
and ResNet-18, and the ResNet branch of the fusion. The DeiT branch has no BatchNorm and stays
fixed. For the fusion, Tent minimises the entropy of the CNN branch's own prediction by default
(`objective="branch"`); `objective="fused"` uses the entropy of the fused prediction instead.

The adaptive methods work on a private copy of the model, so the caller's model is never modified.
`predict` returns the (fused) logits computed *before* the update of that batch, as in the Tent
reference code, and `reset` restores the source state (call it between test distributions).
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def batchnorm_layers(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


class PriorBatchNorm(nn.Module):
    """A BatchNorm layer that normalises with a mixture of its source statistics and the batch's:

        statistic = (N * source + n * batch) / (N + n)

    with N = `prior` pseudo-samples and n = the batch size (Schneider et al. 2020). A batch of one
    image then moves the statistics by 1 / (N + 1) instead of replacing them, which is what makes
    plain test-time BatchNorm collapse at small batch sizes. The wrapped layer keeps its weight,
    bias and stored statistics.
    """

    def __init__(self, bn: nn.modules.batchnorm._BatchNorm, prior: float):
        super().__init__()
        self.bn, self.prior = bn, float(prior)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Written out instead of calling F.batch_norm: the mixed statistics depend on the batch, and
        # Tent has to back-propagate through them (as it does through plain batch statistics).
        bn = self.bn
        dims = [d for d in range(x.dim()) if d != 1]
        shape = [1, -1] + [1] * (x.dim() - 2)
        xf = x.float()
        a = x.shape[0] / (x.shape[0] + self.prior)
        mean = a * xf.mean(dim=dims) + (1 - a) * bn.running_mean
        var = a * xf.var(dim=dims, correction=0) + (1 - a) * bn.running_var
        out = (xf - mean.view(shape)) / torch.sqrt(var.view(shape) + bn.eps)
        if bn.weight is not None:
            out = out * bn.weight.view(shape) + bn.bias.view(shape)
        return out.to(x.dtype)


def _use_batch_statistics(model: nn.Module, prior: float = 0.0) -> list:
    """Make the BatchNorm layers normalise with test-batch statistics; returns those layers.

    prior = 0: the layers go to training mode with momentum 0, so they use the statistics of the
    current batch alone and leave the stored source statistics untouched.
    prior > 0: every layer is wrapped in PriorBatchNorm (source statistics as a prior of that weight).
    """
    layers = batchnorm_layers(model)
    if prior > 0:
        plain = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        odd = sorted({type(m).__name__ for m in layers if type(m) not in plain})
        if odd:
            raise NotImplementedError(f"a statistics prior is implemented for plain BatchNorm layers only, found {odd}")
        for parent in list(model.modules()):
            for name, child in list(parent.named_children()):
                if type(child) in plain:
                    setattr(parent, name, PriorBatchNorm(child, prior))
        return layers
    for m in layers:
        m.train()
        m.momentum = 0.0
    return layers


def entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=1)
    return -(logp.exp() * logp).sum(dim=1)


class NoAdapt:
    name = "none"

    def __init__(self, model: nn.Module, amp: bool = True):
        self.model = model.eval()
        self.amp = amp

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, enabled=self.amp and x.device.type == "cuda"):
            return self.model(x).float()


class BNAdapt(NoAdapt):
    name = "bn"

    def __init__(self, model: nn.Module, amp: bool = True, prior: float = 0.0):
        super().__init__(deepcopy(model), amp)
        if not _use_batch_statistics(self.model, prior):
            raise ValueError("model has no BatchNorm layers: test-time BN adaptation does not apply")


class Tent:
    name = "tent"

    def __init__(self, model: nn.Module, lr: float = 1e-3, steps: int = 1, amp: bool = True,
                 objective: str = "branch", prior: float = 0.0):
        if objective not in ("branch", "fused"):
            raise ValueError(f"tent objective must be branch | fused, got {objective!r}")
        self.model = deepcopy(model).eval()
        self.amp, self.steps = amp, int(steps)
        for p in self.model.parameters():
            p.requires_grad_(False)
        layers = _use_batch_statistics(self.model, prior)
        if not layers:
            raise ValueError("model has no BatchNorm layers: Tent does not apply")
        # For a fusion model, 'branch' minimises the entropy of each CNN branch's own prediction
        # (plain Tent on that branch); 'fused' minimises the entropy of the fused prediction, whose
        # gradient vanishes whenever the gate gives the CNN branch no weight.
        self.branch_ids = None
        if objective == "branch" and hasattr(self.model, "branches"):
            self.branch_ids = [k for k, b in enumerate(self.model.branches) if batchnorm_layers(b)]
        self.params = [p for m in layers for p in (m.weight, m.bias) if p is not None]
        for p in self.params:
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.params, lr=lr, betas=(0.9, 0.999))
        self._source = [p.detach().clone() for p in self.params]
        self._opt_state = deepcopy(self.opt.state_dict())
        self._scaler = None

    def reset(self) -> None:
        with torch.no_grad():
            for p, src in zip(self.params, self._source):
                p.copy_(src)
        self.opt.load_state_dict(deepcopy(self._opt_state))
        self._scaler = None

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        use_amp = self.amp and x.device.type == "cuda"
        if self._scaler is None:
            self._scaler = torch.amp.GradScaler(x.device.type, enabled=use_amp)
        first = None
        for _ in range(self.steps):
            with torch.enable_grad():
                with torch.autocast(x.device.type, enabled=use_amp):
                    if self.branch_ids is None:
                        logits = self.model(x)
                        loss = entropy(logits).mean()
                    else:
                        out = self.model.forward_detailed(x)
                        logits = out["logits"]
                        loss = sum(entropy(out["branch_logits"][k]).mean() for k in self.branch_ids) / len(self.branch_ids)
                self.opt.zero_grad(set_to_none=True)
                self._scaler.scale(loss).backward()
                self._scaler.step(self.opt)
                self._scaler.update()
            if first is None:
                first = logits.detach().float()
        return first


def build_tta(method: str, model: nn.Module, amp: bool = True, lr: float = 1e-3, steps: int = 1,
              objective: str = "branch", prior: float = 0.0):
    """method: none | bn | tent. prior: pseudo-sample weight of the source BatchNorm statistics (0 = batch only)."""
    if method == "none":
        return NoAdapt(model, amp)
    if method == "bn":
        return BNAdapt(model, amp, prior)
    if method == "tent":
        return Tent(model, lr=lr, steps=steps, amp=amp, objective=objective, prior=prior)
    raise ValueError(f"unknown test-time adaptation method: {method!r}")
