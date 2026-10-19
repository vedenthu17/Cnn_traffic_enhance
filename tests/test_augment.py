"""Batched augmentation ops against PIL / torchvision references, and AugMix invariants."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms import functional as TVF
from torchvision.transforms import InterpolationMode

from roadlens.augment import AugMix, ColorJitter, TrainAugment, jsd_loss, resolve_ops
from roadlens.augment import ops as O


def batch(n=6, size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(n, 3, 8, 8, generator=g) * 0.8 + 0.1          # keep away from 0/1 so ranges are not full
    x = F.interpolate(low, size=size, mode="bicubic", align_corners=False).clamp(0.05, 0.9)
    return (x * 255).round() / 255


def to_pil(x):
    return [Image.fromarray((img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) for img in x]


def from_pil(imgs):
    return torch.stack([torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255 for im in imgs])


def max_diff_255(a, b):
    return float((a - b).abs().max() * 255)


def test_autocontrast_matches_pil():
    x = batch()
    ref = from_pil([ImageOps.autocontrast(im) for im in to_pil(x)])
    assert max_diff_255(O.autocontrast(x, torch.zeros(len(x))), ref) <= 1.01


def test_equalize_matches_pil():
    x = batch()
    ref = from_pil([ImageOps.equalize(im) for im in to_pil(x)])
    assert max_diff_255(O.equalize(x, torch.zeros(len(x))), ref) <= 1.01


def test_posterize_matches_pil():
    x = batch(4)
    for level, bits in [(0.0, 4), (0.3, 3), (0.5, 2), (0.76, 1), (1.0, 1)]:
        ref = from_pil([ImageOps.posterize(im, bits) for im in to_pil(x)])
        assert max_diff_255(O.posterize(x, torch.full((4,), level)), ref) == 0, (level, bits)


def test_solarize_matches_pil():
    x = batch(4)
    for level in (0.0, 0.25, 0.5, 1.0):
        thr = 256 - int(level * 256)
        ref = from_pil([ImageOps.solarize(im, thr) for im in to_pil(x)])
        assert max_diff_255(O.solarize(x, torch.full((4,), level)), ref) < 1e-3, level


def test_ops_are_identity_at_level_zero():
    x = batch()
    zero = torch.zeros(len(x))
    for name in ["solarize", "rotate", "shear_x", "shear_y", "translate_x", "translate_y",
                 "fourier_noise", "spectral_filter", "spatial_warp", "color_curve"]:
        assert max_diff_255(O.OPS[name](x, zero), x) < 0.01, name


def test_rotate_matches_torchvision():
    x = batch(2)
    torch.manual_seed(0)
    out = O.rotate(x, torch.ones(2))                                  # +-30 degrees
    refs = [TVF.rotate(x, a, interpolation=InterpolationMode.BILINEAR) for a in (30.0, -30.0)]
    for i in range(2):
        assert min(max_diff_255(out[i], r[i]) for r in refs) < 1.0


def test_translate_moves_by_a_third():
    x = torch.zeros(1, 3, 60, 60)
    x[:, :, 30, 30] = 1.0
    torch.manual_seed(0)
    out = O.translate_x(x, torch.ones(1))
    col = int(out[0, 0, 30].argmax())
    assert abs(abs(col - 30) - 20) <= 1 and float(out[0, 0, 30].max()) > 0.9


def test_all_ops_keep_shape_range_and_batch_independence():
    x = batch(8)
    for name, op in O.OPS.items():
        torch.manual_seed(1)
        out = op(x, torch.full((8,), 0.3))
        assert out.shape == x.shape and out.dtype == x.dtype, name
        assert float(out.min()) >= 0 and float(out.max()) <= 1, name
        assert torch.isfinite(out).all(), name


def test_spectral_filter_keeps_flat_images_flat():
    x = torch.full((4, 3, 32, 32), 0.4)
    torch.manual_seed(0)
    assert max_diff_255(O.spectral_filter(x, torch.ones(4)), x) < 0.01


def test_spatial_warp_displacement_is_bounded():
    # warp a horizontal ramp: the value change equals the horizontal displacement (as a fraction of width)
    w = 64
    ramp = ((torch.arange(w) + 0.5) / w).view(1, 1, 1, w).expand(4, 3, w, w).contiguous()
    torch.manual_seed(0)
    out = O.spatial_warp(ramp, torch.ones(4), max_disp=0.1)
    shift = (out - ramp).abs().amax(dim=(1, 2, 3))
    assert float(shift.max()) <= 0.1 + 1e-3 and float(shift.min()) > 0.01


def test_color_curve_fixes_black_and_white():
    x = torch.zeros(2, 3, 8, 8)
    x[:, :, 4:] = 1.0
    torch.manual_seed(0)
    assert max_diff_255(O.color_curve(x, torch.ones(2)), x) < 0.01


def test_fourier_noise_amplitude():
    x = torch.full((16, 3, 64, 64), 0.5)
    torch.manual_seed(0)
    out = O.fourier_noise(x, torch.full((16,), 0.4), max_amp=0.5)
    amp = (out - x).abs().amax(dim=(2, 3))
    assert float(amp.max()) <= 0.2 + 1e-4 and float(amp.min()) > 0.19


def test_augmix_is_deterministic_bounded_and_changes_images():
    x = batch(16)
    aug = AugMix(resolve_ops(["augmix", "prime"]), width=3, depth=(1, 3), alpha=1.0, severity=3)
    torch.manual_seed(7)
    a, a2 = aug(x, 2)
    torch.manual_seed(7)
    b, b2 = aug(x, 2)
    assert torch.equal(a, b) and torch.equal(a2, b2) and not torch.equal(a, a2)
    assert a.shape == x.shape and float(a.min()) >= 0 and float(a.max()) <= 1
    per_image = (a - x).abs().flatten(1).mean(1)
    assert (per_image > 1e-4).float().mean() > 0.8


def test_merged_affine_equals_the_separate_ops():
    x = batch(6)
    level = torch.linspace(0.1, 1.0, 6)
    for name, kind in O.AFFINE_KIND.items():
        torch.manual_seed(3)
        separate = O.OPS[name](x, level)
        torch.manual_seed(3)
        merged = O.affine(x, level, torch.full((6,), kind))
        assert torch.allclose(separate, merged, atol=1e-6), name


def test_augmix_applies_each_op_only_to_its_own_rows():
    # one marker op per group type: rows must come back in their original order after the per-op sorting
    from roadlens.augment import ops
    ops.OPS["_plus"] = lambda x, level: (x + 0.25).clamp(0, 1)
    ops.OPS["_zero"] = lambda x, level: torch.zeros_like(x)
    try:
        x = torch.rand(32, 3, 8, 8) * 0.5
        torch.manual_seed(0)
        out = AugMix(["_plus", "_zero"], width=1, depth=(1, 1), alpha=1.0)(x)[0]
        # width 1, depth 1: out = (1 - m) x + m op(x) with op in {x + .25, 0} -> per image either brighter or darker
        delta = (out - x).flatten(1)
        brighter, darker = (delta >= -1e-6).all(1), (delta <= 1e-6).all(1)
        assert bool((brighter | darker).all()) and bool(brighter.any()) and bool(darker.any())
        ratio = delta[darker] / x.flatten(1)[darker]                 # = -m, constant within an image
        assert float(ratio.std(dim=1).max()) < 1e-4
    finally:
        del ops.OPS["_plus"], ops.OPS["_zero"]


def test_augmix_convexity_with_identity_ops():
    # with ops that do nothing, any Dirichlet/Beta mixing must return the input
    from roadlens.augment import ops
    ops.OPS["_identity"] = lambda x, level: x
    try:
        x = batch(8)
        out = AugMix(["_identity"], width=3, depth=(1, 3))(x)[0]
        assert max_diff_255(out, x) < 1e-2
    finally:
        del ops.OPS["_identity"]


def test_resolve_ops():
    assert len(resolve_ops("augmix")) == 9
    assert len(resolve_ops(["augmix", "prime"])) == 13
    no_geom = resolve_ops(["augmix", "prime"], exclude=["geometric"])
    assert "rotate" not in no_geom and "spatial_warp" not in no_geom and "fourier_noise" in no_geom
    assert resolve_ops(["rotate", "rotate", "augmix"]).count("rotate") == 1


def test_train_augment_views():
    x = batch(8)
    cj = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02}
    assert len(TrainAugment({"mode": "none", "color_jitter": cj})(x)) == 1
    assert len(TrainAugment({"mode": "mix", "ops": "augmix", "color_jitter": cj})(x)) == 1
    views = TrainAugment({"mode": "jsd", "ops": ["augmix", "prime"], "color_jitter": cj})(x)
    assert len(views) == 3 and not torch.equal(views[1], views[2])


def test_color_jitter_is_mild_and_identity_when_disabled():
    x = batch(8)
    torch.manual_seed(0)
    out = ColorJitter(0.2, 0.2, 0.2, 0.02)(x)
    assert float((out - x).abs().max()) < 0.5 and float((out - x).abs().mean()) > 1e-3
    assert torch.equal(ColorJitter(0, 0, 0, 0)(x), x)


def test_jsd_loss():
    z = torch.randn(10, 43)
    assert float(jsd_loss([z, z, z])) < 1e-6
    assert float(jsd_loss([z, torch.randn(10, 43), torch.randn(10, 43)])) > 0.05
