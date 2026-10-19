"""The v3 components: local-global contrast normalisation, amplitude-phase recombination, EMA weights,
distillation, SimCLR and the compatibility of older runs (random weights / synthetic data, CPU)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import roadlens.engine as engine
from roadlens.augment import TrainAugment, resolve_ops
from roadlens.augment.ops import amplitude_mix
from roadlens.models import SmallCNN, build_model
from roadlens.models.base import Preprocess
from roadlens.runs import _classifier_prefixes, fingerprint, load_run_config
from roadlens.ssl import knn_accuracy, nt_xent


def images(n=8, seed=0, size=32):
    return torch.rand(n, 3, size, size, generator=torch.Generator().manual_seed(seed))


def small(seed, classes=3, **kw):
    torch.manual_seed(seed)
    return SmallCNN(classes, width=8, **kw)


# ---------------------------------------------------------------------------- local-global normalisation
def test_local_norm_is_invariant_to_gain_and_offset_and_safe_on_flat_images():
    x = images(4) * 0.6 + 0.2
    gain = torch.tensor([0.1, 0.5, 2.0]).view(1, 3, 1, 1)
    offset = torch.tensor([0.3, -0.2, 0.05]).view(1, 3, 1, 1)
    a, b = Preprocess.local_norm(x), Preprocess.local_norm(x * gain + offset)
    assert torch.allclose(a, b, atol=1e-3)
    flat = Preprocess.local_norm(torch.full((2, 3, 32, 32), 0.5))
    assert torch.isfinite(flat).all() and flat.abs().max() < 1e-3


def test_local_norm_removes_a_smooth_additive_veil_better_than_instance_norm():
    x = images(4, size=64) * 0.6 + 0.2
    yy = torch.linspace(0, 1, 64).view(1, 1, 64, 1)
    veil = 0.6 * yy.expand(4, 3, 64, 64)                      # haze that grows from top to bottom
    foggy = 0.5 * x + veil
    inst = lambda z: (z - z.mean((2, 3), keepdim=True)) / z.std((2, 3), keepdim=True, correction=0)  # noqa: E731
    err_local = (Preprocess.local_norm(foggy) - Preprocess.local_norm(x)).abs().mean()
    err_inst = (inst(foggy) - inst(x)).abs().mean()
    assert err_local < 0.5 * err_inst


def test_local_stem_is_a_learnable_blend_and_reaches_timm_models():
    pre = Preprocess(None, norm="local")
    assert torch.allclose(torch.sigmoid(pre.mix), torch.full_like(pre.mix, 0.5))
    out = pre(images(2))
    out.sum().backward()
    assert pre.mix.grad is not None and torch.isfinite(pre.mix.grad).all()
    m = build_model({"name": "resnet18", "pretrained": False, "input_size": 64, "input_norm": "local"}, 43).eval()
    assert m.pre.norm == "local" and "pre.mix" in m.state_dict()
    with pytest.raises(ValueError):
        Preprocess(None, norm="global")


# ---------------------------------------------------------------------------- amplitude-phase recombination
def test_amplitude_mix_keeps_shape_range_and_is_identity_at_level_zero():
    x = images(6)
    same = amplitude_mix(x, torch.zeros(6))
    assert torch.allclose(same, x, atol=1e-4)
    out = amplitude_mix(x, torch.full((6,), 0.3))
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    assert (out - x).abs().mean() > 1e-3
    assert torch.equal(amplitude_mix(x[:1], torch.full((1,), 0.3)), x[:1])          # nothing to mix with


def test_amplitude_mix_keeps_the_fourier_phase():
    x = images(4) * 0.2 + 0.4                                 # mid-range, so the clamp to [0, 1] stays inactive
    out = amplitude_mix(x, torch.full((4,), 0.2))
    fx, fo = torch.fft.fft2(x), torch.fft.fft2(out)
    strong = (fx.abs() > 1e-2) & (fo.abs() > 1e-2)
    diff = torch.angle(fx * fo.conj())[strong].abs()
    assert diff.median() < 1e-3


def test_apr_op_set_runs_inside_augmix():
    assert resolve_ops(["apr"]) == ["amplitude_mix"]
    aug = TrainAugment({"mode": "jsd", "ops": ["augmix", "prime", "apr"], "width": 2, "depth": [1, 2], "severity": 3})
    views = aug(images(5))
    assert len(views) == 3 and all(v.shape == (5, 3, 32, 32) for v in views)
    assert all(v.min() >= 0 and v.max() <= 1 for v in views)


# ---------------------------------------------------------------------------- EMA and distillation
def test_ema_averages_weights_with_warmup():
    a, b = small(0), small(1)
    ema = engine.ModelEMA(a, decay=0.9)
    ema.update(b)                                             # step 1: d = min(0.9, 2 / 11)
    d = 2 / 11
    k = "fc.weight"
    assert torch.allclose(ema.module.state_dict()[k], d * a.state_dict()[k] + (1 - d) * b.state_dict()[k], atol=1e-6)
    assert not any(p.requires_grad for p in ema.module.parameters())


def test_kd_loss_is_zero_for_identical_logits_and_grows_with_disagreement():
    z = torch.randn(8, 5)
    assert engine.kd_loss(z, z, 2.0).abs() < 1e-6
    assert engine.kd_loss(z, z.flip(1), 2.0) > 0.1


def test_training_loss_adds_the_distillation_term():
    torch.manual_seed(0)
    student, teacher = small(0).eval(), small(1).eval()          # eval: no dropout, so the two calls match
    views = [images(4, s, size=16) for s in (0, 1, 2)]
    y = torch.tensor([0, 1, 2, 0])
    stats = {}
    loss, ce, jsd, _ = engine.training_loss(student, views, y, True, 0.0, 1.0, False, teacher=teacher,
                                            kd_weight=0.5, kd_temperature=2.0, stats=stats)
    assert torch.allclose(loss, ce + jsd + 0.5 * stats["kd"], atol=1e-5) and stats["kd"] > 0
    plain, _, _, _ = engine.training_loss(student, views, y, True, 0.0, 1.0, False)
    assert torch.allclose(plain, ce + jsd, atol=1e-5)


def synthetic(n, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, n)
    x = rng.randint(0, 80, (n, 16, 16, 3))
    x[np.arange(n), :, :, y] += 120
    return x.astype(np.uint8), y.astype(np.int64)


@pytest.fixture
def cfg(monkeypatch):
    data = {"train": synthetic(256, 0), "val": synthetic(64, 1)}
    monkeypatch.setattr(engine, "load_split", lambda cfg, split: data[split])
    monkeypatch.setattr(engine, "load_val_c_split", lambda cfg: (*synthetic(64, 2), None, None))
    return {"name": "toy", "data": {"num_classes": 3},
            "train": {"epochs": 3, "batch_size": 32, "lr": 3e-3, "weight_decay": 0.01, "warmup_epochs": 1,
                      "label_smoothing": 0.1, "grad_clip": 1.0, "patience": 0, "amp": False, "eval_batch_size": 64},
            "augment": {"mode": "jsd", "ops": ["augmix", "prime", "apr"], "width": 2, "depth": [1, 2], "severity": 3,
                        "jsd_weight": 12.0}}


def test_ema_and_distillation_train_and_resume(cfg, tmp_path):
    cfg["train"].update(ema=0.9, distill={"teacher": "t", "weight": 1.0, "temperature": 2.0},
                        cudnn_benchmark=True, fused_optimizer=True, channels_last=True)   # GPU-only flags are ignored on CPU
    plain_cfg = cfg | {"train": {**cfg["train"], "distill": {}}}
    teacher = small(5)
    info = engine.train_run(plain_cfg, 0, tmp_path / "teach", teacher, torch.device("cpu"))     # EMA, no teacher
    assert info["best_val_c_acc"] > 0.8
    info = engine.train_run(cfg, 0, tmp_path / "student", small(0), torch.device("cpu"), teacher=teacher)
    log = pd.read_csv(tmp_path / "student" / "log.csv")
    assert info["best_val_c_acc"] > 0.8 and (log.kd > 0).all()
    with pytest.raises(ValueError):
        engine.train_run(cfg, 0, tmp_path / "no_teacher", small(0), torch.device("cpu"))


# ---------------------------------------------------------------------------- self-supervised learning
def test_nt_xent_prefers_matching_pairs():
    torch.manual_seed(0)
    z = torch.randn(16, 8)
    matched = nt_xent(z, z + 0.01 * torch.randn_like(z))
    shuffled = nt_xent(z, z[torch.randperm(16)])
    assert matched < shuffled


def test_knn_accuracy_on_separable_features():
    centers = F.normalize(torch.eye(4), dim=1)
    y = torch.arange(4).repeat(10)
    bank = F.normalize(centers[y] + 0.05 * torch.randn(40, 4), dim=1)
    assert knn_accuracy(bank, y, bank[:8], y[:8], 4, k=5) == 1.0


# ---------------------------------------------------------------------------- compatibility
def test_classifier_head_is_not_transferred():
    m = build_model({"name": "resnet18", "pretrained": False, "input_size": 64}, 43)
    assert _classifier_prefixes(m) == ["backbone.fc."]


def test_older_runs_keep_their_fingerprints():
    # values written next to the 15 Oct 2026 pilot checkpoints: they must not change, or every
    # finished run would be retrained
    saved = {"resnet18_full": "a30ebc9f2a868891", "fusion_v2": "872a492026715fa9", "ensemble_views_in": "75a3e5b3dd1569f6",
             "smallcnn_base": "058cf6d928c7450b", "deit_s_full": "ed74201c13179c56"}
    for name, fp in saved.items():
        assert fingerprint(load_run_config(name, "pilot")) == fp, name


def test_v3_dependencies_enter_the_fingerprint():
    student = load_run_config("student_v3", "pilot")
    other = json.loads(json.dumps(student))
    other["train"]["distill"]["teacher"] = "ensemble_views"
    assert fingerprint(student) != fingerprint(other)
    ssl = load_run_config("resnet18_v3_ssl", "pilot")
    assert fingerprint(ssl) != fingerprint(load_run_config("resnet18_v3", "pilot"))


# ---------------------------------------------------------------------------- heatmaps
def test_corruption_heatmaps_are_written_one_per_model(tmp_path):
    from roadlens.eval import plots

    rows = []
    for model in ("SVM", "Net A"):
        rows.append({"model": model, "corruption": "clean", "severity": 0, "acc": 0.9})
        for c in plots.HEATMAP_ORDER:
            rows += [{"model": model, "corruption": c, "severity": s, "acc": 1 - 0.1 * s} for s in range(1, 6)]
    paths = plots.plot_corruption_heatmaps(pd.DataFrame(rows), "model", tmp_path)
    assert sorted(p.name for p in paths) == ["net_a.png", "svm.png"] and all(p.stat().st_size > 5000 for p in paths)
