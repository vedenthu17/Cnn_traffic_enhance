"""The v2 components: instance input normalisation, view experts, the stacked gate, the BatchNorm
statistics prior and temperature scaling (random weights / synthetic data, CPU)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import roadlens.engine as engine
from roadlens.data.gtsrb import halve_by_track
from roadlens.eval.calibration import fit_temperature, nll_at
from roadlens.models import Fusion, SmallCNN, build_model
from roadlens.models.views import VIEWS, expert_stats, gauss, image_stats, median3
from roadlens.runs import eval_tag
from roadlens.tta import BNAdapt, NoAdapt, PriorBatchNorm, Tent, batchnorm_layers, build_tta


def images(n=8, seed=0, size=32):
    return torch.rand(n, 3, size, size, generator=torch.Generator().manual_seed(seed))


def small(seed, classes=43, **kw):
    torch.manual_seed(seed)
    return SmallCNN(classes, width=8, **kw).eval()


# ---------------------------------------------------------------------------- instance normalisation
def gap(model, a, b) -> float:
    """Largest change of a logit between two inputs, relative to the largest logit."""
    with torch.no_grad():
        return float((model(a) - model(b)).abs().max() / model(a).abs().max())


def test_instance_norm_removes_per_channel_gain_and_offset():
    x = images(6) * 0.5 + 0.2
    gain = torch.tensor([0.05, 0.4, 1.7]).view(1, 3, 1, 1)
    offset = torch.tensor([0.3, -0.1, 0.05]).view(1, 3, 1, 1)
    model, plain = small(0, input_norm="instance"), small(0)
    assert gap(model, x, x * gain + offset) < 0.03
    mean = x.mean(dim=(2, 3), keepdim=True)
    for c in (0.4, 0.05):                                   # the contrast corruption, severities 1 and 5
        faint = (x - mean) * c + mean
        assert gap(model, x, faint) < 0.03
        assert gap(plain, x, faint) > 5 * gap(model, x, faint)
    assert torch.isfinite(model(torch.full((2, 3, 32, 32), 0.5))).all()         # a constant image is not a division by zero


def test_instance_norm_reaches_timm_models_and_rejects_unknown_names():
    m = build_model({"name": "resnet18", "pretrained": False, "input_size": 64, "input_norm": "instance"}, 43).eval()
    assert m.pre.norm == "instance"
    x = images(2, size=64)
    assert gap(m, x, 0.3 * x + 0.1) < 0.03
    assert build_model({"name": "resnet18", "pretrained": False, "input_size": 64}, 43).pre.norm == "dataset"
    with pytest.raises(ValueError):
        build_model({"name": "smallcnn", "input_norm": "batch"}, 43)


# ---------------------------------------------------------------------------- views and image statistics
def test_views_denoise_and_keep_flat_images():
    flat = torch.full((2, 3, 16, 16), 0.4)
    for name, fn in VIEWS.items():
        assert torch.allclose(fn(flat), flat, atol=1e-6), name
    spiky = flat.clone()
    spiky[0, :, 5, 7], spiky[1, 0, 0, 0] = 1.0, 0.0                              # isolated salt and pepper
    assert torch.equal(median3(spiky), flat)
    clean = images(4) * 0.2 + 0.4
    noisy = (clean + 0.15 * torch.randn(clean.shape, generator=torch.Generator().manual_seed(1))).clamp(0, 1)
    for fn in (median3, gauss):
        assert (fn(noisy) - fn(clean)).std() < 0.6 * (noisy - clean).std()        # the noise shrinks ...
        assert fn(noisy).shape == noisy.shape
    step = torch.zeros(1, 3, 16, 16)
    step[..., 8:] = 1.0
    assert torch.equal(median3(step), step)                                       # ... and a median keeps an edge


def test_image_stats_react_to_noise_blur_and_contrast():
    smooth = gauss(images(8, size=32), sigma=2.0)
    noisy = (smooth + 0.1 * torch.randn(smooth.shape, generator=torch.Generator().manual_seed(2))).clamp(0, 1)
    mean = smooth.mean(dim=(1, 2, 3), keepdim=True)
    faint = (smooth - mean) * 0.2 + mean
    s, n, f = image_stats(smooth), image_stats(noisy), image_stats(faint)
    assert s.shape == (8, 5) and torch.isfinite(s).all()
    assert (n[:, 2] > s[:, 2]).all() and (n[:, 3] > s[:, 3]).all()               # noise raises the high-frequency terms
    assert (f[:, 1] < 0.5 * s[:, 1]).all()                                       # low contrast lowers the contrast term
    assert (image_stats(gauss(noisy))[:, 2] < n[:, 2]).all()                     # blur lowers them again


def test_expert_stats():
    p = torch.tensor([[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]])
    q = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]])
    s = expert_stats([p, q])
    assert s.shape == (2, 7)
    assert torch.allclose(s[0, :3], torch.tensor([0.7, 0.5, float(-(p[0] * p[0].log()).sum() / np.log(3))]), atol=1e-6)
    assert torch.allclose(s[1, 2], torch.tensor(1.0), atol=1e-6)                 # uniform = maximal entropy
    assert torch.allclose(s[:, 6], torch.tensor([0.5, 1.0]))                     # row 0 disagrees, row 1 agrees
    assert float(s.min()) >= 0 and float(s.max()) <= 1 + 1e-6


# ---------------------------------------------------------------------------- view-expert fusion
def test_fusion_averages_over_branches_and_views():
    x = images(4)
    a, b = small(1), small(2)
    views = ("raw", "median3", "gauss")
    avg = Fusion([a, b], 43, mode="average", views=views).eval()
    expected = torch.stack([F.softmax(m(VIEWS[v](x)), 1) for v in views for m in (a, b)]).mean(0)
    out = avg.forward_detailed(x)
    assert avg.n_experts == 6 and out["weights"].shape == (4, 6) and len(out["branch_logits"]) == 6
    assert torch.allclose(out["logits"].exp(), expected, atol=1e-5)
    assert out["features"].shape == (4, avg.feature_dim)                         # features of the first view only
    assert torch.allclose(out["branch_logits"][0], a(x)) and torch.allclose(out["branch_logits"][3], b(median3(x)))
    assert torch.allclose(Fusion([a, b], 43, mode="average").eval()(x).exp(),
                          (F.softmax(a(x), 1) + F.softmax(b(x), 1)) / 2, atol=1e-5)   # default: one view, as before
    with pytest.raises(ValueError):
        Fusion([a, b], 43, views=("raw", "sharpen"))
    with pytest.raises(ValueError):
        Fusion([a, b], 43, gate_inputs=("logits",))


def test_stacked_gate_starts_as_the_average_and_trains_only_the_gate():
    x = images(6)
    a, b = small(1), small(2)
    kw = dict(views=("raw", "median3", "gauss"))
    gate = Fusion([a, b], 43, mode="gate", gate_inputs=("stats", "image"), hidden=8, dropout=0.0, **kw)
    avg = Fusion([a, b], 43, mode="average", **kw).eval()
    assert gate.gate[0].in_features == 3 * 6 + 1 + 5
    assert torch.allclose(gate.eval()(x), avg(x), atol=1e-5)
    assert set(gate.checkpoint_state()) == {"gate.0.weight", "gate.0.bias", "gate.3.weight", "gate.3.bias"}

    gate.train()
    assert all(not m.training for m in gate.branches.modules())
    opt = torch.optim.SGD([p for p in gate.parameters() if p.requires_grad], lr=1.0)
    before = [p.detach().clone() for p in gate.branches.parameters()]
    for _ in range(2):                                  # the first step only moves the zero-initialised output layer
        loss = F.cross_entropy(gate(x), torch.arange(6))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert all(torch.equal(p, q) for p, q in zip(gate.branches.parameters(), before))
    w = gate.eval().forward_detailed(x)["weights"]
    assert torch.allclose(w.sum(1), torch.ones(6), atol=1e-5) and not torch.allclose(w, torch.full_like(w, 1 / 6))
    with pytest.raises(ValueError):
        gate.branch_weights([None] * 6, [torch.zeros(6, 43)] * 6)                # image statistics need the input


def test_tent_on_a_view_fusion_adapts_the_cnn_branch():
    cnn = small(0)
    vit = build_model({"name": "deit_small_patch16_224", "pretrained": False, "input_size": 32}, 43).eval()
    fusion = Fusion([cnn, vit], 43, mode="average", views=("raw", "gauss")).eval()
    tent = build_tta("tent", fusion, amp=False, lr=1e-2, prior=8)
    assert tent.branch_ids == [0]
    out = tent.predict(images(1))
    assert out.shape == (1, 43) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------- BatchNorm statistics prior
def test_prior_batchnorm_interpolates_between_source_and_batch_statistics():
    model = small(0)
    with torch.no_grad():                               # give the source statistics some structure
        for m in batchnorm_layers(model):
            m.running_mean.normal_(0, 0.2)
            m.running_var.uniform_(0.5, 1.5)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = images(16) * 0.3 + 0.6
    src, plain = NoAdapt(model, amp=False).predict(x), BNAdapt(model, amp=False).predict(x)
    assert torch.allclose(BNAdapt(model, amp=False, prior=1e9).predict(x), src, atol=1e-4)      # all prior = source model
    assert torch.allclose(BNAdapt(model, amp=False, prior=1e-9).predict(x), plain, atol=1e-4)   # no prior = plain test-time BN
    mid = BNAdapt(model, amp=False, prior=16).predict(x)
    assert (mid - src).abs().mean() < (plain - src).abs().mean() and not torch.allclose(mid, src, atol=1e-4)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())                # caller's model untouched
    assert not any(isinstance(m, PriorBatchNorm) for m in model.modules())

    one = x[:1]                                          # a single image: its own statistics are useless
    src1 = NoAdapt(model, amp=False).predict(one)
    with_prior = BNAdapt(model, amp=False, prior=16).predict(one)
    assert (with_prior - src1).abs().mean() < 0.5 * (BNAdapt(model, amp=False).predict(one) - src1).abs().mean()


def test_tent_with_prior_updates_the_wrapped_layers_and_resets():
    model = small(0)
    tent = Tent(model, lr=1e-2, amp=False, prior=16)
    assert sum(isinstance(m, PriorBatchNorm) for m in tent.model.modules()) == len(batchnorm_layers(model))
    assert len(tent.params) == 2 * len(batchnorm_layers(model))
    x = images(4)
    first = tent.predict(x)
    assert torch.allclose(first, BNAdapt(model, amp=False, prior=16).predict(x), atol=1e-5)
    assert not torch.allclose(tent.predict(x), first, atol=1e-6)
    tent.reset()
    assert torch.allclose(tent.predict(x), first, atol=1e-5)


def test_eval_tags_keep_old_names_and_separate_the_new_options():
    assert eval_tag("gtsrb_c") == "gtsrb_c"
    assert eval_tag("gtsrb_c", {"method": "bn", "batch_size": 16, "lr": 1e-3}, [5], 500) == "gtsrb_c__bn_bs16__sev5__n500"
    assert eval_tag("gtsrb_c", {"method": "bn", "batch_size": 16, "prior": 0.0}, [5], 500) == "gtsrb_c__bn_bs16__sev5__n500"
    assert eval_tag("gtsrb_c", {"method": "bn", "batch_size": 1, "prior": 16}, [5], 100) == "gtsrb_c__bn_bs1_p16__sev5__n100"
    assert eval_tag("gtsrb_c", {"method": "tent", "batch_size": 64, "lr": 1e-3}) == "gtsrb_c__tent_bs64_lr0.001"
    assert eval_tag("gtsrb_c", calibrate=True) == "gtsrb_c__cal"


# ---------------------------------------------------------------------------- temperature scaling
def test_fit_temperature_recovers_a_known_scale_and_never_raises_the_nll():
    g = torch.Generator().manual_seed(0)
    z = 2.0 * torch.randn(20000, 10, generator=g)
    y = torch.multinomial(F.softmax(z, 1), 1, generator=g).squeeze(1)            # z is calibrated at T = 1
    for scale in (0.25, 1.0, 4.0):
        t = fit_temperature(z * scale, y)
        assert abs(t / scale - 1) < 0.05
        assert nll_at(z * scale, y, t) <= nll_at(z * scale, y, 1.0) + 1e-9
        assert torch.equal((z * scale / t).argmax(1), z.argmax(1))               # accuracy cannot change
    with pytest.raises(ValueError):
        fit_temperature(torch.zeros(0, 10), torch.zeros(0, dtype=torch.long))


# ---------------------------------------------------------------------------- stacking: split and training
def test_halve_by_track_shares_no_track_and_is_balanced():
    rng = np.random.RandomState(0)
    y = np.repeat(np.arange(6), [30, 60, 90, 30, 150, 240])                      # 1, 2, 3, 1, 5, 8 tracks of 30 frames
    tracks = y * 10000 + np.concatenate([np.repeat(np.arange(n // 30), 30) for n in np.bincount(y)])
    order = rng.permutation(len(y))
    y, tracks = y[order], tracks[order]
    a, b = halve_by_track(y, tracks, seed=0)
    assert len(np.intersect1d(a, b)) == 0 and len(a) + len(b) == len(y)
    assert not set(tracks[a]) & set(tracks[b])
    assert abs(len(set(tracks[a])) - len(set(tracks[b]))) <= 1                   # 20 tracks -> 10 + 10
    a2, b2 = halve_by_track(y, tracks, seed=0)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)


def synthetic(n, seed):
    """Three classes that differ in which colour channel is bright."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, n)
    x = rng.randint(0, 80, (n, 16, 16, 3))
    x[np.arange(n), :, :, y] += 120
    return x.astype(np.uint8), y.astype(np.int64)


@pytest.fixture
def stack_cfg(monkeypatch):
    data = (*synthetic(96, 0), *synthetic(64, 1), *synthetic(64, 2))
    monkeypatch.setattr(engine, "load_stack_data", lambda cfg: data)
    monkeypatch.setattr(engine, "load_split", lambda *a: pytest.fail("a stacked run must not load the training split"))
    return {
        "name": "toy_gate", "data": {"num_classes": 3},
        "train": {"fit_split": "val", "epochs": 3, "batch_size": 32, "lr": 2e-3, "weight_decay": 0.0, "warmup_epochs": 1,
                  "label_smoothing": 0.0, "patience": 0, "amp": False, "eval_batch_size": 64,
                  "ce_all_views": True, "eval_at_init": True},
        "augment": {"mode": "jsd", "ops": ["augmix", "prime"], "width": 2, "depth": [1, 2], "severity": 6, "jsd_weight": 0.0},
    }


def toy_gate():
    return Fusion([small(1, 3), small(2, 3)], 3, mode="gate", gate_inputs=("stats", "image"), hidden=8, dropout=0.0,
                  views=("raw", "gauss"))


def test_stacked_gate_trains_on_held_out_data_and_logs_every_epoch(stack_cfg, tmp_path):
    info = engine.train_run(stack_cfg, 0, tmp_path, toy_gate(), torch.device("cpu"))
    assert info["epochs_run"] == 3 and 0 <= info["best_epoch"] <= 3
    state = torch.load(tmp_path / "best.pt", weights_only=False)["model"]
    assert set(state) == {"gate.0.weight", "gate.0.bias", "gate.3.weight", "gate.3.bias"}
    again = engine.train_run(stack_cfg, 0, tmp_path / "again", toy_gate(), torch.device("cpu"))
    assert again["best_val_c_acc"] == info["best_val_c_acc"] and again["best_epoch"] == info["best_epoch"]


def test_selection_keeps_the_untrained_average_when_training_does_not_help(stack_cfg, tmp_path, monkeypatch):
    accs = iter([0.9, 0.8] + [0.5] * 20)                # epoch 0: (val, val-C) = (0.9, 0.8); every later epoch is worse
    monkeypatch.setattr(engine, "accuracy", lambda *a, **k: next(accs))
    gate = toy_gate()
    info = engine.train_run(stack_cfg, 0, tmp_path, gate, torch.device("cpu"))
    assert info["best_epoch"] == 0 and info["best_val_c_acc"] == 0.8 and info["val_acc_at_best"] == 0.9
    state = torch.load(tmp_path / "best.pt", weights_only=False)["model"]
    assert float(state["gate.3.weight"].abs().sum()) == 0 and float(state["gate.3.bias"].abs().sum()) == 0   # still the average

    stack_cfg["train"]["eval_at_init"] = False          # without it the first epoch always wins by default
    accs = iter([0.5] * 20)
    info = engine.train_run(stack_cfg, 0, tmp_path / "plain", toy_gate(), torch.device("cpu"))
    assert info["best_epoch"] == 1


def test_ce_all_views_changes_the_loss_only_when_asked(tmp_path):
    torch.manual_seed(0)
    model = small(0, 3)
    views = [images(4, s, size=16) for s in (0, 1, 2)]
    y = torch.tensor([0, 1, 2, 0])
    _, ce_clean, _, _ = engine.training_loss(model, views, y, True, 0.0, 0.0, False)
    _, ce_all, _, _ = engine.training_loss(model, views, y, True, 0.0, 0.0, False, ce_all_views=True)
    logits = [model(v) for v in views]
    assert torch.allclose(ce_clean, F.cross_entropy(logits[0], y), atol=1e-6)
    assert torch.allclose(ce_all, sum(F.cross_entropy(z, y) for z in logits) / 3, atol=1e-6)
    with pytest.raises(ValueError):
        engine.train_run({"name": "x", "data": {"num_classes": 3}, "train": {"fit_split": "test"}, "augment": {}},
                         0, tmp_path, small(0, 3).train(), torch.device("cpu"))
