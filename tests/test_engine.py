"""Training loop on synthetic data: it learns, early-stops, and an interrupted run resumes exactly."""

import numpy as np
import pandas as pd
import pytest
import torch

import roadlens.engine as engine
from roadlens.models import SmallCNN


def synthetic(n, seed):
    """Three classes that differ in which colour channel is bright."""
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
    return {
        "name": "toy", "data": {"num_classes": 3},
        "train": {"epochs": 4, "batch_size": 32, "lr": 3e-3, "weight_decay": 0.01, "warmup_epochs": 1,
                  "label_smoothing": 0.1, "grad_clip": 1.0, "patience": 0, "amp": False, "eval_batch_size": 64},
        "augment": {"mode": "jsd", "ops": ["augmix", "prime"], "width": 2, "depth": [1, 2], "severity": 3,
                    "jsd_weight": 12.0, "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.01}},
    }


def model(seed=0):
    torch.manual_seed(seed)
    return SmallCNN(3, width=8, dropout=0.1)


def test_training_learns_and_writes_outputs(cfg, tmp_path):
    info = engine.train_run(cfg, 0, tmp_path, model(), torch.device("cpu"))
    assert info["epochs_run"] == 4 and info["best_val_c_acc"] > 0.9
    assert (tmp_path / "best.pt").exists() and (tmp_path / "done.json").exists() and not (tmp_path / "last.pt").exists()
    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log.epoch) == [1, 2, 3, 4] and log.jsd.iloc[0] > 0 and log.loss.iloc[-1] < log.loss.iloc[0]
    assert engine.train_run(cfg, 0, tmp_path, model(), torch.device("cpu")) == info      # finished run: no-op


def test_same_seed_same_result_and_different_seed_differs(cfg, tmp_path):
    a = engine.train_run(cfg, 0, tmp_path / "a", model(), torch.device("cpu"))
    b = engine.train_run(cfg, 0, tmp_path / "b", model(), torch.device("cpu"))
    sa = torch.load(tmp_path / "a" / "best.pt", weights_only=False)["model"]
    sb = torch.load(tmp_path / "b" / "best.pt", weights_only=False)["model"]
    assert a["best_val_c_acc"] == b["best_val_c_acc"] and all(torch.equal(sa[k], sb[k]) for k in sa)
    engine.train_run(cfg, 1, tmp_path / "c", model(), torch.device("cpu"))
    sc = torch.load(tmp_path / "c" / "best.pt", weights_only=False)["model"]
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def test_interrupted_run_resumes_exactly(cfg, tmp_path, monkeypatch):
    engine.train_run(cfg, 0, tmp_path / "straight", model(), torch.device("cpu"))

    real_save, calls = engine._save, {"n": 0}

    def flaky_save(obj, path):
        real_save(obj, path)
        if path.name == "last.pt":
            calls["n"] += 1
            if calls["n"] == 2:                      # die right after the checkpoint of epoch 2
                raise KeyboardInterrupt

    monkeypatch.setattr(engine, "_save", flaky_save)
    with pytest.raises(KeyboardInterrupt):
        engine.train_run(cfg, 0, tmp_path / "resumed", model(), torch.device("cpu"))
    assert (tmp_path / "resumed" / "last.pt").exists() and not (tmp_path / "resumed" / "done.json").exists()
    monkeypatch.setattr(engine, "_save", real_save)
    info = engine.train_run(cfg, 0, tmp_path / "resumed", model(seed=123), torch.device("cpu"))   # weights come from the checkpoint
    assert info["epochs_run"] == 4

    a = pd.read_csv(tmp_path / "straight" / "log.csv").drop(columns=["epoch_s", "img_per_s"])
    b = pd.read_csv(tmp_path / "resumed" / "log.csv").drop(columns=["epoch_s", "img_per_s"])
    pd.testing.assert_frame_equal(a, b)
    sa = torch.load(tmp_path / "straight" / "best.pt", weights_only=False)["model"]
    sb = torch.load(tmp_path / "resumed" / "best.pt", weights_only=False)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_scratch_dir_keeps_per_epoch_checkpoints_out_of_the_run_dir(cfg, tmp_path, monkeypatch):
    engine.train_run(cfg, 0, tmp_path / "straight", model(), torch.device("cpu"))
    run, scratch = tmp_path / "run", tmp_path / "scratch"
    real_save, calls = engine._save, {"n": 0}

    def flaky_save(obj, path):
        real_save(obj, path)
        calls["n"] += path.name == "last.pt"
        if calls["n"] == 2 and path.name == "last.pt":
            raise KeyboardInterrupt

    monkeypatch.setattr(engine, "_save", flaky_save)
    with pytest.raises(KeyboardInterrupt):
        engine.train_run(cfg, 0, run, model(), torch.device("cpu"), scratch_dir=scratch, fingerprint="a")
    assert (scratch / "last.pt").exists() and (scratch / "best.pt").exists()
    assert not (run / "last.pt").exists() and not (run / "best.pt").exists()      # nothing big in the run dir yet
    monkeypatch.setattr(engine, "_save", real_save)

    info = engine.train_run(cfg, 0, run, model(seed=9), torch.device("cpu"), scratch_dir=scratch, fingerprint="a")
    assert info["epochs_run"] == 4 and (run / "best.pt").exists() and (run / "done.json").exists()
    assert not (scratch / "last.pt").exists() and not (scratch / "best.pt").exists()
    sa = torch.load(tmp_path / "straight" / "best.pt", weights_only=False)["model"]
    sb = torch.load(run / "best.pt", weights_only=False)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)                             # resumed from scratch, exactly


def test_resume_checkpoint_of_another_configuration_is_ignored(cfg, tmp_path, monkeypatch):
    real_save = engine._save

    def die_after_first_epoch(obj, path):
        real_save(obj, path)
        if path.name == "last.pt":
            raise KeyboardInterrupt

    monkeypatch.setattr(engine, "_save", die_after_first_epoch)
    with pytest.raises(KeyboardInterrupt):
        engine.train_run(cfg, 0, tmp_path / "old", model(), torch.device("cpu"), scratch_dir=tmp_path / "s", fingerprint="old")
    monkeypatch.setattr(engine, "_save", real_save)
    engine.train_run(cfg, 0, tmp_path / "new", model(), torch.device("cpu"), scratch_dir=tmp_path / "s", fingerprint="new")
    engine.train_run(cfg, 0, tmp_path / "ref", model(), torch.device("cpu"))
    a = pd.read_csv(tmp_path / "new" / "log.csv").drop(columns=["epoch_s", "img_per_s"])
    b = pd.read_csv(tmp_path / "ref" / "log.csv").drop(columns=["epoch_s", "img_per_s"])
    pd.testing.assert_frame_equal(a, b)                                           # started from epoch 1, not resumed


def test_early_stopping(cfg, tmp_path, monkeypatch):
    cfg["train"].update(epochs=12, patience=2)
    accs = iter([0.5, 0.6] + [0.4] * 40)             # (val, val-C) pairs: val-C = 0.6, then never better
    monkeypatch.setattr(engine, "accuracy", lambda *a, **k: next(accs))
    info = engine.train_run(cfg, 0, tmp_path, model(), torch.device("cpu"))
    assert info["early_stopped"] and info["best_epoch"] == 1 and info["epochs_run"] == 3
