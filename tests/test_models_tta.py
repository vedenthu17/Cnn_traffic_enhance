"""Models, fusion, Grad-CAM and test-time adaptation invariants (random weights, CPU)."""

import pytest
import torch
import torch.nn.functional as F

from roadlens.eval.gradcam import grad_cam
from roadlens.models import Fusion, SmallCNN, build_model
from roadlens.tta import BNAdapt, NoAdapt, Tent, batchnorm_layers, build_tta, entropy


def images(n=8, seed=0):
    return torch.rand(n, 3, 64, 64, generator=torch.Generator().manual_seed(seed))


def small(seed):
    torch.manual_seed(seed)
    return SmallCNN(43, width=8).eval()


def test_smallcnn_and_timm_shapes():
    x = images(2)
    feat, logits = small(0).features_logits(x)
    assert feat.shape == (2, 64) and logits.shape == (2, 43)
    for name in ("resnet18", "deit_small_patch16_224"):
        m = build_model({"name": name, "pretrained": False, "input_size": 64}, 43).eval()
        feat, logits = m.features_logits(x)
        assert feat.shape == (2, m.feature_dim) and logits.shape == (2, 43)
        assert len(m.cam_targets()) == 1


def test_fusion_modes():
    x = images(4)
    a, b = small(1), small(2)
    pa, pb = F.softmax(a(x), 1), F.softmax(b(x), 1)
    avg = Fusion([a, b], 43, mode="average").eval()
    assert torch.allclose(avg(x).exp(), (pa + pb) / 2, atol=1e-5)
    conf = Fusion([a, b], 43, mode="confidence").eval()
    w = torch.stack([pa.amax(1), pb.amax(1)], 1)
    w = w / w.sum(1, keepdim=True)
    assert torch.allclose(conf(x).exp(), w[:, :1] * pa + w[:, 1:] * pb, atol=1e-5)
    gate = Fusion([a, b], 43, mode="gate").eval()
    out = gate.forward_detailed(x)
    assert torch.allclose(out["weights"], torch.full((4, 2), 0.5))          # zero-initialised gate = average
    assert torch.allclose(out["logits"].exp().sum(1), torch.ones(4), atol=1e-5)
    assert torch.allclose(out["logits"].exp(), (pa + pb) / 2, atol=1e-5)


def test_fusion_freezes_branches_and_checkpoints_only_the_gate():
    f = Fusion([small(1), small(2)], 43, mode="gate")
    f.train()
    assert all(not m.training for m in f.branches.modules())
    assert f.gate.training
    trainable = [n for n, p in f.named_parameters() if p.requires_grad]
    assert trainable and all(not n.startswith("branches.") for n in trainable)
    state = f.checkpoint_state()
    assert state and all(not k.startswith("branches.") for k in state)
    g = Fusion([small(1), small(2)], 43, mode="gate")
    torch.nn.init.normal_(f.gate[-1].weight)
    g.load_checkpoint_state(f.checkpoint_state())
    x = images(3)
    assert torch.allclose(f.eval()(x), g.eval()(x), atol=1e-6)
    with pytest.raises(RuntimeError):
        g.load_checkpoint_state({"gate.0.weight": state["gate.0.weight"]})


def test_gradcam_shapes_and_range():
    x = images(3)
    cnn = small(0)
    cams, logits = grad_cam(cnn, cnn.cam_targets(), x)
    assert cams[0].shape == (3, 64, 64) and float(cams[0].min()) >= 0 and float(cams[0].max()) <= 1 + 1e-6
    vit = build_model({"name": "deit_small_patch16_224", "pretrained": False, "input_size": 64}, 43).eval()
    cams, _ = grad_cam(vit, vit.cam_targets(), x)
    assert cams[0].shape == (3, 64, 64) and torch.isfinite(cams[0]).all()
    fusion = Fusion([cnn, vit], 43, mode="gate").eval()                    # frozen branches still give gradients
    cams, _ = grad_cam(fusion, fusion.cam_targets(), x)
    assert len(cams) == 2 and all(float(c.abs().sum()) > 0 for c in cams)


def test_bn_adapt_uses_batch_statistics_and_leaves_source_untouched():
    model = small(0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = images(16) * 0.3 + 0.6                                             # shifted input distribution
    src = NoAdapt(model, amp=False).predict(x)
    assert torch.allclose(src, model(x))
    adapted = BNAdapt(model, amp=False)
    out = adapted.predict(x)
    assert not torch.allclose(out, src, atol=1e-3)
    assert torch.allclose(adapted.predict(x), out)                         # stateless across batches
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items() if "num_batches" not in k)
    assert not any(m.training for m in model.modules())
    assert all(torch.equal(v, adapted.model.state_dict()[k]) for k, v in before.items() if "running" in k)


def test_tent_updates_only_batchnorm_affine_and_resets():
    model = small(0)
    tent = Tent(model, lr=1e-2, amp=False)
    names = {n for n, p in tent.model.named_parameters() if p.requires_grad}
    bn_names = {f"{n}.{k}" for n, m in tent.model.named_modules() if m in batchnorm_layers(tent.model) for k in ("weight", "bias")}
    assert names == bn_names and len(names) == 2 * len(batchnorm_layers(model))
    x = images(16)
    first = tent.predict(x)
    assert torch.allclose(first, BNAdapt(model, amp=False).predict(x), atol=1e-5)   # logits from before the step
    second = tent.predict(x)
    assert not torch.allclose(first, second, atol=1e-6)                    # the step changed the model
    src, new = model.state_dict(), tent.model.state_dict()
    changed = {k for k in src if not torch.equal(src[k], new[k]) and "num_batches" not in k}
    assert changed and changed <= bn_names
    ent_before = float(F.softmax(first, 1).mul(F.log_softmax(first, 1)).sum(1).mean().neg())
    for _ in range(10):
        last = tent.predict(x)
    ent_after = float(F.softmax(last, 1).mul(F.log_softmax(last, 1)).sum(1).mean().neg())
    assert ent_after < ent_before                                          # entropy is being minimised
    tent.reset()
    assert torch.allclose(tent.predict(x), first, atol=1e-5)


def test_tent_on_fusion_adapts_only_the_cnn_branch_and_works_at_batch_size_one():
    cnn = small(0)
    vit = build_model({"name": "deit_small_patch16_224", "pretrained": False, "input_size": 64}, 43).eval()
    fusion = Fusion([cnn, vit], 43, mode="gate").eval()
    tent = build_tta("tent", fusion, amp=False, lr=1e-2)
    trainable = [n for n, p in tent.model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("branches.0.") for n in trainable)
    out = tent.predict(images(1))
    assert out.shape == (1, 43) and torch.isfinite(out).all()

    # default 'branch' objective: the CNN branch's own entropy goes down even if the gate ignores that branch
    with torch.no_grad():
        fusion.gate[-1].bias.copy_(torch.tensor([-30.0, 30.0]))            # all weight on the ViT branch
    x = images(8)
    tent = build_tta("tent", fusion, amp=False, lr=1e-2)

    def cnn_entropy():
        with torch.no_grad():
            return float(entropy(tent.model.forward_detailed(x)["branch_logits"][0]).mean())

    before = cnn_entropy()
    for _ in range(10):
        tent.predict(x)
    assert cnn_entropy() < before
    assert build_tta("tent", fusion, amp=False, objective="fused").branch_ids is None
    with pytest.raises(ValueError):
        build_tta("bn", vit)
    with pytest.raises(ValueError):
        build_tta("tent", vit)
