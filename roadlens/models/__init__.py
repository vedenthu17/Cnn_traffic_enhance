from __future__ import annotations

from .base import CamTarget, Classifier  # noqa: F401
from .fusion import Fusion
from .smallcnn import SmallCNN
from .timm_models import TimmClassifier


def build_model(mcfg: dict, num_classes: int = 43, branches: list | None = None,
                pretrained: bool | None = None) -> Classifier:
    """Build a model from the `model:` section of a run config.

    name: smallcnn | fusion | <any timm model>.  For `fusion`, pass the already-built branch
    models. `pretrained=False` skips the ImageNet download (used when weights come from a checkpoint).
    """
    name = mcfg["name"]
    input_norm = mcfg.get("input_norm", "dataset")
    if name == "smallcnn":
        return SmallCNN(num_classes, mcfg.get("width", 32), mcfg.get("dropout", 0.3), input_norm)
    if name == "fusion":
        if not branches:
            raise ValueError("a fusion model needs its branch models")
        f = mcfg.get("fusion", {})
        return Fusion(branches, num_classes, f.get("mode", "gate"), f.get("gate_inputs", ("features", "probs")),
                      f.get("hidden", 256), f.get("dropout", 0.1), f.get("freeze_branches", True),
                      f.get("views", ("raw",)))
    use_pretrained = mcfg.get("pretrained", True) if pretrained is None else pretrained
    return TimmClassifier(name, num_classes, use_pretrained, mcfg.get("input_size", 128), mcfg.get("drop_path", 0.0),
                          input_norm)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
