"""ImageNet-pretrained backbones from timm (ResNet-18, DeiT-S, Swin-T, ...), fine-tuned end to end."""

from __future__ import annotations

import timm

from .base import IMAGENET_MEAN, IMAGENET_STD, CamTarget, Classifier, Preprocess


class TimmClassifier(Classifier):
    """`name` is any timm model. The 64 px input is upsampled to `input_size` before the backbone:
    pretrained filters and patch embeddings expect objects at roughly ImageNet scale, and a ViT with
    16 px patches would otherwise see only 4 x 4 tokens."""

    def __init__(self, name: str, num_classes: int = 43, pretrained: bool = True,
                 input_size: int = 128, drop_path: float = 0.0, input_norm: str = "dataset"):
        super().__init__()
        kwargs = dict(pretrained=pretrained, num_classes=num_classes)
        if drop_path:
            kwargs["drop_path_rate"] = drop_path
        try:        # transformers need the input size up front (position embeddings are resampled)
            self.backbone = timm.create_model(name, img_size=input_size, **kwargs)
        except TypeError:   # CNNs accept any size
            self.backbone = timm.create_model(name, **kwargs)
        pcfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        self.pre = Preprocess(input_size, pcfg.get("mean", IMAGENET_MEAN), pcfg.get("std", IMAGENET_STD), input_norm)
        self.feature_dim = self.backbone.num_features
        self.arch = name

    def features_logits(self, x):
        feats = self.backbone.forward_features(self.pre(x))
        pooled = self.backbone.forward_head(feats, pre_logits=True)
        return pooled, self.backbone.get_classifier()(pooled)

    def cam_targets(self):
        b = self.backbone
        if hasattr(b, "layer4"):                                    # ResNet family
            return [CamTarget(self.arch, b.layer4, "conv")]
        if hasattr(b, "blocks") and hasattr(b, "num_prefix_tokens") and hasattr(b.blocks[-1], "norm1"):
            return [CamTarget(self.arch, b.blocks[-1].norm1, "tokens", b.num_prefix_tokens)]   # ViT / DeiT
        return []
