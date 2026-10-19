"""SmallCNN: the from-scratch baseline, and the model whose errors normalise mCE."""

from __future__ import annotations

import torch.nn as nn

from .base import CamTarget, Classifier, Preprocess


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SmallCNN(Classifier):
    """Four conv-BN-ReLU x2 stages (width, 2w, 4w, 8w) with max-pooling, global average pooling,
    dropout and a linear head. 1.2 M parameters at width 32."""

    def __init__(self, num_classes: int = 43, width: int = 32, dropout: float = 0.3, input_norm: str = "dataset"):
        super().__init__()
        chans = [width, width * 2, width * 4, width * 8]
        self.pre = Preprocess(None, norm=input_norm)
        stages, cin = [], 3
        for i, c in enumerate(chans):
            stages.append(_block(cin, c))
            if i < len(chans) - 1:
                stages.append(nn.MaxPool2d(2))
            cin = c
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)
        self.feature_dim = cin

    def features_logits(self, x):
        feat = self.pool(self.features(self.pre(x))).flatten(1)
        return feat, self.fc(self.drop(feat))

    def cam_targets(self):
        return [CamTarget("smallcnn", self.features[-1], "conv")]
