import torch.nn as nn

from modules.yunet_backbone import YuNetBackbone


class YuNetClassifier(nn.Module):
    def __init__(self, class_count, activation=nn.ReLU, channel_scale=1):
        super().__init__()

        self._backbone = YuNetBackbone(activation=activation, channel_scale=channel_scale)
        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=64 * channel_scale, out_features=class_count)
        )

    def forward(self, x):
        features = self._backbone(x)[-1]
        return self._classifier(features)
