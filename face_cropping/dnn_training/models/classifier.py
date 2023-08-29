import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, backbone, class_count):
        super().__init__()

        self._backbone = backbone
        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=self._backbone.output_channels()[-1], out_features=class_count)
        )

    def forward(self, x):
        features = self._backbone(x)[-1]
        return self._classifier(features)
