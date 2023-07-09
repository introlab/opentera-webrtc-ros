import torch.nn as nn

import torchvision.models as models


class EfficientNetBackbone(nn.Module):
    SUPPORTED_TYPES = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                       'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
    LAST_CHANNEL_COUNT_BY_TYPE = {'efficientnet_b0': 1280,
                                  'efficientnet_b1': 1280,
                                  'efficientnet_b2': 1408,
                                  'efficientnet_b3': 1536,
                                  'efficientnet_b4': 1792,
                                  'efficientnet_b5': 2048,
                                  'efficientnet_b6': 2304,
                                  'efficientnet_b7': 2560}

    def __init__(self, type, pretrained_backbone=True):
        super(EfficientNetBackbone, self).__init__()

        if pretrained_backbone:
            backbone_weights = 'DEFAULT'
        else:
            backbone_weights = None

        if type not in self.SUPPORTED_TYPES or type not in self.LAST_CHANNEL_COUNT_BY_TYPE:
            raise ValueError('Invalid backbone type')

        self._features_layers = models.__dict__[type](weights=backbone_weights).features
        self._last_channel_count = self.LAST_CHANNEL_COUNT_BY_TYPE[type]

    def forward(self, x):
        return self._features_layers(x)

    def last_channel_count(self):
        return self._last_channel_count


class HeadDetector(nn.Module):
    def __init__(self, backbone):
        super(HeadDetector, self).__init__()

        self._backbone = backbone
        self._global_avg_pool = nn.AdaptiveAvgPool2d(8)
        self._detector = nn.Sequential(
            nn.Linear(in_features=self._backbone.last_channel_count() * 64, out_features=5),
            nn.Sigmoid()
        )

    def class_count(self):
        return self._class_count

    def forward(self, x):
        features = self._global_avg_pool(self._backbone(x))
        return self._detector(features.view(x.size()[0], -1)).clip(min=1e-6, max=0.999999)
