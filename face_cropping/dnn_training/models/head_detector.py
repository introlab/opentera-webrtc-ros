import torch
import torch.nn as nn

import torchvision.models as models


IMAGE_SIZE = (256, 256)

OUTPUT_CLASS_INDEX_MIN = 0
OUTPUT_CLASS_INDEX_MAX = 3
OUTPUT_X_INDEX = 3
OUTPUT_Y_INDEX = 4
OUTPUT_W_INDEX = 5
OUTPUT_H_INDEX = 6


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
        self._features_layers[0] = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        self._last_channel_count = self.LAST_CHANNEL_COUNT_BY_TYPE[type]

        ys = torch.linspace(0.0, 1.0, steps=IMAGE_SIZE[0])
        xs = torch.linspace(0.0, 1.0, steps=IMAGE_SIZE[1])
        x_grid, y_grid = torch.meshgrid(xs, ys, indexing='xy')
        self.register_buffer('_x_grid', x_grid.unsqueeze(0).unsqueeze(0).float().clone())
        self.register_buffer('_y_grid', y_grid.unsqueeze(0).unsqueeze(0).float().clone())

    def forward(self, x):
        N = x.size(0)
        x = torch.cat([x, self._x_grid.repeat(N, 1, 1, 1), self._y_grid.repeat(N, 1, 1, 1)], dim=1)
        return self._features_layers(x)

    def last_channel_count(self):
        return self._last_channel_count


class HeadDetector(nn.Module):
    def __init__(self, backbone):
        super(HeadDetector, self).__init__()

        self._backbone = backbone
        self._global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self._detector = nn.Sequential(
            nn.Linear(in_features=self._backbone.last_channel_count(), out_features=7),
        )

    def class_count(self):
        return self._class_count

    def forward(self, x):
        features = self._global_avg_pool(self._backbone(x))
        y = self._detector(features.view(x.size()[0], -1))
        class_scores = y[:, :3]
        center = torch.sigmoid(y[:, 3:5]) * 2.0 - 0.5
        size = 4.0 * torch.sigmoid(y[:, 5:7]) ** 2.0
        return torch.cat([class_scores, center, size], dim=1)
