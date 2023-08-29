import torch.nn as nn

from .backbone import Backbone
from ..common.yunet_modules import ConvHead, DWBlock


class YuNetBackbone(Backbone):
    def __init__(self, activation=nn.ReLU, channel_scale=1):
        super().__init__()
        self._channel_scale = channel_scale

        self._stage0 = ConvHead(in_channels=3,
                                mid_channels=int(16 * channel_scale),
                                out_channels=int(16 * channel_scale),
                                activation=activation)

        self._stage1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DWBlock(in_channels=int(16 * channel_scale), out_channels=int(16 * channel_scale), activation=activation),
            DWBlock(in_channels=int(16 * channel_scale), out_channels=int(64 * channel_scale), activation=activation)
        )
        self._stage2 = nn.Sequential(
            DWBlock(in_channels=int(64 * channel_scale), out_channels=int(64 * channel_scale), activation=activation),
            nn.MaxPool2d(kernel_size=2)
        )
        self._stage3 = nn.Sequential(
            DWBlock(in_channels=int(64 * channel_scale), out_channels=int(64 * channel_scale), activation=activation),
            nn.MaxPool2d(kernel_size=2)
        )
        self._stage4 = nn.Sequential(
            DWBlock(in_channels=int(64 * channel_scale), out_channels=int(64 * channel_scale), activation=activation),
            nn.MaxPool2d(kernel_size=2)
        )

    def output_channels(self):
        return [int(64 * self._channel_scale)]

    def output_strides(self):
        return [32]

    def forward(self, x):
        y0 = self._stage0(x)
        y1 = self._stage1(y0)
        y2 = self._stage2(y1)
        y3 = self._stage3(y2)
        y4 = self._stage4(y3)

        return [y4]
