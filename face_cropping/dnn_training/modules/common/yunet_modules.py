from typing import Callable

import torch.nn as nn


class DWUnit(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, end_activation=True, kernel_size=3):
        super().__init__()

        if not isinstance(activation, Callable):
            raise ValueError('The activation is invalid.')

        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, stride=1, padding=0, bias=False, groups=1),
                  nn.BatchNorm2d(out_channels),
                  activation(),
                  nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                            bias=not end_activation, groups=out_channels)
                  ]

        if end_activation:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation())

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()

        if not isinstance(activation, Callable):
            raise ValueError('The activation is invalid.')

        self._layers = nn.Sequential(
            DWUnit(in_channels=in_channels, out_channels=in_channels, activation=activation),
            DWUnit(in_channels=in_channels, out_channels=out_channels, activation=activation)
        )

    def forward(self, x):
        return self._layers(x)


class ConvHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, activation=nn.ReLU):
        super().__init__()

        if not isinstance(activation, Callable):
            raise ValueError('The activation is invalid.')

        self._layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation(),

            DWUnit(in_channels=mid_channels, out_channels=out_channels, activation=activation)
        )

    def forward(self, x):
        return self._layers(x)
