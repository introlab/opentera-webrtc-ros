import torch.nn as nn
import torch.nn.functional as F

from ..common.yunet_modules import DWUnit


class YunetFpn(nn.Module):
    def __init__(self, channels, activation=nn.ReLU):
        super().__init__()
        if len(channels) == 0:
            raise ValueError('The channels list is empty.')
        if any(x != channels[0] for x in channels):
            raise ValueError('All channels must be equals')

        self._convs = nn.ModuleList([DWUnit(in_channels=c, out_channels=c, activation=activation) for c in channels])

    def forward(self, feature_maps):
        """
        :param feature_maps: List of features maps from low to high level
        :return: A new list of feature maps
        """
        if len(feature_maps) != len(self._convs):
            raise ValueError('feature_maps must have the same length as channels.')

        outputs = []
        for conv, x in zip(reversed(self._convs), reversed(feature_maps)):
            if len(outputs) > 0:
                y = conv(x) + F.interpolate(outputs[0], scale_factor=2, mode='nearest')
            else:
                y = conv(x)

            outputs.insert(0, y)

        return outputs
