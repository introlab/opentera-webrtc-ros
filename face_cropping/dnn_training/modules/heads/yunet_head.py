import torch
import torch.nn as nn

from .head import Head
from ..common.yunet_modules import DWUnit


class YunetHead(Head):
    def __init__(self, channels, strides, head_kernel_size=3, activation=nn.ReLU):
        super().__init__(strides)
        if len(channels) == 0:
            raise ValueError('The channels list is empty.')
        if len(channels) != len(strides):
            raise ValueError('channels must have the same length as strides.')
        if any(x != channels[0] for x in channels):
            raise ValueError('All channels must be equals')

        self._convs = nn.ModuleList([DWUnit(in_channels=c, out_channels=c, activation=activation) for c in channels])
        self._confidence_heads = nn.ModuleList([DWUnit(in_channels=c,
                                                       out_channels=1,
                                                       activation=activation,
                                                       kernel_size=head_kernel_size,
                                                       end_activation=False)
                                                for c in channels])
        self._bbox_heads = nn.ModuleList([DWUnit(in_channels=c,
                                                 out_channels=4,
                                                 activation=activation,
                                                 kernel_size=head_kernel_size,
                                                 end_activation=False)
                                          for c in channels])

    def forward(self, in_feature_maps):
        """
        :param in_feature_maps: List of features maps from low to high level
        :return: predictions (N, 5), priors (N, 4)
        """
        if len(in_feature_maps) != len(self._convs):
            raise ValueError('in_feature_maps must have the same length as channels.')

        mid_feature_maps = [conv(x) for conv, x in zip(self._convs, in_feature_maps)]

        predictions = []
        priors = self.generate_prior_grids(mid_feature_maps, offset=0.0)
        priors = [self.prior_grid_to_list(p) for p in priors]
        for i, mid_feature_map in enumerate(mid_feature_maps):
            confidences = self._confidence_heads[i](mid_feature_map)
            bboxes = self._bbox_heads[i](mid_feature_map)
            predictions.append(self.feature_map_grid_to_list(torch.cat([confidences, bboxes], dim=1)))

        return torch.cat(predictions, dim=1), torch.cat(priors, dim=0)

    def decode_predictions(self, predictions, priors):
        """
        :param predictions: tensor (N, 5)
        :param priors: tensor (N, 4)
        :return: bboxes (N, 5) where the second dimension is [c, tl_x, tl_y, br_x, br_y]
        """
        c = torch.sigmoid(predictions[:, :, 0])
        cx = predictions[:, :, 1] * priors[:, 2] + priors[:, 0]
        cy = predictions[:, :, 2] * priors[:, 3] + priors[:, 1]
        w = predictions[:, :, 3].exp() * priors[:, 2]
        h = predictions[:, :, 4].exp() * priors[:, 3]

        tl_x = cx - w / 2
        tl_y = cy - h / 2
        br_x = cx + w / 2
        br_y = cy + h / 2

        return torch.stack([c, tl_x, tl_y, br_x, br_y], dim=2)
