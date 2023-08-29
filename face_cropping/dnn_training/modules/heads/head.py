from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import torchvision

CONFIDENCE_INDEX = 0
TL_X_INDEX = 1
TL_Y_INDEX = 2
BR_X_INDEX = 3
BR_Y_INDEX = 4


class Head(nn.Module, ABC):
    def __init__(self, strides):
        super().__init__()
        self._strides = strides

    def generate_prior_grids(self, feature_maps, offset=0.5):
        if len(self._strides) != len(feature_maps):
            raise ValueError('feature_maps must have the same length as strides.')

        grid_priors = []
        for feature_map, stride in zip(feature_maps, self._strides):
            _, _, H, W = feature_map.size()
            xs = (torch.arange(0, W, device=feature_map.device, dtype=feature_map.dtype) + offset) * stride
            ys = (torch.arange(0, H, device=feature_map.device, dtype=feature_map.dtype) + offset) * stride
            cx, cy = torch.meshgrid(xs, ys, indexing='xy')

            strides = xs.new_full((H, W), stride)

            grid_priors.append(torch.stack([cx, cy, strides, strides]))

        return grid_priors

    def feature_map_grid_to_list(self, x):
        N, C, H, W = x.size()
        return x.view(N, C, -1).permute(0, 2, 1)

    def prior_grid_to_list(self, x):
        C, H, W = x.size()
        return x.view(C, -1).permute(1, 0)

    @abstractmethod
    def forward(self, feature_maps):
        """
        :param in_feature_maps: List of features maps from low to high level
        :return: predictions (N, 5), priors (N, 4)
        """
        raise NotImplementedError

    @abstractmethod
    def decode_predictions(self, predictions, priors):
        """
        :param predictions: tensor (N, 5)
        :param priors: tensor (N, 4)
        :return: bboxes (N, 5) where the second dimension is [c, tl_x, tl_y, br_x, br_y]
        """
        raise NotImplementedError


def filter_decoded_bboxes(bboxes, confidence_threshold=0.5, nms_threshold=0.45):
    """
    :param bboxes: bboxes (N, 5) where the second dimension is [c, tl_x, tl_y, br_x, br_y]
    :param confidence_threshold:
    :param nms_threshold:
    :return: bboxes (N, 5) where the second dimension is [c, tl_x, tl_y, br_x, br_y]
    """
    confident_bboxes = bboxes[bboxes[:, CONFIDENCE_INDEX] >= confidence_threshold, :]
    indexes = torchvision.ops.nms(confident_bboxes[:, TL_X_INDEX:],
                                  confident_bboxes[:, CONFIDENCE_INDEX],
                                  iou_threshold=nms_threshold)
    return confident_bboxes[indexes, :]
