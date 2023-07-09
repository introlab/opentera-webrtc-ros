import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.open_images_head_detector_dataset import TARGET_CONFIDENCE_INDEX, TARGET_X_INDEX, TARGET_Y_INDEX
from datasets.open_images_head_detector_dataset import TARGET_W_INDEX, TARGET_H_INDEX

from metrics import compute_iou, get_tl_br_points


class HeadDetectorLoss(nn.Module):
    def forward(self, prediction, target):
        N = prediction.size(0)

        loss = 0.0
        for n in range(N):
            loss += F.binary_cross_entropy(prediction[n, TARGET_CONFIDENCE_INDEX],
                                           target[n, TARGET_CONFIDENCE_INDEX])
            if target[n, TARGET_CONFIDENCE_INDEX] != 0:
                loss += _calculate_ciou(prediction[n], target[n])

        return loss / N


def _calculate_ciou(box_a, box_b):
    """
    :param box_a: (4) tensor confidence center_x, center_y, w, h
    :param box_b: (4) tensor confidence center_x, center_y, w, h
    :return: ciou loss
    """
    iou = compute_iou(box_a, box_b)
    normalized_center_distance_squared = _calculate_normalized_center_distance_squared(box_a, box_b)
    v = _calculate_aspect_ratio_consistency(box_a, box_b)
    alpha = _calculate_ciou_trade_off_parameter(iou, v)

    ciou = 1 - iou + normalized_center_distance_squared + alpha * v
    return ciou.mean(dim=0)


def _calculate_normalized_center_distance_squared(box_a, box_b):
    """
    :param box_a: (4) tensor confidence center_x, center_y, w, h
    :param box_b: (4) tensor confidence center_x, center_y, w, h
    :return: normalized center distance squared (N) tensor
    """
    a_tl_x, a_tl_y, a_br_x, a_br_y = get_tl_br_points(box_a)
    b_tl_x, b_tl_y, b_br_x, b_br_y = get_tl_br_points(box_b)

    box_tl_x = torch.min(a_tl_x, b_tl_x)
    box_tl_y = torch.min(a_tl_y, b_tl_y)
    box_br_x = torch.max(a_br_x, b_br_x)
    box_br_y = torch.max(a_br_y, b_br_y)

    box_w = box_br_x - box_tl_x
    box_h = box_br_y - box_tl_y

    c_squared = torch.pow(box_w, 2) + torch.pow(box_h, 2)
    center_distance_x = box_a[TARGET_X_INDEX] - box_b[TARGET_X_INDEX]
    center_distance_y = box_a[TARGET_Y_INDEX] - box_b[TARGET_Y_INDEX]
    center_distance_squared = torch.pow(center_distance_x, 2) + torch.pow(center_distance_y, 2)

    return center_distance_squared / c_squared


def _calculate_aspect_ratio_consistency(box_a, box_b):
    """
    :param box_a: (4) tensor confidence center_x, center_y, w, h
    :param box_b: (4) tensor confidence center_x, center_y, w, h
    :return: aspect ratio consistency value (N) tensor
    """
    a = torch.atan(box_a[TARGET_W_INDEX] / box_a[TARGET_H_INDEX]) - \
        torch.atan(box_b[TARGET_W_INDEX] / box_b[TARGET_H_INDEX])
    return 4 / math.pi ** 2 * torch.pow(a, 2)


def _calculate_ciou_trade_off_parameter(iou, v):
    return v / (1 - iou + v)
