import torch

from datasets.open_images_head_detector_dataset import TARGET_CONFIDENCE_INDEX, TARGET_X_INDEX, TARGET_Y_INDEX
from datasets.open_images_head_detector_dataset import TARGET_W_INDEX, TARGET_H_INDEX


class AccuracyIoUMetric:
    def __init__(self):
        self._good = 0
        self._count = 0
        self._iou = 0
        self._iou_count = 0

    def clear(self):
        self._good = 0
        self._count = 0
        self._iou = 0
        self._iou_count = 0

    def add(self, prediction, target):
        prediction_presence = prediction[:, TARGET_CONFIDENCE_INDEX] > 0.5
        target_presence = target[:, TARGET_CONFIDENCE_INDEX] > 0.5

        for i in range(prediction.size(0)):
            if prediction_presence[i].item() and target_presence[i].item():
                self._good += 1
                self._iou += compute_iou(prediction[i], target[i]).item()
                self._iou_count += 1
            elif not prediction_presence[i].item() and not target_presence[i].item():
                self._good += 1
            elif not prediction_presence[i].item() and target_presence[i].item():
                self._iou_count += 1

            self._count += 1

    def get_accuracy(self):
        if self._count == 0:
            return 0
        return self._good / self._count

    def get_iou(self):
        if self._iou_count == 0:
            return 0
        return self._iou / self._iou_count


def compute_iou(box_a, box_b):
    """
    :param box_a: (5) tensor confidence center_x, center_y, w, h
    :param box_b: (5) tensor confidence center_x, center_y, w, h
    :return: iou
    """
    areas_a = box_a[TARGET_W_INDEX] * box_a[TARGET_H_INDEX]
    areas_b = box_b[TARGET_W_INDEX] * box_b[TARGET_H_INDEX]

    a_tl_x, a_tl_y, a_br_x, a_br_y = get_tl_br_points(box_a)
    b_tl_x, b_tl_y, b_br_x, b_br_y = get_tl_br_points(box_b)

    intersection_w = torch.min(a_br_x, b_br_x) - torch.max(a_tl_x, b_tl_x)
    intersection_h = torch.min(a_br_y, b_br_y) - torch.max(a_tl_y, b_tl_y)
    intersection_w = torch.max(intersection_w, torch.zeros_like(intersection_w))
    intersection_h = torch.max(intersection_h, torch.zeros_like(intersection_h))

    intersection_area = intersection_w * intersection_h

    return intersection_area / (areas_a + areas_b - intersection_area)


def get_tl_br_points(bbox):
    """
    :param bboxes: (4) tensor center_x, center_y, w, h
    :return: top left x, top left y, bottom right x and bottom right y
    """
    tl_x = bbox[TARGET_X_INDEX] - bbox[TARGET_W_INDEX] / 2
    tl_y = bbox[TARGET_Y_INDEX] - bbox[TARGET_H_INDEX] / 2
    br_x = bbox[TARGET_X_INDEX] + bbox[TARGET_W_INDEX] / 2
    br_y = bbox[TARGET_Y_INDEX] + bbox[TARGET_H_INDEX] / 2

    return tl_x, tl_y, br_x, br_y
