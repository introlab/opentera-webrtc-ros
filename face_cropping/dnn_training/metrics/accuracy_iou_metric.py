import torch

from datasets.open_images_head_detector_dataset import TARGET_CLASS_INDEX, TARGET_X_INDEX, NO_HEAD_CLASS_INDEX

from models.head_detector import OUTPUT_CLASS_INDEX_MIN, OUTPUT_CLASS_INDEX_MAX, OUTPUT_X_INDEX


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
        prediction_class_index = prediction[:, OUTPUT_CLASS_INDEX_MIN:OUTPUT_CLASS_INDEX_MAX].argmax(dim=1)
        target_class_index = target[:, TARGET_CLASS_INDEX].long()

        for i in range(prediction.size(0)):
            if prediction_class_index[i] == target_class_index[i]:
                self._good += 1

            if target_class_index[i] != NO_HEAD_CLASS_INDEX:
                self._iou += compute_iou(prediction[i, OUTPUT_X_INDEX:], target[i, TARGET_X_INDEX:]).item()
                self._iou_count += 1
            elif target_class_index[i] == NO_HEAD_CLASS_INDEX and prediction_class_index[i] != NO_HEAD_CLASS_INDEX:
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
    :param box_a: (5) tensor center_x, center_y, w, h
    :param box_b: (5) tensor center_x, center_y, w, h
    :return: iou
    """
    areas_a = box_a[2] * box_a[3]
    areas_b = box_b[2] * box_b[3]

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
    tl_x = bbox[0] - bbox[2] / 2
    tl_y = bbox[1] - bbox[3] / 2
    br_x = bbox[0] + bbox[2] / 2
    br_y = bbox[1] + bbox[3] / 2

    return tl_x, tl_y, br_x, br_y
