import matplotlib.pyplot as plt

import torch

from modules.heads import filter_decoded_bboxes
from utils.path import to_path


class AveragePrecisionMetric:
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.1, nms_threshold=0.45):
        self._iou_threshold = iou_threshold
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold

        self._target_count = 0
        self._confidences = []
        self._true_positives = []
        self._false_positives = []

    def clear(self):
        self._target_count = 0
        self._confidences = []
        self._true_positives = []
        self._false_positives = []

    def add(self, predictions, targets):
        """
        :param prediction: tensor (N, M, 5) where the third dimension is [c, tl_x, tl_y, br_x, br_y]
        :param target: list of tensors (K, 4) where the second dimension is [tl_x, tl_y, br_x, br_y]
        """
        if predictions.size(0) != len(targets):
            raise ValueError('predictions.size(0) must be equal to len(targets)')

        for n in range(predictions.size(0)):
            self._add_single(predictions[n], targets[n])

    def _add_single(self, prediction, target):
        """
        :param prediction: tensor (M, 5) where the third dimension is [c, tl_x, tl_y, br_x, br_y]
        :param target: tensor (K, 4) where the second dimension is [tl_x, tl_y, br_x, br_y]
        """

        filtered_prediction = filter_decoded_bboxes(prediction,
                                                    confidence_threshold=self._confidence_threshold,
                                                    nms_threshold=self._nms_threshold)
        sorted_prediction = filtered_prediction[torch.argsort(filtered_prediction[:, 0], descending=True), :]
        self._target_count += target.shape[0]

        M = sorted_prediction.size(0)
        K = target.size(0)

        all_ious = calculate_iou(sorted_prediction[:, 1:].repeat_interleave(K, dim=0), target.repeat(M, 1))

        found_target = set()
        for m in range(M):
            confidence = sorted_prediction[m, 0]

            ious = all_ious[m * K:(m + 1) * K]
            target_index = torch.argmax(ious).item()
            iou = ious[target_index].item()

            true_positive = 0
            false_positive = 0
            if target_index in found_target or iou < self._iou_threshold:
                false_positive = 1
            elif iou > self._iou_threshold:
                true_positive = 1
                found_target.add(target_index)

            self._confidences.append(confidence)
            self._true_positives.append(true_positive)
            self._false_positives.append(false_positive)

    def get_value(self, output_curve=False, eps=1e-7):
        confidences = torch.tensor(self._confidences)
        true_positives = torch.tensor(self._true_positives)
        false_positives = torch.tensor(self._false_positives)

        sorted_index = torch.argsort(confidences, descending=True)
        true_positives = true_positives[sorted_index]
        false_positives = false_positives[sorted_index]

        cum_true_positives = torch.cumsum(true_positives, dim=0)
        cum_false_positives = torch.cumsum(false_positives, dim=0)

        recalls = cum_true_positives / (self._target_count + eps)
        precisions = cum_true_positives / (cum_true_positives + cum_false_positives + eps)

        sorted_index = torch.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]

        ap = torch.trapz(y=precisions, x=recalls).item()

        if output_curve:
            return ap, recalls.tolist(), precisions.tolist()
        else:
            return ap

    def save_curve(self, output_path, suffix=''):
        ap, recalls, precisions = self.get_value(output_curve=True)
        fig = plt.figure(figsize=(5, 5), dpi=300)
        ax1 = fig.add_subplot(111)

        ax1.plot(recalls, precisions)
        ax1.set_title(u'PR curve {}'.format(ap))
        ax1.set_xlabel(u'Recall')
        ax1.set_ylabel(u'Precision')

        fig.savefig(to_path(output_path) / f'pr_curve{suffix}.png')
        plt.close(fig)


def calculate_iou(bboxes_a, bboxes_b):
    """
    :param bboxes_a: tensor (N, 4) where the second dimension is [tl_x, tl_y, br_x, br_y]
    :param bboxes_b: tensor (N, 4) where the second dimension is [tl_x, tl_y, br_x, br_y]
    :return: iou (N) tensor
    """

    a_tl_x = bboxes_a[:, 0]
    a_tl_y = bboxes_a[:, 1]
    a_br_x = bboxes_a[:, 2]
    a_br_y = bboxes_a[:, 3]
    areas_a = (a_br_x - a_tl_x) * (a_br_y - a_tl_y)

    b_tl_x = bboxes_b[:, 0]
    b_tl_y = bboxes_b[:, 1]
    b_br_x = bboxes_b[:, 2]
    b_br_y = bboxes_b[:, 3]
    areas_b = (b_br_x - b_tl_x) * (b_br_y - b_tl_y)

    intersection_w = torch.min(a_br_x, b_br_x) - torch.max(a_tl_x, b_tl_x)
    intersection_h = torch.min(a_br_y, b_br_y) - torch.max(a_tl_y, b_tl_y)
    intersection_w = torch.max(intersection_w, torch.zeros_like(intersection_w))
    intersection_h = torch.max(intersection_h, torch.zeros_like(intersection_h))

    intersection_area = intersection_w * intersection_h

    return intersection_area / (areas_a + areas_b - intersection_area)
