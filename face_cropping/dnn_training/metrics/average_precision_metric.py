import torch

from modules.heads import filter_decoded_bboxes


class AveragePrecisionMetric:
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.1, nms_threshold=0.45):
        self._iou_threshold = iou_threshold
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold

        self._target_count = 0
        self._results = []

    def clear(self):
        self._target_count = 0
        self._results = []

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

        found_target = set()
        for m in range(M):
            confidence = sorted_prediction[m, 0]

            ious = calculate_iou(sorted_prediction[m:m + 1, 1:].repeat(K, 1), target)
            target_index = torch.argmax(ious)
            iou = ious[target_index]

            true_positive = 0
            false_positive = 0
            if target_index in found_target or iou < self._iou_threshold:
                false_positive = 1
            elif iou > self._iou_threshold:
                true_positive = 1
            found_target.add(target_index)

            self._results.append({
                'confidence': confidence,
                'true_positive': true_positive,
                'false_positive': false_positive,
            })

    def get_value(self):
        sorted_results = sorted(self._results, key=lambda result: result['confidence'], reverse=True)

        recalls = [0]
        precisions = [1]

        true_positive = 0
        false_positive = 0
        for result in sorted_results:
            true_positive += result['true_positive']
            false_positive += result['false_positive']

            recalls.append(true_positive / self._target_count if self._target_count > 0 else 0)

            precision_denominator = true_positive + false_positive
            precisions.append(true_positive / precision_denominator if precision_denominator > 0 else 1)

        recalls = torch.tensor(recalls)
        precisions = torch.tensor(precisions)

        sorted_index = torch.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]

        return torch.trapz(y=precisions, x=recalls)


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
