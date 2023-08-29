import torch
import torch.nn as nn

from .sigmoid_focal_loss import SigmoidFocalLossWithLogits
from .eiou_loss import EiouLoss

from mmdet.sim_ota_assigner import SimOTAAssigner

BBOX_LOSS_SCALE = 5.0


class SingleClassDetectionLoss(nn.Module):
    def __init__(self):
        super(SingleClassDetectionLoss, self).__init__()
        self._confidence_loss = SigmoidFocalLossWithLogits()
        self._bbox_loss = EiouLoss()

    def forward(self, predictions, priors, decoded_bboxes, targets):
        assigner = SimOTAAssigner()

        N = predictions.size(0)
        loss = 0
        for n in range(N):
            loss += self._loss_single(assigner, predictions[n], priors, decoded_bboxes[n], targets[n])

        return loss / N

    def _loss_single(self, assigner, prediction, priors, decoded_bboxes, targets):
        with torch.no_grad():
            assignation_priors = _get_assignation_priors(priors)
            assign_result = assigner.assign(decoded_bboxes[:, 0:1],
                                            assignation_priors,
                                            decoded_bboxes[:, 1:],
                                            targets,
                                            torch.zeros(targets.size(0), dtype=torch.long, device=targets.device))

            positive_indexes, positive_target_indexes, confidence_target = _process_assign_result(assign_result,
                                                                                                  prediction)

        confidence_loss = self._confidence_loss(prediction[:, 0], confidence_target)
        bbox_loss = self._bbox_loss(decoded_bboxes[positive_indexes, 1:], targets[positive_target_indexes, :])

        return confidence_loss + BBOX_LOSS_SCALE * bbox_loss


class DistillationSingleClassDetectionLoss(nn.Module):
    def __init__(self, alpha=0.25):
        super(DistillationSingleClassDetectionLoss, self).__init__()
        self._confidence_loss = SigmoidFocalLossWithLogits()
        self._bbox_loss = EiouLoss()

        self._alpha = alpha

    def forward(self,
                student_predictions, student_priors, student_decoded_bboxes,
                teacher_predictions, teacher_priors, teacher_decoded_bboxes,
                targets):
        assigner = SimOTAAssigner()

        N = student_predictions.size(0)
        loss = 0
        for n in range(N):
            loss += self._loss_single(assigner,
                                      student_predictions[n], student_priors, student_decoded_bboxes[n],
                                      teacher_predictions[n], student_priors, teacher_decoded_bboxes[n],
                                      targets[n])

        return loss / N

    def _loss_single(self, assigner,
                     student_prediction, student_priors, student_decoded_bboxes,
                     teacher_prediction, teacher_priors, teacher_decoded_bboxes,
                     targets):
        with torch.no_grad():
            student_assignation_priors = _get_assignation_priors(student_priors)
            teacher_assignation_priors = _get_assignation_priors(teacher_priors)

            student_assign_result = assigner.assign(
                student_decoded_bboxes[:, 0:1],
                student_assignation_priors,
                student_decoded_bboxes[:, 1:],
                targets,
                torch.zeros(targets.size(0), dtype=torch.long, device=targets.device))
            teacher_assign_result = assigner.assign(
                teacher_decoded_bboxes[:, 0:1],
                teacher_assignation_priors,
                teacher_decoded_bboxes[:, 1:],
                targets,
                torch.zeros(targets.size(0), dtype=torch.long, device=targets.device))

            student_positive_indexes, student_positive_target_indexes, student_confidence_target = (
                _process_assign_result(student_assign_result, student_prediction))

            teacher_positive_indexes, _, _ = (
                _process_assign_result(teacher_assign_result, teacher_prediction))

        target_confidence_loss = self._confidence_loss(student_prediction[:, 0], student_confidence_target)
        target_bbox_loss = self._bbox_loss(student_decoded_bboxes[student_positive_indexes, 1:],
                                           targets[student_positive_target_indexes, :])

        teacher_confidence_loss = self._confidence_loss(student_prediction[:, 0],
                                                        torch.sigmoid(teacher_prediction[:, 0]))
        teacher_bbox_loss = self._bbox_loss(student_decoded_bboxes[teacher_positive_indexes, 1:],
                                            teacher_decoded_bboxes[teacher_positive_indexes, 1:])

        confidence_loss = self._alpha * target_confidence_loss + (1 - self._alpha) * teacher_confidence_loss
        bbox_loss = self._alpha * target_bbox_loss + (1 - self._alpha) * teacher_bbox_loss

        return confidence_loss + BBOX_LOSS_SCALE * bbox_loss


def _get_assignation_priors(priors):
    stride_w = priors[:, 2]
    stride_h = priors[:, 3]
    cx = priors[:, 0] + stride_w * 0.5
    cy = priors[:, 1] + stride_h * 0.5
    return torch.stack([cx, cy, stride_w, stride_h], dim=1)


def _process_assign_result(assign_result, prediction):
    positive_indexes = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).unique()
    positive_ious = assign_result.max_overlaps[positive_indexes]
    positive_target_indexes = assign_result.gt_inds[positive_indexes] - 1

    confidence_target = torch.zeros_like(prediction[:, 0])
    confidence_target[positive_indexes] = positive_ious

    return positive_indexes, positive_target_indexes, confidence_target
