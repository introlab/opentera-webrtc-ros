import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.25):
        super(DistillationCrossEntropyLoss, self).__init__()
        self._alpha = alpha

    def forward(self, student_class_scores, target, teacher_class_scores):
        target_loss = F.cross_entropy(student_class_scores, target)
        teacher_loss = F.cross_entropy(student_class_scores, torch.softmax(teacher_class_scores, dim=1))
        return self._alpha * target_loss + (1 - self._alpha) * teacher_loss
