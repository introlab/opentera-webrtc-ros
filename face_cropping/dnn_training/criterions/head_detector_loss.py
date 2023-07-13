import torch.nn as nn
import torch.nn.functional as F

from datasets.open_images_head_detector_dataset import TARGET_CLASS_INDEX, TARGET_X_INDEX
from datasets.open_images_head_detector_dataset import NO_HEAD_CLASS_INDEX

from models.head_detector import OUTPUT_CLASS_INDEX_MIN, OUTPUT_CLASS_INDEX_MAX, OUTPUT_X_INDEX


BOX_LOSS_SCALE = 10


class HeadDetectorLoss(nn.Module):
    def forward(self, prediction, target):
        N = prediction.size(0)

        class_loss = F.cross_entropy(prediction[:, OUTPUT_CLASS_INDEX_MIN:OUTPUT_CLASS_INDEX_MAX],
                                     target[:, TARGET_CLASS_INDEX].long())

        box_loss = 0.0
        box_count = 1e-6
        for n in range(N):
            if target[n, TARGET_CLASS_INDEX] != NO_HEAD_CLASS_INDEX:
                box_loss += F.mse_loss(prediction[n, OUTPUT_X_INDEX:], target[n, TARGET_X_INDEX:])
                box_count += 1

        return class_loss + BOX_LOSS_SCALE * box_loss / box_count
