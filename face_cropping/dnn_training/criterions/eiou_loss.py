import torch
import torch.nn as nn


class EiouLoss(nn.Module):
    def __init__(self, smooth_point=0.1, eps=1e-7):
        super(EiouLoss, self).__init__()
        self._smooth_point = smooth_point
        self._eps = eps

    def forward(self, predictions, targets):
        """
        :param prediction: tensor (M, 5) where the second dimension is [tl_x, tl_y, br_x, br_y]
        :param target: tensor (K, 4) where the second dimension is [tl_x, tl_y, br_x, br_y]
        """
        if predictions.size(0) == 0 or targets.size(0) == 0:
            return torch.tensor(0.0)

        xp1 = predictions[:, 0]
        yp1 = predictions[:, 1]
        xp2 = predictions[:, 2]
        yp2 = predictions[:, 3]

        xt1 = targets[:, 0]
        yt1 = targets[:, 1]
        xt2 = targets[:, 2]
        yt2 = targets[:, 3]

        x0 = torch.min(xt1, xp1)
        y0 = torch.min(yt1, yp1)
        x1 = torch.max(xt1, xp1)
        y1 = torch.max(yt1, yp1)
        x2 = torch.min(xt2, xp2)
        y2 = torch.min(yt2, yp2)

        xmin = torch.min(x1, x2)
        ymin = torch.min(y1, y2)
        xmax = torch.max(x1, x2)
        ymax = torch.max(y1, y2)

        intersection = ((x2 - x0) * (y2 - y0) +
                        (xmin - x0) * (ymin - y0) -
                        (x1 - x0) * (ymax - y0) -
                        (xmax - x0) * (y1 - y0))
        union = (xt2 - xt1) * (yt2 - yt1) + (xp2 - xp1) * (yp2 - yp1) - intersection

        iou = 1 - (intersection / (union + self._eps))

        smooth_mask = (iou < self._smooth_point).detach().float()
        loss = (0.5 * smooth_mask * iou ** 2 / self._smooth_point +
                (1 - smooth_mask) * (iou - 0.5 * self._smooth_point)).mean()

        return loss
