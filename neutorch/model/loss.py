import torch
from torch import nn
import numpy as np


def gunpowder_balance(target: torch.Tensor, mask: torch.Tensor = None, thresh: float = 0.):
    if not torch.any(target):
        return None

    if mask is not None:
        bmsk = (mask > 0)
        nmsk = bmsk.sum().item()
        assert nmsk > 0
    else:
        bmsk = torch.ones_like(target, dtype=torch.uint8)
        nmsk = np.prod(bmsk.size())

    lpos = (torch.gt(target, thresh) * bmsk).type(torch.float)
    lneg = (torch.le(target, thresh) * bmsk).type(torch.float)

    npos = lpos.sum().item()

    fpos = np.clip(npos / nmsk, 0.05, 0.95)
    fneg = (1.0 - fpos)

    wpos = 1. / (2. * fpos)
    wneg = 1. / (2. * fneg)

    return (lpos * wpos + lneg * wneg).type(torch.float32)


class BinomialCrossEntropyWithLogits(nn.Module):
    """
    A version of BCE w/ logits with the ability to mask
    out regions of output.
    """

    def __init__(self, rebalance: bool = True):
        super().__init__()
        self.rebalance = rebalance
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _reduce_loss(self, loss: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            cost = loss.sum()  # / np.prod(loss.size())
        else:
            cost = (loss * mask).sum()  # / mask.sum()
        return cost

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask=None):
        loss = self.bce(pred, target)

        if mask is not None:
            rebalance_weight = gunpowder_balance(target, mask=mask)
            loss *= rebalance_weight

        cost = self._reduce_loss(loss, mask=mask)
        return cost + 0.001


class FocalLoss(BinomialCrossEntropyWithLogits):
    def __init__(self, alpha: float = 0.25, gamma: float = 2., rebalance: bool = True):
        """reweight the loss to focus more on the inaccurate rear spots

        Args:
            alpha (float, optional): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. . Defaults to 0.25.
            gamma (float, optional): Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples. Defaults to 2.
            rebalance (bool, optional): rebalance the positive and negative voxels. Defaults to True.
        """
        super().__init__(rebalance=rebalance)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        implementation was partially copied from here.
        https://github.com/pytorch/vision/blob/master/torchvision/ops/focal_loss.py
        Note that the license is BSD 3-Clause License
        """
        loss = self.bce(pred, target)

        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        loss = loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1. - self.alpha) * (1. - target)
            loss = alpha_t * loss

        if mask is not None:
            rebalance_weight = gunpowder_balance(target, mask=mask)
            loss *= rebalance_weight

        cost = self._reduce_loss(loss, mask=mask)
        return cost

# TO-DO
# tversky loss
# https://gitlab.mpcdf.mpg.de/connectomics/codat/-/blob/master/codat/training/losses.py


# https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
class MultiTaskLoss(nn.Module):

    def __init__(self, number_of_tasks):
        super(MultiTaskLoss, self).__init__()
        self.number_of_tasks = number_of_tasks
        self.log_vars = nn.Parameter(torch.zeros((number_of_tasks)))

    def forward(self, preds, gts):

        crossEntropy = BinomialCrossEntropyWithLogits()
        mse = nn.MSELoss()

        loss_aff = crossEntropy(preds[0], gts[0])
        precision0 = torch.exp(-self.log_vars[0])
        loss_aff = precision0*loss_aff + self.log_vars[0]

        loss_LSD = mse(preds[1], gts[1])
        precision1 = torch.exp(-self.log_vars[1])
        loss_LSD = precision1*loss_LSD + self.log_vars[1]

        return loss_aff + loss_LSD


# class WeightedAffLSD_MSELoss(torch.nn.MSELoss):

#     def __init__(self):
#         super(WeightedAffLSD_MSELoss, self).__init__()

#     def forward(self, lsds_prediction, lsds_target, lsds_weights, affs_prediction, affs_target, affs_weights,):

#         loss1 = super(WeightedAffLSD_MSELoss, self).forward(
#                 lsds_prediction*lsds_weights,
#                 lsds_target*lsds_weights)

#         loss2 = super(WeightedAffLSD_MSELoss, self).forward(
#             affs_prediction*affs_weights,
#             affs_target*affs_weights)

#         return loss1 + loss2
