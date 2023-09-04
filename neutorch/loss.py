import numpy as np
import torch
from torch import nn

def gunpowder_balance(target: torch.Tensor, mask: torch.Tensor=None, thresh: float=0.5):
    
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

    def _reduce_loss(self, loss: torch.Tensor, mask: torch.Tensor=None):
        if mask is None:
            cost = loss.sum() #/ np.prod(loss.size())
        else:
            cost = (loss * mask).sum() #/ mask.sum()
        return cost

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask=None):
     
        loss = self.bce(pred, target)
        if self.rebalance: 
            rebalance_weight = gunpowder_balance(target, mask)
            loss *= rebalance_weight
        cost = self._reduce_loss(loss, mask=mask)
        return cost

class MeanSquareErrorLoss(nn.Module):
    """
    A version of MeanSquareLoss with the ability to mask out regions of output
    """
    def __init__(self, rebalance: bool = True):
        super().__init__() 
        self.rebalance = rebalance
        self.mse = nn.MSELoss(reduction="none")

    def _reduce_loss(self, loss: torch.Tensor, mask: torch.Tensor=None):
        
        if mask is None:
            cost = loss.sum() #/ np.prod(loss.size())
        else:
            cost = (loss * mask).sum() #/ mask.sum()
        return cost

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask=None):

        loss = self.mse(pred, target)

    if self.rebalance:
            rebalance_weight = gunpowder_balance(target, mask=mask)
            loss *= rebalance_weight

        cost = self._reduce_loss(loss, mask=mask)
        return cost


class FocalLoss(BinomialCrossEntropyWithLogits):
    def __init__(self, alpha: float = 0.25, gamma: float=2., rebalance: bool=True):
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
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
        
        if self.rebalance:
            rebalance_weight = gunpowder_balance(target, mask=mask)
            loss *= rebalance_weight
   
        cost = self._reduce_loss(loss, mask=mask)
        return cost



# TO-DO
# tversky loss
# https://gitlab.mpcdf.mpg.de/connectomics/codat/-/blob/master/codat/training/losses.py
=======
import numpy as np
import torch
from torch import nn

def gunpowder_balance(target: torch.Tensor, mask: torch.Tensor=None, thresh: float=0.5):
    
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

    def _reduce_loss(self, loss: torch.Tensor, mask: torch.Tensor=None):
        if mask is None:
            cost = loss.sum() #/ np.prod(loss.size())
        else:
            cost = (loss * mask).sum() #/ mask.sum()
        return cost

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask=None):
     
        loss = self.bce(pred, target)
        if self.rebalance: 
            rebalance_weight = gunpowder_balance(target, mask)
            loss *= rebalance_weight
        cost = self._reduce_loss(loss, mask=mask)
        return cost

        if self.rebalance:
            rebalance_weight = gunpowder_balance(target, mask=mask)
            loss *= rebalance_weight

        cost = self._reduce_loss(loss, mask=mask)
        return cost


class FocalLoss(BinomialCrossEntropyWithLogits):
    def __init__(self, alpha: float = 0.25, gamma: float=2., rebalance: bool=True):
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
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
        
        if self.rebalance:
            rebalance_weight = gunpowder_balance(target, mask=mask)
            loss *= rebalance_weight
   
        cost = self._reduce_loss(loss, mask=mask)
        return cost

# TO-DO
# tversky loss
# https://gitlab.mpcdf.mpg.de/connectomics/codat/-/blob/master/codat/training/losses.py

