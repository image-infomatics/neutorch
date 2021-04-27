import torch
from torch import nn
import numpy as np


def gunpowder_balance(label: torch.Tensor, mask: torch.Tensor=None, thresh: float=0.):
    if not torch.any(label):
        return None

    if mask is not None:
        bmsk = (mask > 0)
        nmsk = bmsk.sum().item()
        assert nmsk > 0
    else:
        bmsk = torch.ones_like(label, dtype=torch.uint8)
        nmsk = np.prod(bmsk.size())
    
    lpos = (torch.gt(labels, thresh) * bmsk).type(torch.float)
    lneg = (torch.le(labels, thresh) * bmsk).type(torch.float)

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
    def __init__(self, rebalance: bool = False):
        super().__init__()
        self.rebalance = rebalance
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, label, mask=None):
        err = self.bce(pred, label)
        rebalance_weight = gunpowder_balance(label, mask=mask)
        if rebalance_weight is not None:
            err *= rebalance_weight

        if mask is None:
            cost = err.sum() #/ np.prod(err.size())
        else:
            cost = (err * mask).sum() #/ mask.sum()
        
        return cost