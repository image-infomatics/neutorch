import os
import math

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def save_chkpt(model: nn.Module, fpath: str, chkpt_num: int, optimizer):
    """ Save trained network as file

    Args:
        model (nn.Module): current model
        fpath (str): file path of saved model
        chkpt_num (int): current iteration index
        optimizer (Optimizer): the optimizer used
    """
    print("SAVE CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model_{}.chkpt".format(chkpt_num))
    state = {'iter': chkpt_num,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, fname)

def load_chkpt(model: nn.Module, fpath: str, chkpt_num: int):
    print("LOAD CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model_{}.chkpt".format(chkpt_num))
    model.load(fname)
    return model


def log_tensor(writer: SummaryWriter, tag: str, tensor: torch.Tensor,
        iter_idx: int, nrow: int=16):
    """write a 5D tensor in tensorboard log

    Args:
        writer (SummaryWriter): 
        tag (str): the name of the tensor in the log
        tensor (torch.Tensor): 
        iter_idx (int): training iteration index
        nrow (int): number of images in a row
    """
    assert torch.is_tensor(tensor)
    assert tensor.ndim >= 3
    # this should work for ndim >= 3
    tensor = tensor.cpu()
    tensor = (tensor * 255.).type(torch.uint8)
    depth = tensor.shape[-3]
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    # make_grid only works well for 3 channel color images
    # imgs = [torch.squeeze(tensor[..., z, :, :], axis=0) for z in range(depth)]
    # img = make_grid(imgs, nrow=depth, padding=0)
    # number of images in a column
    ncol = math.ceil( depth / nrow )
    img = torch.zeros((height*ncol, width*nrow))
    # breakpoint()
    for z in range(depth):
        row = math.floor( z / nrow )
        col = z % nrow
        # print(col)
        img[row*height : (row+1)*height, col*width : (col+1)*width ] = torch.squeeze(tensor[...,z, :, :])
    # breakpoint()
    writer.add_image(tag, img, iter_idx, dataformats='HW')