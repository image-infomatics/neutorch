import os
import math

import numpy as np


import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


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
    if os.path.exists(fname):
        print('found existing model and load: ', fname)
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('did not find existing model to load: ', fname)
    return model


def volume_to_image(tensor: torch.Tensor, 
        nrow: int=8, zstride: int = 1):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    else:
        assert torch.is_tensor(tensor)
    assert tensor.ndim >= 3
    
    if tensor.dtype == torch.uint8:
        # normalize from 0 to 1
        tensor = tensor.type(torch.float64)
        tensor -= tensor.min()
        tensor /= tensor.max()
    
    # this should work for ndim >= 3
    tensor = tensor.cpu()
    # tensor = (tensor * 255.).type(torch.uint8)
    depth = tensor.shape[-3]
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    # make_grid only works well for 4D (BxCxHxW) images
    # imgs = [torch.squeeze(tensor[..., z, :, :], axis=0) for z in range(depth)]
    # img = make_grid(imgs, nrow=depth, padding=0)
    # number of images in a column
    ncol = math.ceil( depth / nrow / zstride)
    img = torch.zeros((height*ncol, width*nrow), dtype=torch.float64)
    # breakpoint()
    for z in range(0, depth, zstride):
        row = math.floor( z / nrow / zstride )
        col = z % nrow
        # print(col)
        img[
            row*height : (row+1)*height, 
            col*width  : (col+1)*width ] = torch.squeeze(
                tensor[..., z, :, :]
        )
    return img


def log_tensor(writer: SummaryWriter, tag: str, tensor: torch.Tensor,
        iter_idx: int, nrow: int=8, zstride: int = 1, 
        mask: torch.Tensor = None):
    """write a 5D tensor in tensorboard log

    Args:
        writer (SummaryWriter): 
        tag (str): the name of the tensor in the log
        tensor (torch.Tensor): the tensor to visualize
        iter_idx (int): training iteration index
        nrow (int): number of images in a row
        zstride (int): skip a number of z sections to make the image smaller. 
            Note that zstride is not working correctly. This is a bug here.
            It only works correctly for zstride=1 for now.
        mask (Tensor): the binary mask that used to indicate the manual label
    """
    img = volume_to_image(tensor, nrow=nrow, zstride=zstride)
    if mask is not None:
        mask = volume_to_image(mask, nrow=nrow, zstride=zstride)
        assert img.size() == mask.size()
        img = torch.unsqueeze(img, 0)
        img = img.repeat(3, 1, 1)
        img[0, :,:][mask>0.5] = 1.
        img[1, :,:][mask>0.5] = 0.
        img[2, :,:][mask>0.5] = 0.
        # img = img.double()
        dataformats = 'CHW'
    else:
        # assert img.dim == 2
        dataformats = 'HW'
    # breakpoint()
    # img = img.numpy()
    writer.add_image(tag, img, iter_idx, dataformats=dataformats)