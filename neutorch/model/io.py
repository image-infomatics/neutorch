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
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def log_tensor(writer: SummaryWriter, tag: str, tensor: torch.Tensor,
               iter_idx: int, nrow: int = 8, depth: int = 0):
    """write a 5D tensor in tensorboard log

    Args:
        writer (SummaryWriter):
        tag (str): the name of the tensor in the log
        tensor (torch.Tensor):
        iter_idx (int): training iteration index
        nrow (int): number of images in a row
        depth (int): if 0 will log whole volume on grid, otherwise will log up 0:depth
    """
    assert torch.is_tensor(tensor)
    assert tensor.ndim >= 3
    # normalize from 0 to 1
    tensor -= tensor.min()
    tensor /= tensor.max()

    # this should work for ndim >= 3
    tensor = tensor.cpu()
    # tensor = (tensor * 255.).type(torch.uint8)
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    depth_index = tensor.shape[-3]

    if depth > 0:
        depth_index = depth
        nrow = math.ceil(math.sqrt(depth_index))

    ncol = math.ceil(depth / nrow)
    img = torch.zeros((height*ncol, width*nrow))

    for z in range(depth_index):
        row = math.floor(z / nrow)
        col = z % nrow
        # print(col)
        img[row*height: (row+1)*height, col*width: (col+1) *
            width] = torch.squeeze(tensor[..., z, :, :])
    writer.add_image(tag, img, iter_idx, dataformats='HW')


def log_affinity_output(writer: SummaryWriter, tag: str, tensor: torch.Tensor,
                        iter_idx: int, depth: int = 0):
    """write a Affinity Output tensor in tensorboard log

    Args:
        writer (SummaryWriter):
        tag (str): the name of the tensor in the log
        tensor (torch.Tensor):
        iter_idx (int): training iteration index
        depth (int): if 0 will log whole volume on grid, otherwise will log up 0:depth
    """
    assert torch.is_tensor(tensor)
    assert tensor.ndim >= 3
    # normalize from 0 to 1
    tensor -= tensor.min()
    tensor /= tensor.max()

    # this should work for ndim >= 3
    tensor = tensor.cpu()
    depth_index = tensor.shape[-3]
    if depth > 0:
        depth_index = depth

    nrow = math.ceil(math.sqrt(depth_index))

    def log_channels(channels, subtag):
        imgs = [torch.squeeze(tensor[..., slice(channels[0], channels[1]), z, :, :], axis=0)
                for z in range(depth_index)]
        img = make_grid(imgs, padding=0, nrow=nrow)
        writer.add_image(f'{tag}_{subtag}_{channels}', img, iter_idx)

    # log affinity channels
    log_channels((0, 1), 'affinity_x')
    log_channels((1, 2), 'affinity_y')
    log_channels((2, 3), 'affinity_z')

    # log lsd channels
    log_channels((0, 3), 'lsd')
    log_channels((3, 6), 'lsd')
    log_channels((6, 9), 'lsd')
    log_channels((9, 10), 'lsd')
