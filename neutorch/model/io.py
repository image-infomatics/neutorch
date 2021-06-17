import matplotlib.cm
import matplotlib
import os
import math

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def colorize(tensor):
    tensor.unsqueeze_(0)
    tesnor = tensor.repeat(3, 1, 1)
    return tesnor


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
               iter_idx: int, nrow: int = 8, depth: int = 0, batch_index: int = 0):
    """write a 5D tensor in tensorboard log

    Args:
        writer (SummaryWriter):
        tag (str): the name of the tensor in the log
        tensor (torch.Tensor):
        iter_idx (int): training iteration index
        nrow (int): number of images in a row
        depth (int): if 0 will log whole volume on grid, otherwise will log up 0:depth
        batch_index (int): index of batch to select example to log
    """
    assert torch.is_tensor(tensor)
    assert tensor.ndim >= 3
    # normalize from 0 to 1
    tensor -= tensor.min()
    tensor /= tensor.max()

    # select example from batch
    tensor = tensor[batch_index]

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


def log_image(writer: SummaryWriter, tag: str, tensor: torch.Tensor,
              iter_idx: int, batch_index: int = 0,  slice_index: int = 0):
    """write a input iimage tensor in tensorboard log

    Args:
        writer (SummaryWriter):
        tag (str): the name of the tensor in the log
        tensor (torch.Tensor):
        iter_idx (int): training iteration index
        slice_index (int): index of slice to select example to log
        batch_index (int): index of batch to select example to log
    """
    assert torch.is_tensor(tensor)
    assert tensor.ndim >= 3
    tensor = tensor.cpu()
    # normalize from 0 to 1
    tensor -= tensor.min()
    tensor /= tensor.max()

    # select slice from batch
    slice = tensor[batch_index, :, slice_index, :, :]
    writer.add_image(f'{tag}_image', slice, iter_idx)


def log_affinity_output(writer: SummaryWriter, tag: str, tensor: torch.Tensor,
                        iter_idx: int, batch_index: int = 0,  slice_index: int = 0):
    """write a Affinity Output tensor in tensorboard log

    Args:
        writer (SummaryWriter):
        tag (str): the name of the tensor in the log
        tensor (torch.Tensor):
        iter_idx (int): training iteration index
        slice_index (int): index of slice to select example to log
        batch_index (int): index of batch to select example to log
    """
    assert torch.is_tensor(tensor)
    assert tensor.ndim >= 3
    tensor = tensor.cpu()
    # normalize from 0 to 1
    tensor -= tensor.min()
    tensor /= tensor.max()
    h = tensor.shape[-1]
    w = tensor.shape[-2]

    # select slice from batch
    slice = tensor[batch_index, :, slice_index, :, :]

    # def log_channels(channels, subtag):
    #     imgs = [torch.squeeze(tensor[..., slice(channels[0], channels[1]), z, :, :], axis=0)
    #             for z in range(depth_index)]
    #     img = make_grid(imgs, padding=0, nrow=nrow)
    #     writer.add_image(f'{tag}_{subtag}_{channels}', img, iter_idx)

    # log affinity map, channels [0,3)
    a_maps = torch.reshape(slice[0:3, :, :], (3, 1, w, h))
    grid = make_grid(a_maps, padding=0, nrow=2)
    writer.add_image(f'{tag}_affinity', grid, iter_idx)

    # log lsd channels, channels [3,12)
    last_channels = colorize(slice[12, :, :])
    lsd_slice = torch.cat([slice[3:12, :, :], last_channels])
    lsd_maps = torch.reshape(lsd_slice, (4, 3, w, h))
    grid = make_grid(lsd_maps, padding=0, nrow=2)
    writer.add_image(f'{tag}_lsd', grid, iter_idx)


def log_weights(writer: SummaryWriter, model, iter_idx):

    # get arrays of up and down
    down_convs = model.module.core.dconvs
    up_convs = model.module.core.uconvs

    for i, dc in enumerate(down_convs):
        pre = dc[1].pre.conv.weight
        post = dc[1].post.conv.weight

        writer.add_histogram(f'down_conv/pre_{i}', pre, iter_idx)
        writer.add_histogram(f'down_conv/post_{i}', post, iter_idx)

    for i, uc in enumerate(up_convs):
        pre = uc.conv.pre.conv.weight
        post = uc.conv.post.conv.weight

        writer.add_histogram(f'up_conv/pre_{i}', pre, iter_idx)
        writer.add_histogram(f'up_conv/post_{i}', post, iter_idx)
