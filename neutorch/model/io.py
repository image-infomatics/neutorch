import matplotlib.cm
import matplotlib
import os
import math
import numpy as np
from skimage import color
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
    fpath = os.path.join(fpath, 'chkpts')

    # make folder if doesnt exist
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    fname = os.path.join(fpath, f'model_{chkpt_num}.chkpt')
    state = {'iter': chkpt_num,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, fname)


def load_chkpt(model: nn.Module, path: str):
    print(f"LOAD CHECKPOINT: {path} iters.")
    checkpoint = torch.load(path)
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
                        iter_idx: int, batch_index: int = 0,  slice_index: int = 0, lsd=False):
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
    d = tensor.shape[-3]
    c = tensor.shape[-4]

    # look halfway in
    if slice_index == 0:
        slice_index = d // 2

    # select slice from batch
    slice = tensor[batch_index, :, slice_index, :, :]

    # log affinity map, channels [0,3)
    a_maps = torch.reshape(slice[0:3, :, :], (3, 1, w, h))
    grid = make_grid(a_maps, padding=0, nrow=2)
    writer.add_image(f'{tag}_affinity', grid, iter_idx)

    # log lsd channels, channels [3,12)
    if lsd:
        last_channels = colorize(slice[12, :, :])
        lsd_slice = torch.cat([slice[3:12, :, :], last_channels])
        lsd_maps = torch.reshape(lsd_slice, (4, 3, w, h))
        grid = make_grid(lsd_maps, padding=0, nrow=2)
        writer.add_image(f'{tag}_lsd', grid, iter_idx)


def log_segmentation(writer: SummaryWriter, tag: str, seg: np.ndarray,
                     iter_idx: int,  slice_index: int = 0):
    """write a input iimage tensor in tensorboard log

    Args:
        writer (SummaryWriter):
        tag (str): the name of the tensor in the log
        seg (np.ndarray): the segmentation
        iter_idx (int): training iteration index
        slice_index (int): index of slice to select example to log
    """

    d = seg.shape[-3]
    # look halfway in
    if slice_index == 0:
        slice_index = d // 2

    # pick slice from volume
    slice = seg[slice_index, :, :]

    # color slice
    dummy_vol = np.zeros(slice.shape)
    colored = color.label2rgb(slice, dummy_vol, alpha=1, bg_label=0)
    colored = torch.Tensor(colored)
    writer.add_image(f'{tag}_image', colored, iter_idx, dataformats='HWC')


def label_data(vol, seg):
    length = vol.shape[0]
    size = vol.shape[1]
    # reshape for labeling
    seg = np.reshape(seg, (size, length*size))
    vol = np.reshape(vol, (size, length*size))
    # label
    labeled = color.label2rgb(seg, vol, alpha=0.1, bg_label=-1)
    # shape back
    labeled = np.reshape(labeled, (length, size, size, 3))


def reassemble_img_from_cords(cords, img_arr):
    cords = cords.astype(int)
    z_cords, y_cords, x_cords = cords[0, ...], cords[1, ...], cords[2, ...]
    x_min, x_max = np.amin(x_cords), np.amax(x_cords)
    y_min, y_max = np.amin(y_cords), np.amax(y_cords)
    z_min, z_max = np.amin(z_cords), np.amax(z_cords)
    (sz, sy, sx) = img_arr[0][0].shape
    new_image = np.zeros(
        (z_max-z_min+sz+1, y_max-y_min+sy+1, x_max-x_min+sx+1))
    for j in range(img_arr.shape[0]):
        patch = img_arr[j][0]
        st = cords[j, :, 0, 0, 0].astype(int)
        bz, by, bx = st[0]-z_min, st[1]-y_min, st[2]-x_min
        new_image[bz:bz+sz, by:by+sy, bx:bx+sx] = patch

    return new_image
