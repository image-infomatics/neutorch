from __future__ import print_function
from itertools import repeat
import collections
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import GPUtil

__all__ = ['Model']


WIDTH = [16, 32, 64, 128, 256, 512]
GPUS = [0, 1, 2, 2, 3]


def _ntuple(n):
    """
    Copied from the PyTorch source code (https://github.com/pytorch).
    """
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_triple = _ntuple(3)


def pad_size(kernel_size, mode):
    assert mode in ['valid', 'same', 'full']
    ks = _triple(kernel_size)
    if mode == 'valid':
        return _triple(0)
    elif mode == 'same':
        assert all([x % 2 for x in ks])
        return tuple(x // 2 for x in ks)
    elif mode == 'full':
        return tuple(x - 1 for x in ks)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 bias=False):
        super(Conv, self).__init__()
        padding = pad_size(kernel_size, 'same')
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class BilinearUp(nn.Module):
    """Caffe style bilinear upsampling.
    Currently everything's hardcoded and only supports upsampling factor of 2.
    """

    def __init__(self, in_channels, out_channels):
        super(BilinearUp, self).__init__()
        assert in_channels == out_channels
        self.groups = in_channels
        self.init_weights()

    def forward(self, x):
        return F.conv_transpose3d(x, self.weight,
                                  stride=(1, 2, 2), padding=(0, 1, 1), groups=self.groups
                                  )

    def init_weights(self):
        weight = torch.Tensor(self.groups, 1, 1, 4, 4)
        width = weight.size(-1)
        hight = weight.size(-2)
        assert width == hight
        f = float(math.ceil(width / 2.0))
        c = float(width - 1) / (2.0 * f)
        for w in range(width):
            for h in range(hight):
                weight[..., h, w] = (1 - abs(w/f - c)) * (1 - abs(h/f - c))
        self.register_buffer('weight', weight)


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    padding = pad_size(kernel_size, 'same')
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BNReLUConv, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', conv(in_channels, out_channels,
                                     kernel_size=kernel_size))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = BNReLUConv(channels, channels)
        self.conv2 = BNReLUConv(channels, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.add_module('pre',  BNReLUConv(in_channels, out_channels))
        self.add_module('res',  ResBlock(out_channels))
        self.add_module('post', BNReLUConv(out_channels, out_channels))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1, 2, 2)):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=up, mode='trilinear'),
            BilinearUp(in_channels, in_channels),
            conv(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, skip):
        return self.up(x) + skip


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1, 2, 2)):
        super(UpConvBlock, self).__init__()
        self.up = UpBlock(in_channels, out_channels, up=up)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x, skip)
        x = self.conv(x)
        return x


class RSUNet(nn.Module):
    def __init__(self, width=WIDTH, split_gpus=False):
        super(RSUNet, self).__init__()
        assert len(width) > 1
        depth = len(width) - 1
        self.depth = depth
        self.in_channels = width[0]
        self.split_gpus = split_gpus

        self.iconv = ConvBlock(width[0], width[0])
        if self.split_gpus:
            self.iconv.cuda(0)

        self.dconvs = nn.ModuleList()
        for d in range(depth):
            module = nn.Sequential(nn.MaxPool3d((1, 2, 2)),
                                   ConvBlock(width[d], width[d+1]))
            if self.split_gpus:
                module = module.cuda(GPUS[d])

            self.dconvs.append(module)

        self.uconvs = nn.ModuleList()
        for d in reversed(range(depth)):

            module = UpConvBlock(width[d+1], width[d])

            if self.split_gpus:
                module.cuda(GPUS[depth - d - 1])

            self.uconvs.append(module)

        self.out_channels = width[0]

        self.init_weights()

    def forward(self, x):
        x.cuda(0)
        x = self.iconv(x)

        skip = list()
        for d, dconv in enumerate(self.dconvs):
            if self.split_gpus:
                x = x.cuda(GPUS[d])
            skip.append(x)
            x = dconv(x)

        for d, uconv in enumerate(self.uconvs):
            res = skip.pop()
            if self.split_gpus:
                x = x.cuda(GPUS[d])
                res = res.cuda(GPUS[d])
            x = uconv(x, res)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = Conv(in_channels, out_channels, kernel_size, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x.cuda(1)
        return x


class UNetModel(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """

    def __init__(self, in_channels: int, out_channels: int, width=WIDTH, io_kernel=(1, 5, 5), split_gpus=False):
        super().__init__()

        self.add_module('in', InputBlock(in_channels, width[0], io_kernel))
        self.add_module('core', RSUNet(width=width, split_gpus=split_gpus))
        self.add_module('out', OutputBlock(width[0], out_channels, io_kernel))

        if split_gpus:
            self.get_submodule('in').cuda(0)
            self.get_submodule('out').cuda(3)
