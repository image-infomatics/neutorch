from typing import Tuple
import math
from itertools import repeat
from collections.abc import Iterable

from torch import nn
import torch


WIDTH = [16, 32, 64, 128, 256, 512]


def pad_size(kernel_size: Tuple[int, int, int], mode: str):
    if mode == 'valid':
        return (0, 0, 0)
    elif mode == 'same':
        assert all([x % 2 for x in kernel_size])
        return tuple(x // 2 for x in kernel_size)
    elif mode == 'full':
        return tuple(x - 1 for x in kernel_size)
    else:
        raise ValueError('invalide mode option, only support valid, same or full')

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
            kernel_size: Tuple[int, int, int], 
            stride: int = 1, bias: bool = False):
        super().__init__()
        padding = pad_size(kernel_size, 'same')
        self.conv = nn.Conv3d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


def conv(in_channels: int, out_channels: int, 
         kernel_size: Tuple[int, int, int], stride: int = 1,
         bias: bool = False):
    padding = pad_size(kernel_size, 'same')
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias)

class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('norm', nn.InstanceNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', conv(in_channels, out_channels, kernel_size=kernel_size))

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size: Tuple[int, int, int]):
        super().__init__()
        self.conv1 = BNReLUConv(channels, channels, kernel_size)
        self.conv2 = BNReLUConv(channels, channels, kernel_size)

    def forward(self, x):
        residule = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residule
        return x

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int]):
        super().__init__()
        self.add_module('pre', BNReLUConv(in_channels, out_channels, kernel_size))
        self.add_module('res', ResBlock(out_channels, kernel_size))
        self.add_module('post', BNReLUConv(out_channels, out_channels, kernel_size))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int]):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=2),
            conv(in_channels, out_channels, kernel_size),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x += skip
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int]):
        super().__init__()
        self.up = UpBlock(in_channels, out_channels, kernel_size)
        self.conv = ConvBlock(out_channels, out_channels, kernel_size)

    def forward(self, x, skip):
        x = self.up(x, skip)
        x = self.conv(x)
        return x


class RSUNet(nn.Module):
    """Residual Symmetric UNet
    
    Residual: there are residual blocks in convolutional chain.
    Symmetric: the left and right side is the same
    UNet: 3D UNet
    """
    def __init__(self, kernel_size: Tuple[int, int, int], width: list=WIDTH) -> None:
        super().__init__()
        assert len(width) > 1
        for w in width:
            assert w >= 1
            
        depth = len(width) - 1
        self.in_channels = width[0]
        self.input_conv = ConvBlock(width[0], width[0], kernel_size)
        self.down_convs = nn.ModuleList()
        for d in range(depth):
            self.down_convs.append( nn.Sequential(
                nn.MaxPool3d((2, 2, 2)),
                ConvBlock(width[d], width[d+1], kernel_size)
            ))
        
        self.up_convs = nn.ModuleList()
        for d in reversed(range(depth)):
            self.up_convs.append(UpConvBlock(width[d+1], width[d], kernel_size))
        
        self.out_channels = width[0]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.input_conv(x)
        skip = []
        for down_conv in self.down_convs:
            skip.append(x)
            x = down_conv(x)
        
        for up_conv in self.up_convs:
            x = up_conv(x, skip.pop())
        
        return x


class InputBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple):
        super().__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple):
        super().__init__()
        # self.norm = nn.BatchNorm3d(in_channels)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # spec = collections.OrderedDict(sorted(out_spec.items(), key=lambda x: x[0]))
        # outs = []
        # for k, v in spec.items():
        #     out_channels = v[-4]
        #     outs.append(
        #         Conv(in_channels, out_channels, kernel_size, bias=True)
        #     )
        # self.outs = nn.ModuleList(outs)
        # self.keys = spec.keys()
        self.conv = Conv(in_channels, out_channels, kernel_size, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        # return [out[x] for k, out in zip(self.keys, self.outs)]
        x = self.conv(x)
        return x


class Model(nn.Sequential):
    """
    Residule Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Tuple[int, int, int], width: list=WIDTH):
        super().__init__()

        # assert len(in_spec)==1, "model takes a single input"
        # in_channels = list(in_spec.values()[0][-4])
        # matches the RSUNet output
        # out_channels = width[0] 

        self.add_module('in', InputBlock(in_channels, width[0], kernel_size))
        self.add_module('core', RSUNet(kernel_size=kernel_size, width=width))
        self.add_module('out', OutputBlock(width[0], out_channels, kernel_size))


if __name__ == '__main__':
    model = Model(1, 1, kernel_size=(1, 3, 3))
    input = torch.rand((1,1, 1, 64, 64), dtype=torch.float32)
    logits = model(input)
    assert logits.shape[1] == 1
