import math
from itertools import repeat
from collections.abc import Iterable

from torch import nn
import torch


WIDTH = [16, 32, 64, 128, 256, 512]

def _ntuple(n: int):
    """
    Copied from PyTorch source code
    """
    def parse(x):
        if isinstance(x, Iterable):
            return x
        else:
            return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)


def pad_size(kernel_size: int, mode: str):
    ks = _triple(kernel_size)
    if mode == 'valid':
        return _triple(0)
    elif mode == 'same':
        assert all([x % 2 for x in ks])
        return tuple(x // 2 for x in ks)
    elif mode == 'full':
        return tuple(x - 1 for x in ks)
    else:
        raise ValueError('invalide mode option, only support valid, same or full')

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
            kernel_size: int = 3, stride: int = 1, bias: bool = False):
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


class TrilinearUp(nn.Module):
    """Caffe style trilinear upsampling.

    Currently everything's hardcoded and only supports upsampling factor of 2.
    Note that this implementation is isotropic compared with BilinearUp.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.groups = channels
        self._init_weights()
    
    def _init_weights(self):
        weight = torch.Tensor(self.groups, 1, 4, 4, 4)
        width = weight.size(-1)
        hight = weight.size(-2)
        depth = weight.size(-3)
        assert width == hight
        f = float(math.ceil(width / 2.0))
        c = float(width - 1) / (2.0 * f)
        for w in range(width):
            for h in range(hight):
                for d in range(depth):
                    weight[...,d,h,w] = (1 - abs(w/f - c)) * (1 - abs(h/f - c)) * (1 - abs(d/f - c))
        self.register_buffer('weight', weight)
    
    def forward(self, x):
        x = nn.functional.conv_transpose3d(x, self.weight,
            stride=(2,2,2), padding=(1,1,1), groups=self.groups
        )
        return x


def conv(in_channels, out_channels, kernel_size: int = 3, stride: int = 1,
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
    def __init__(self, channels):
        super().__init__()
        self.conv1 = BNReLUConv(channels, channels)
        self.conv2 = BNReLUConv(channels, channels)

    def forward(self, x):
        residule = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residule
        return x

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.add_module('pre', BNReLUConv(in_channels, out_channels))
        self.add_module('res', ResBlock(out_channels))
        self.add_module('post', BNReLUConv(out_channels, out_channels))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            TrilinearUp(in_channels),
            conv(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x += skip
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = UpBlock(in_channels, out_channels)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x, skip)
        x = self.conv(x)
        return x


class IsoRSUNet(nn.Module):
    """Isotropic Residual Symmetric UNet
    
    Isotropic: the convolutional kernel size is isotropic, normally 3x3x3
    Residual: there are residual blocks in convolutional chain.
    Symmetric: the left and right side is the same
    UNet: 3D UNet
    """
    def __init__(self, width: list=WIDTH) -> None:
        super().__init__()
        assert len(width) > 1
        for w in width:
            assert w >= 1
            
        depth = len(width) - 1
        self.in_channels = width[0]
        self.input_conv = ConvBlock(width[0], width[0])
        self.down_convs = nn.ModuleList()
        for d in range(depth):
            self.down_convs.append( nn.Sequential(
                nn.MaxPool3d((2, 2, 2)),
                ConvBlock(width[d], width[d+1])
            ))
        
        self.up_convs = nn.ModuleList()
        for d in reversed(range(depth)):
            self.up_convs.append(UpConvBlock(width[d+1], width[d]))
        
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
    def __init__(self, in_channels: int, out_channels: int, width: list=WIDTH):
        super().__init__()

        # assert len(in_spec)==1, "model takes a single input"
        # in_channels = list(in_spec.values()[0][-4])
        # matches the RSUNet output
        # out_channels = width[0] 
        io_kernel = (3, 3, 3)

        self.add_module('in', InputBlock(in_channels, width[0], io_kernel))
        self.add_module('core', IsoRSUNet(width=width))
        self.add_module('out', OutputBlock(width[0], out_channels, io_kernel))


if __name__ == '__main__':
    model = Model(1, 1)
    input = torch.rand((1,1, 64, 64, 64), dtype=torch.float32)
    logits = model(input)
    assert logits.shape[1] == 1
