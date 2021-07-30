from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import numpy as np


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, patch_size, in_channels, out_channels,  dim, depth, expansion_factor=4, dropout=0.):

    patch_vol = np.product(patch_size)
    image_vol = np.product(image_size)
    (iz, iy, ix) = image_size
    (pz, py, px) = patch_size

    assert (image_size[0] % patch_size[0]
            ) == 0, 'image must be divisible by patch size in z'
    assert (image_size[1] % patch_size[1]
            ) == 0, 'image must be divisible by patch size in y'
    assert (image_size[2] % patch_size[2]
            ) == 0, 'image must be divisible by patch size in x'

    num_patches = (image_vol // patch_vol)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange('b ci (z pz) (y py) (x px) -> b (z y x) (ci pz py px)',
                  ci=in_channels, pz=pz, py=py, px=px),
        nn.Linear(patch_vol * in_channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(
                num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(
                dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        nn.Linear(dim, patch_vol * out_channels),
        Rearrange('b (z y x) (co pz py px) -> b co (z pz) (y py) (x px)',
                  co=out_channels, pz=pz, py=py, px=px, z=iz//pz, y=iy//py, x=ix//px),
    )
