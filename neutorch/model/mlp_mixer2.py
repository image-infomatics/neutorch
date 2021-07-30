import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from neutorch.model.RSUNet import UNetModel


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixer2(nn.Module):

    def __init__(self, in_channels, out_channels, dim, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

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

        self.num_patches = (image_vol // patch_vol)

        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c z y x -> b (z y x) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(
                dim, self.num_patches, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        double_patch = (pz*2+1, py*2+1, px*2+1)
        self.patch_unembed = nn.Sequential(
            nn.Linear(dim, patch_vol * out_channels*4),
            Rearrange('b (z y x) (co pz py px) -> b co (z pz) (y py) (x px)',
                      co=out_channels*4, pz=pz, py=py, px=px, z=iz//pz, y=iy//py, x=ix//px),
            nn.Conv3d(out_channels*4, out_channels*3,
                      double_patch, 1, padding=patch_size),
            nn.Conv3d(out_channels*3, out_channels*2,
                      double_patch, 1, padding=patch_size),
            nn.Conv3d(out_channels*2, out_channels*1,
                      double_patch, 1, padding=patch_size),
        )

    def forward(self, x):

        x = self.patch_embed(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.patch_unembed(x)
        # x = self.smooth(x)
        return x


if __name__ == "__main__":
    img = torch.ones([1, 1, 30, 300, 300])

    model = MLPMixer2(1, 3, 1024, (3, 30, 30), (30, 300, 300), 24, 512, 4096)

    out_img = model(img)

    print("Shape of out :", out_img.shape)
