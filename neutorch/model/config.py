import torch
from neutorch.model.swin_transformer3D import SwinUNet3D
from neutorch.model.RSUNet import UNetModel
from neutorch.dataset.affinity import Dataset
from neutorch.model.mlp_mixer import MLPMixer
from neutorch.model.mlp_mixer2 import MLPMixer2
from neutorch.model.loss import BinomialCrossEntropyWithLogits
import pprint


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TransformerConfig(object):
    def __init__(self,
                 #### Global ####
                 name,
                 #### Model ####
                 in_channels=1,
                 out_channels=3,
                 model='swin',
                 # RSUnet
                 io_kernel=(1, 5, 5),
                 # mlp
                 mlp_patch_size=(2, 4, 4),
                 depth=24,
                 expansion_factor=4,
                 token_dim=512,
                 channel_dim=4096,
                 # swin
                 swin_patch_size=(2, 4, 4),
                 embed_dim=96,
                 depths=[2, 8, 8, 2],
                 res_conns=True,
                 num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7),
                 upsampler='patch',
                 #### Optimizer ####
                 optimizer='AdamW',
                 learning_rate=0.0005,
                 betas=(0.9, 0.999),
                 weight_decay=0.05,
                 #### Loss ####
                 loss='BNCE',
                 #### Dataset ####
                 dataset='cremi',
                 num_examples=1000000,
                 patch_size=(26, 256, 256),
                 affinity_offsets=[(1, 1, 1)],
                 lsd=False,
                 aug=True,
                 border_width=2,
                 ):

        self.name = name
        self.model = dotdict({
            'model': model,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'io_kernel': io_kernel,
            'swin_patch_size': swin_patch_size,
            'embed_dim': embed_dim,
            'depths': depths,
            'res_conns': res_conns,
            'num_heads': num_heads,
            'window_size':  window_size,
            'upsampler': upsampler,
            'patch_size': patch_size,
            'mlp_patch_size': mlp_patch_size,
            'depth': depth,
            'expansion_factor': expansion_factor,
            'token_dim': token_dim,
            'channel_dim': channel_dim,
        })
        self.loss = dotdict({
            'loss': loss,
        })
        self.optimizer = dotdict({
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'betas': betas,
            'weight_decay': weight_decay
        })
        self.dataset = dotdict({
            'dataset': dataset,
            'num_examples': num_examples,
            'patch_size': patch_size,
            'lsd': lsd,
            'aug': aug,
            'border_width': border_width,
            'affinity_offsets': affinity_offsets,
        })

    def toString(self):
        d = pprint.pformat(self.dataset)
        o = pprint.pformat(self.optimizer)
        l = pprint.pformat(self.loss)
        m = pprint.pformat(self.model)
        return f'NAME\n{self.name}\nDATASET\n{d}\nOPTIMIZER\n{o}\nLOSS\n{l}\nMODEL\n{m}\n'


def build_model_from_config(config):
    model = config.model
    if model == 'swin':
        return SwinUNet3D(in_channels=config.in_channels,
                          out_channels=config.out_channels,
                          patch_size=config.swin_patch_size,
                          embed_dim=config.embed_dim,
                          depths=config.depths,
                          res_conns=config.res_conns,
                          num_heads=config.num_heads,
                          window_size=config.window_size,
                          upsampler=config.upsampler)
    elif model == 'RSUnet':
        return UNetModel(config.in_channels, config.out_channels, io_kernel=config.io_kernel)
    elif model == 'mlp':
        return MLPMixer(
            image_size=config.patch_size,
            patch_size=config.mlp_patch_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            dim=config.embed_dim,
            depth=config.depth,
            expansion_factor=config.expansion_factor,
        )
    elif model == 'mlp2':

        return MLPMixer2(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            image_size=config.patch_size,
            patch_size=config.mlp_patch_size,
            dim=config.embed_dim,
            depth=config.depth,
            token_dim=config.token_dim,
            channel_dim=config.channel_dim,
        )
    else:
        print(f'model {model} not supported. aborting.')
        return


def build_optimizer_from_config(config, params):
    if config.optimizer == 'AdamW':
        return torch.optim.AdamW(
            params, lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay)

    raise ValueError(f'optimizer {config.optimizer} not implemented yet.')


def build_loss_from_config(config):
    if config.loss == 'BNCE':
        return BinomialCrossEntropyWithLogits()
    if config.loss == 'MSE':
        return torch.nn.MSELoss()
    raise ValueError(f'loss {config.loss} not implemented yet.')


def build_dataset_from_config(config, path):
    return Dataset(path, patch_size=config.patch_size, length=config.num_examples,
                   affinity_offsets=config.affinity_offsets,
                   lsd=config.lsd,
                   aug=config.aug,
                   border_width=config.border_width)


def get_config(name):
    for c in CONFIGS:
        if c.name == name:
            return c
    raise ValueError(f'config {name} not found.')


c0 = TransformerConfig('swin', model='swin')

c3 = TransformerConfig('RSUnet', model='RSUnet',
                       learning_rate=0.001)

c5 = TransformerConfig('RSUnet2', model='RSUnet',
                       learning_rate=0.001)

c7 = TransformerConfig('mlp', model='mlp', embed_dim=1024, mlp_patch_size=(
    3, 30, 30),  patch_size=(30, 300, 300), num_examples=2000000,)
c8 = TransformerConfig('mlp2unet', model='mlp2', embed_dim=1024, mlp_patch_size=(
    4, 64, 64), patch_size=(52, 512, 512), num_examples=2000000,)
c9 = TransformerConfig('mlp2big', model='mlp2', embed_dim=1024, mlp_patch_size=(
    5, 50, 50),  patch_size=(100, 1000, 1000), num_examples=2000000,)
c10 = TransformerConfig('RSUnetBIG', model='RSUnet',
                        learning_rate=0.001, patch_size=(52, 512, 512),)

CONFIGS = [c0, c3, c5, c7, c8, c9, c10]
