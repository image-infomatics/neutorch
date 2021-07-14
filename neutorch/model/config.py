import torch
from neutorch.model.swin_transformer3D import SwinUNet3D
from neutorch.model.RSUNet import UNetModel
from neutorch.model.FunUnet import FunUnet

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
                 io_kernel = (1, 5, 5),
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
                 num_examples=600000,
                 patch_size=(26, 256, 256),
                 lsd=False,
                 aug=True,
                 border_width=2,
                 ):

        self.name = name
        self.model = dotdict({
            'model': model,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'io_kernel':io_kernel,
            'swin_patch_size': swin_patch_size,
            'embed_dim': embed_dim,
            'depths': depths,
            'res_conns': res_conns,
            'num_heads': num_heads,
            'window_size':  window_size,
            'upsampler': upsampler
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
            'num_examples': num_examples,
            'patch_size': patch_size,
            'lsd': lsd,
            'aug': aug,
            'border_width': border_width
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
    elif model == 'FunUnet':
        return FunUnet(config.in_channels, config.out_channels)
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


def get_config(name):
    for c in CONFIGS:
        if c.name == name:
            return c
    raise ValueError(f'config {name} not found.')


c0 = TransformerConfig('swin', model='swin')
c1 = TransformerConfig('swin_bi', model='swin', upsampler='bilinear')
c2 = TransformerConfig('swin_bigwin', model='swin', window_size=(7,7,7))
c3 = TransformerConfig('RSUnet2', model='RSUnet',
                       learning_rate=0.001)
CONFIGS = [c0, c1, c2, c3]
