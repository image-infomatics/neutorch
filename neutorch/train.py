import random

import click


import torch
from torch import nn
import torchio as tio
from .model.IsoRSUNet import Model
from .loss import BinomialCrossEntropyWithLogits


@click.command()
@click.option('--seed', 
    type=int, default=1,
    help='for reproducibility'
)
@click.option('--training-split-ratio', '-s',
    type=float, default=0.9,
    help='use 90% of samples for training, 10% of samples for validation.'
)
@click.option('--patch-size', '-p',
    type=tuple, default=(64, 64, 64),
    help='input and output patch size.'
)
@click.option('--iter-start', '-b',
    type=int, default=0,
    help='the starting index of training iteration.'
)
@click.option('--iter-stop', '-e',
    type=int, default=1000000,
    help='the stopping index of training iteration.'
)
@click.option('--gpus', '-g',
    type=tuple, default=[0], multiple=True,
    help='use some GPUs. could be multiple GPUs in a single machine.'
)
@click.option('--output-dir', '-o',
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    required=True,
    help='the directory to save all the outputs, such as checkpoints.'
)
def train(seed: int, training_split_ratio: float, patch_size: tuple,
        iter_start: int, iter_stop: int, gpus: tuple, output_dir: str):
    random.seed(seed)

    in_spec = {"image": patch_size}
    out_spec = {"tbar": patch_size}
    model = Model(in_spec, out_spec)

    optimizer = torch.optim.Adam
    loss = BinomialCrossEntropyWithLogits


    

if __name__ == '__main__':
    train()