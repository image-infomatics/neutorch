import random

import click


import torch
from torch import nn
import torchio as tio
from .model.IsoRSUNet import Model
from .loss import BinomialCrossEntropyWithLogits
from .dataset.tbar import Dataset


@click.command()
@click.option('--seed', 
    type=int, default=1,
    help='for reproducibility'
)
@click.option('--training-split-ratio', '-s',
    type=float, default=0.8,
    help='use 80% of samples for training, 20% of samples for validation.'
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
@click.option('--in-channels', '-c', 
    type=int, default=1, help='channel number of input tensor.'
)
@click.option('--out-channels', '-n',
    type=int, default=1, help='channel number of output tensor.')
@click.option('--learning-rate', '-l',
    type=float, default=0.001, help='learning rate'
)
def train(seed: int, training_split_ratio: float, patch_size: tuple,
        iter_start: int, iter_stop: int, gpus: tuple, output_dir: str,
        in_channels: int, out_channels: int, learning_rate: float):
    
    random.seed(seed)

    model = Model(1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_module = BinomialCrossEntropyWithLogits()
    dataset = Dataset(
        "~/Dropbox (Simons Foundation)/40_gt/tbar.toml",
        num_workers=2,
        sampling_distance=4,
    )

    for iter_idx in range(iter_start, iter_stop):
        patch = next(iter(dataset.random_patches))
        image = patch['image'][tio.DATA]
        image /= 255.
        target = patch['tbar'][tio.DATA]
        logits = model(image)
        loss = loss_module.forward(logits, target)
        optimizer.zero_grad()
        loss.backward()
    

if __name__ == '__main__':
    train()