import random
import os

import click

import torch
from torch import nn
import torchio as tio
from neutorch.model.IsoRSUNet import Model
from neutorch.model.io import save_chkpt
from neutorch.loss import BinomialCrossEntropyWithLogits
from neutorch.dataset.tbar import Dataset


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
        iter_start: int, iter_stop: int, output_dir: str,
        in_channels: int, out_channels: int, learning_rate: float):
    
    random.seed(seed)

    model = Model(in_channels, out_channels)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_module = BinomialCrossEntropyWithLogits()
    dataset = Dataset(
        "~/Dropbox (Simons Foundation)/40_gt/tbar.toml",
        num_workers=2,
        sampling_distance=4,
        training_split_ratio=training_split_ratio,
    )

    training_patches_loader = torch.utils.data.DataLoader(
        dataset.random_training_patches,
        batch_size=1
    )
    validation_patches_loader = torch.utils.data.DataLoader(
        dataset.random_validation_patches,
        batch_size=1
    )

    for iter_idx in range(iter_start, iter_stop):
        patches_batch = next(iter(training_patches_loader))
        image = patches_batch['image'][tio.DATA]
        target = patches_batch['tbar'][tio.DATA]
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()
        logits = model(image)
        loss = loss_module(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'training loss: {loss}')

        if iter_idx % 1000 == 0:
            fname = os.path.join(output_dir, f'model_{iter_idx}.chkpt')
            print(f'save model to {fname}')
            save_chkpt(model, output_dir, iter_idx, optimizer)

            print('evaluate prediction: ')
            patches_batch = next(iter(validation_patches_loader))
            image = patches_batch['image'][tio.DATA]
            target = patches_batch['tbar'][tio.DATA]
            with torch.no_grad():
                logits = model(image)
                loss = loss_module(logits, target)
                print(f'validation loss: {loss}')





if __name__ == '__main__':
    train()