import random
import os
from time import time
import multiprocessing as mp

import click
import numpy as np

import torch
from torch import nn
import torchio as tio
from torch.utils.tensorboard import SummaryWriter

from neutorch.model.IsoRSUNet import Model
from neutorch.model.io import save_chkpt, log_tensor
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
@click.option('--worker-num', '-p',
    type=int, default=None, help='number of worker processes in data provider.'
)
@click.option('--training-interval', '-t',
    type=int, default=100, help='training interval to record stuffs.'
)
@click.option('--validation-interval', '-v',
    type=int, default=1000, help='validation and saving interval iterations.'
)
@click.option('--sampling-distance', '-d',
    type=int, default=32, help='sampling patch around the annotated point.'
)
def train(seed: int, training_split_ratio: float, patch_size: tuple,
        iter_start: int, iter_stop: int, output_dir: str,
        in_channels: int, out_channels: int, learning_rate: float,
        worker_num: int, training_interval: int, validation_interval: int,
        sampling_distance: int):
    
    random.seed(seed)

    if worker_num is None:
        worker_num = mp.cpu_count() // 2

    writer = SummaryWriter(log_dir=output_dir)

    model = Model(in_channels, out_channels)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_module = BinomialCrossEntropyWithLogits()
    dataset = Dataset(
        "~/Dropbox (Simons Foundation)/40_gt/tbar.toml",
        num_workers=worker_num,
        sampling_distance=sampling_distance,
        training_split_ratio=training_split_ratio,
        patch_size=patch_size,
    )

    training_patches_loader = torch.utils.data.DataLoader(
        dataset.random_training_patches,
        batch_size=1
    )
    validation_patches_loader = torch.utils.data.DataLoader(
        dataset.random_validation_patches,
        batch_size=1
    )

    patch_voxel_num = np.product(patch_size)
    ping = time()
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

        if iter_idx % training_interval == 0:
            per_voxel_loss = round((loss / patch_voxel_num).cpu().tolist(), 3)
            print(f'iter {iter_idx} in {round(time()-ping, 3)} seconds: training loss {per_voxel_loss}')
            writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)

        if iter_idx % validation_interval == 0:
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
                per_voxel_loss = round((loss / patch_voxel_num).cpu().tolist(), 3)
                print(f'iter {iter_idx}: validation loss: {per_voxel_loss}')
                writer.add_scalar('Loss/validation', per_voxel_loss, iter_idx)
                log_tensor(writer, 'image', image, iter_idx)
                log_tensor(writer, 'prediction', torch.sigmoid(logits), iter_idx)
                log_tensor(writer, 'label', target, iter_idx)

        # reset timer
        ping = time()

    writer.close()


if __name__ == '__main__':
    train()