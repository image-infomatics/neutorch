import random
import os
from time import time

import click
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from neutorch.model.IsoRSUNet import UNetModel
from neutorch.model.io import save_chkpt, log_tensor, log_affinity_output
from neutorch.model.loss import BinomialCrossEntropyWithLogits
from neutorch.dataset.affinity import Dataset


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
              type=int, default=200000,
              help='the stopping index of training iteration.'
              )
@click.option('--output-dir', '-o',
              type=click.Path(file_okay=False, dir_okay=True,
                              writable=True, resolve_path=True),
              required=True,
              default='./output',
              help='the directory to save all the outputs, such as checkpoints.'
              )
@click.option('--in-channels', '-c',
              type=int, default=1, help='channel number of input tensor.'
              )
@click.option('--out-channels', '-n',
              type=int, default=13, help='channel number of output tensor.')
@click.option('--learning-rate', '-l',
              type=float, default=0.001, help='learning rate'
              )
@click.option('--training-interval', '-t',
              type=int, default=100, help='training interval to record stuffs.'
              )
@click.option('--validation-interval', '-v',
              type=int, default=1000, help='validation and saving interval iterations.'
              )
def train(seed: int, training_split_ratio: float, patch_size: tuple,
          iter_start: int, iter_stop: int, output_dir: str,
          in_channels: int, out_channels: int, learning_rate: float,
          training_interval: int, validation_interval: int):

    random.seed(seed)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))

    model = UNetModel(in_channels, out_channels)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_module = BinomialCrossEntropyWithLogits()
    dataset = Dataset(
        patch_size=patch_size,
        training_split_ratio=training_split_ratio
    )

    patch_voxel_num = np.product(patch_size)
    accumulated_loss = 0.
    for iter_idx in range(iter_start, iter_stop):

        patch = dataset.random_training_patch

        image = torch.from_numpy(np.array([patch.image]))
        target = torch.from_numpy(np.array([patch.target]))

        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        logits = model(image)
        loss = loss_module(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accumulated_loss += loss.cpu().tolist()

        log_depth = 4
        if iter_idx % training_interval == 0 and iter_idx > 0:
            per_voxel_loss = accumulated_loss / training_interval / patch_voxel_num
            print(f'training loss {round(per_voxel_loss, 3)}')
            accumulated_loss = 0.
            predict = torch.sigmoid(logits)
            writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)
            log_affinity_output(writer, 'train/target',
                                target, iter_idx, depth=log_depth)
            log_affinity_output(writer, 'train/predict',
                                predict, iter_idx, depth=log_depth)
            log_tensor(writer, 'train/image', image, iter_idx, depth=log_depth)

        if iter_idx % validation_interval == 0 and iter_idx > 0:
            fname = os.path.join(output_dir, f'model_{iter_idx}.chkpt')
            print(f'save model to {fname}')
            save_chkpt(model, output_dir, iter_idx, optimizer)

            print('evaluate prediction: ')
            patch = dataset.random_validation_patch
            print('evaluation patch shape: ', patch.shape)
            validation_image = torch.from_numpy(patch.image)
            validation_target = torch.from_numpy(patch.label)
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                validation_image = validation_image.cuda()
                validation_target = validation_target.cuda()

            with torch.no_grad():
                validation_logits = UNetModel(validation_image)
                validation_predict = torch.sigmoid(validation_logits)
                validation_loss = loss_module(
                    validation_logits, validation_target)
                per_voxel_loss = validation_loss.cpu().tolist() / patch_voxel_num
                print(
                    f'iter {iter_idx}: validation loss: {round(per_voxel_loss, 3)}')
                writer.add_scalar('Loss/validation', per_voxel_loss, iter_idx)
                log_affinity_output(writer, 'evaluate/prediction',
                                    validation_predict, iter_idx, depth=log_depth)
                log_affinity_output(writer, 'evaluate/target',
                                    validation_target, iter_idx, depth=log_depth)
                log_tensor(writer, 'evaluate/image',
                           validation_image, iter_idx, depth=log_depth)

    writer.close()


if __name__ == '__main__':
    train()
