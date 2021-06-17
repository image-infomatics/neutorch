import random
import os
from time import time
from tqdm import tqdm

import click
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from neutorch.model.RSUNet import UNetModel
from neutorch.model.io import save_chkpt, log_image, log_affinity_output, log_weights
from neutorch.model.loss import BinomialCrossEntropyWithLogits
from neutorch.dataset.affinity import Dataset


@click.command()
@click.option('--path',
              type=str, default='./data',
              help='path to the training data'
              )
@click.option('--seed',
              type=int, default=1,
              help='for reproducibility'
              )
@click.option('--patch-size', '-p',
              type=str, default='(6, 64, 64)',
              help='patch size from volume.'
              )
@click.option('--batch-size', '-b',
              type=int, default=1,
              help='size of mini-batch, generally can be 1 be should be equal to num_gpu if you want take advatnage of parallel training.'
              )
@click.option('--num_examples', '-e',
              type=int, default=200000,
              help='how many training examples the network will see before completion'
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
              type=float, default=0.001, help='the learning rate.'
              )
@click.option('--training-interval', '-t',
              type=int, default=100, help='training interval to record stuffs.'
              )
@click.option('--validation-interval', '-v',
              type=int, default=1000, help='validation and saving interval iterations.'
              )
@click.option('--verbose',
              type=bool, default=False, help='whether to print messages.'
              )
def train(path: str, seed: int, patch_size: str, batch_size: int,
          num_examples: int, output_dir: str,
          in_channels: int, out_channels: int, learning_rate: float,
          training_interval: int, validation_interval: int, verbose: bool):

    # clear in case was stopped before
    tqdm._instances.clear()

    if verbose:
        print("init...")

    patch_size = eval(patch_size)
    random.seed(seed)
    m_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/model'))
    t_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/train'))
    v_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/valid'))

    model = UNetModel(in_channels, out_channels)
    if torch.cuda.is_available():
        model = model.cuda()
    if verbose:
        print("gpu: ", torch.cuda.is_available())

    # make parallel
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_module = BinomialCrossEntropyWithLogits()
    dataset = Dataset(path, patch_size=patch_size, batch_size=batch_size)
    patch_voxel_num = np.product(patch_size) * batch_size
    accumulated_loss = 0.

    # generate a batch to make graph
    batch = dataset.random_training_batch
    image = torch.from_numpy(batch.images)
    m_writer.add_graph(model, image)

    if verbose:
        print("starting...")

    pbar = tqdm(total=num_examples)
    total_itrs = num_examples // batch_size

    if verbose:
        print("total_itrs: ", total_itrs)

    for iter_idx in range(0, total_itrs):

        if verbose:
            ping = time()
            print("gen batch...")

        batch = dataset.random_training_batch
        image = torch.from_numpy(batch.images)
        target = torch.from_numpy(batch.targets)

        if verbose:
            print(f"finish batch: {round(time()-ping, 3)}s")

        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        if verbose:
            ping = time()
            print("pass model...")

        logits = model(image)
        loss = loss_module(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_loss = loss.cpu().tolist()
        accumulated_loss += cur_loss

        if verbose:
            print(f"finish pass: {round(time()-ping, 3)}s")

        pbar.set_postfix({'cur_loss': round(cur_loss / patch_voxel_num, 3)})
        pbar.update(batch_size)

        if iter_idx % training_interval == 0 and iter_idx > 0:
            per_voxel_loss = accumulated_loss / training_interval / patch_voxel_num
            print(f'training loss {round(per_voxel_loss, 3)}')
            accumulated_loss = 0.
            predict = torch.sigmoid(logits)
            t_writer.add_scalar('Loss', per_voxel_loss, iter_idx)
            log_affinity_output(t_writer, 'train/target',
                                target, iter_idx)
            log_affinity_output(t_writer, 'train/predict',
                                predict, iter_idx)
            log_image(t_writer, 'train/image', image, iter_idx)

        if iter_idx % validation_interval == 0 and iter_idx > 0:
            fname = os.path.join(output_dir, f'model_{iter_idx}.chkpt')
            print(f'save model to {fname}')
            save_chkpt(model, output_dir, iter_idx, optimizer)

            batch = dataset.random_training_batch

            validation_image = torch.from_numpy(batch.images)
            validation_target = torch.from_numpy(batch.targets)

            # log weights,
            log_weights(m_writer, model, iter_idx)

            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                validation_image = validation_image.cuda()
                validation_target = validation_target.cuda()

            with torch.no_grad():
                validation_logits = model(validation_image)
                validation_predict = torch.sigmoid(validation_logits)
                validation_loss = loss_module(
                    validation_logits, validation_target)
                per_voxel_loss = validation_loss.cpu().tolist() / patch_voxel_num
                print(
                    f'iter {iter_idx}: validation loss: {round(per_voxel_loss, 3)}')
                v_writer.add_scalar('Loss', per_voxel_loss, iter_idx)
                log_affinity_output(v_writer, 'validation/prediction',
                                    validation_predict, iter_idx,)
                log_affinity_output(v_writer, 'validation/target',
                                    validation_target, iter_idx)
                log_image(v_writer, 'validation/image',
                          validation_image, iter_idx)

    t_writer.close()
    v_writer.close()
    m_writer.close()
    pbar.close()


if __name__ == '__main__':
    train()
