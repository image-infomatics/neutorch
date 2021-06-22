import random
import os
from time import time
from tqdm import tqdm

import click
import numpy as np

import cv2
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from neutorch.model.RSUNet import UNetModel
from neutorch.model.io import save_chkpt, log_image, log_affinity_output, log_weights, load_chkpt
from neutorch.model.loss import BinomialCrossEntropyWithLogits
from neutorch.dataset.affinity import Dataset


@click.command()
@click.option('--path',
              type=str, default='./data',
              help='path to the training data'
              )
@click.option('--seed',
              type=int, default=7,
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
@click.option('--start_example', '-s',
              type=int, default=0,
              help='which example we are starting from if loading from checkpoint, does not affect num_examples.'
              )
@click.option('--num_workers', '-w',
              type=int, default=0,
              help='num workers for pytorch dataloader.'
              )
@click.option('--num_examples', '-e',
              type=int, default=500000,
              help='how many training examples the network will see before completion.'
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
              type=int, default=100, help='training interval in terms of examples seen to record data points.'
              )
@click.option('--validation-interval', '-v',
              type=int, default=1000, help='validation interval in terms of examples seen to record validation data.'
              )
@click.option('--checkpoint-interval', '-ch',
              type=int, default=50000, help='interval when to log checkpoints.'
              )
@click.option('--load',
              type=str, default='', help='load from checkpoint, pass path to ckpt file'
              )
@click.option('--verbose',
              type=bool, default=False, help='whether to print messages.'
              )
@click.option('--logstd',
              type=bool, default=False, help='whether to redirect stdout to a logfile.'
              )
def train(path: str, seed: int, patch_size: str, batch_size: int,
          start_example: int,  num_examples: int, num_workers: int, output_dir: str,
          in_channels: int, out_channels: int, learning_rate: float,
          training_interval: int, validation_interval: int, checkpoint_interval: int,
          load: str, verbose: bool, logstd: bool):

    cv2.setNumThreads(0)
    # redirect stdout to logfile
    if logstd:
        old_stdout = sys.stdout
        log_path = os.path.join(output_dir, 'message.log')
        log_file = open(log_path, "w")
        sys.stdout = log_file

    if verbose:
        print("init...")

    # clear in case was stopped before
    tqdm._instances.clear()

    # set up
    patch_size = eval(patch_size)
    random.seed(seed)
    patch_voxel_num = np.product(patch_size) * batch_size
    accumulated_loss = 0.0
    pbar = tqdm(total=num_examples)
    total_itrs = num_examples // batch_size

    # init log writers
    # m_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/model'))
    t_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/train'))
    v_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/valid'))

    # init model
    model = UNetModel(in_channels, out_channels)
    # make parallel
    model = nn.DataParallel(model)
    # load chkpt
    if load != '':
        model = load_chkpt(model, load)

    pin_memory = False
    if torch.cuda.is_available():
        model = model.cuda()
        # fine tune convolutions, see: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        torch.backends.cudnn.benchmark = True
        pin_memory = True

    # init optimizer, loss, dataset, dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_module = BinomialCrossEntropyWithLogits()
    dataset = Dataset(path, patch_size=patch_size, length=num_examples)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    if verbose:
        print("gpu: ", torch.cuda.is_available())
        print("starting... total_itrs", total_itrs)

    for step, batch in enumerate(dataloader):

        # get batch
        image, target = batch

        # foward pass
        logits = model(image)

        # compute loss
        loss = loss_module(logits, target)

        # better zero gradients
        # see: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        for param in model.parameters():
            param.grad = None
        # loss backward
        loss.backward()
        optimizer.step()

        # record loss
        cur_loss = loss.cpu().tolist()
        accumulated_loss += cur_loss

        # record progress
        pbar.set_postfix({'cur_loss': round(cur_loss / patch_voxel_num, 3)})
        pbar.update(batch_size)

        # the current number of examples the network has seen
        example_number = ((step+1) * batch_size)+start_example

        # log for training
        if example_number % training_interval == 0 and example_number > 0:
            # compute loss
            per_voxel_loss = accumulated_loss / training_interval / patch_voxel_num
            print(f'training loss {round(per_voxel_loss, 3)}')

            # compute predict
            predict = torch.sigmoid(logits)

            # log values
            t_writer.add_scalar('Loss', per_voxel_loss, example_number)
            log_affinity_output(t_writer, 'train/target',
                                target, example_number)
            log_affinity_output(t_writer, 'train/predict',
                                predict, example_number)
            log_image(t_writer, 'train/image', image, example_number)

            # reset loss
            accumulated_loss = 0.0

        # log for validation
        if example_number % validation_interval == 0 and example_number > 0:

            # get validation_batch
            batch = dataset.random_validation_batch
            validation_image = torch.from_numpy(batch.images)
            validation_target = torch.from_numpy(batch.targets)

            # transfer Data to GPU if available
            if torch.cuda.is_available():
                validation_image = validation_image.cuda()
                validation_target = validation_target.cuda()

            # pass with validation example
            with torch.no_grad():
                # compute loss
                validation_logits = model(validation_image)
                validation_predict = torch.sigmoid(validation_logits)
                validation_loss = loss_module(
                    validation_logits, validation_target)
                per_voxel_loss = validation_loss.cpu().tolist() / patch_voxel_num
                print(f'validation loss: {round(per_voxel_loss, 3)}')

                # log values
                v_writer.add_scalar('Loss', per_voxel_loss, example_number)
                log_affinity_output(v_writer, 'validation/prediction',
                                    validation_predict, example_number,)
                log_affinity_output(v_writer, 'validation/target',
                                    validation_target, example_number)
                log_image(v_writer, 'validation/image',
                          validation_image, example_number)

        # save checkpoint
        if example_number % checkpoint_interval == 0 and example_number > 0 or step == total_itrs-1:
            save_chkpt(model, output_dir, example_number, optimizer)

        # # log weights for every 10 validation
        # if iter_idx % (validation_interval*10) == 0 and iter_idx > 0:
        #     log_weights(m_writer, model, iter_idx)

    # close all
    t_writer.close()
    v_writer.close()
    # m_writer.close()
    pbar.close()
    if logstd:
        sys.stdout = old_stdout
        log_file.close()


if __name__ == '__main__':
    train()
