from math import log
import random
import os
from time import time
from torch.nn.modules import loss
from tqdm import tqdm
import matplotlib.pyplot as plt

import click
import numpy as np

import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from neutorch.model.swin_transformer import SwinUNet
from neutorch.model.io import save_chkpt, log_image, log_affinity_output, log_2d_affinity_output, load_chkpt, log_segmentation
from neutorch.model.loss import BinomialCrossEntropyWithLogits
from neutorch.dataset.affinity import Dataset
from neutorch.cremi.evaluate import do_agglomeration, cremi_metrics


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
              type=int, default=0, help='channel number of output tensor. 0 means automatically computed.')
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
@click.option('--lsd',
              type=bool, default=True, help='whether to train with mutlitask lsd'
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
          lsd: bool, load: str, verbose: bool, logstd: bool):

    patch_size = '(6,64,64)'
    in_channels = 1

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
    # total number of iterations for training
    total_itrs = num_examples // batch_size
    training_iters = 0  # number of iterations that happened since last training_interval

    # init log writers
    # m_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/model'))
    t_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/train'))
    v_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/valid'))

    # compute automatically
    if out_channels == 0:
        if lsd:
            out_channels = 13
        else:
            out_channels = 3

    # init model
    model = SwinUNet(in_channels=in_channels, out_channels=2)
    # make parallel
    model = nn.DataParallel(model)
    # load chkpt
    if load != '':
        model = load_chkpt(model, load)

    pin_memory = False
    if torch.cuda.is_available():
        model = model.cuda()
        pin_memory = True

    # init optimizer, loss, dataset, dataloader
    loss_module = BinomialCrossEntropyWithLogits()
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    dataset = Dataset(path, patch_size=patch_size,
                      length=num_examples, lsd=lsd, batch_size=batch_size, )
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    if verbose:
        print("gpu: ", torch.cuda.is_available())
        print("starting... total_itrs", total_itrs)

    for step, batch in enumerate(dataloader):

        z_index = random.randint(0, patch_size[0]-1)
        # get batch
        image, target = batch

        image = image[:, :, z_index, :, :]
        target = target[:, :, z_index, :, :]
        target = target[:, 0:2, :, :]

        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        # foward pass
        logits = model(image)

        # target: B, C, ... ouput: B, ..., C
        logits = torch.moveaxis(logits, -1, 1)

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
        pbar.set_postfix(
            {'cur_loss': round(cur_loss / patch_voxel_num, 3)})
        pbar.update(batch_size)

        # the previous number of examples the network has seen
        prev_example_number = ((step) * batch_size)+start_example

        # the current number of examples the network has seen
        example_number = ((step+1) * batch_size)+start_example

        # number of iterations that happened since last training_interval
        training_iters += 1

        # log for training
        if example_number // training_interval > prev_example_number // training_interval:

            # compute loss
            per_voxel_loss = accumulated_loss / training_iters / patch_voxel_num

            # compute predict
            predict = torch.sigmoid(logits)

            image = torch.unsqueeze(image, 2)

            # log values
            t_writer.add_scalar('Loss', per_voxel_loss, example_number)
            log_2d_affinity_output(t_writer, 'train/target',
                                   target, example_number)
            log_2d_affinity_output(t_writer, 'train/predict',
                                   predict, example_number)
            log_image(t_writer, 'train/image', image, example_number)

            # reset acc loss
            accumulated_loss = 0.0
            training_iters = 0

        # log for validation
        if example_number // validation_interval > prev_example_number // validation_interval:

            # get validation_batch
            batch = dataset.random_validation_batch
            validation_image = torch.from_numpy(batch.images)
            validation_target = torch.from_numpy(batch.targets)

            validation_image = validation_image[:, :, z_index, :, :]
            validation_target = validation_target[:, :, z_index, :, :]
            validation_target = validation_target[:, 0:2, :, :]

            # transfer Data to GPU if available
            if torch.cuda.is_available():
                validation_image = validation_image.cuda()
                validation_target = validation_target.cuda()

            # pass with validation example
            with torch.no_grad():

                # compute loss
                validation_logits = model(validation_image)
                # target: B, C, ... ouput: B, ..., C
                validation_logits = torch.moveaxis(validation_logits, -1, 1)

                validation_loss = loss_module(
                    validation_logits, validation_target)
                per_voxel_loss = validation_loss.cpu().tolist() / patch_voxel_num

                validation_predict = torch.sigmoid(validation_logits)

                validation_image = torch.unsqueeze(validation_image, 2)

                # log values
                v_writer.add_scalar('Loss', per_voxel_loss, example_number)
                log_2d_affinity_output(v_writer, 'validation/prediction',
                                       validation_predict, example_number,)
                log_2d_affinity_output(v_writer, 'validation/target',
                                       validation_target, example_number)
                log_image(v_writer, 'validation/image',
                          validation_image, example_number)

                # do aggolmoration and metrics
                # metrics = {'voi_split': 0, 'voi_merge': 0,
                #            'adapted_rand': 0, 'cremi_score': 0}

                # # only compute over first in batch for time saving
                # i = 0
                # # get true segmentation and affinity map
                # segmentation_truth = np.squeeze(batch.labels[i])
                # affinity = validation_predict[i][0:3].cpu().numpy()

                # # get predicted segmentation from affinity map
                # segmentation_pred = do_agglomeration(affinity)

                # # get the CREMI metrics from true segmentation vs predicted segmentation
                # metric = cremi_metrics(
                #     segmentation_pred, segmentation_truth)
                # for m in metric.keys():
                #     metrics[m] += metric[m]/batch_size

                # # log the picture for first in batch
                # if i == 0:
                #     log_segmentation(v_writer, 'validation/seg_true',
                #                      segmentation_truth, example_number)
                #     log_segmentation(v_writer, 'validation/seg_pred',
                #                      segmentation_pred, example_number)

                # # log metrics
                # for k, v in metrics.items():
                #     v_writer.add_scalar(
                #         f'cremi_metrics/{k}', v, example_number)

        # save checkpoint
        if example_number // checkpoint_interval > prev_example_number // checkpoint_interval or step == total_itrs-1:
            save_chkpt(model, output_dir, example_number, optimizer)
            # log_weights(m_writer, model, iter_idx)

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
