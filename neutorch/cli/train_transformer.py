import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from neutorch.cremi.evaluate import do_agglomeration, cremi_metrics
from neutorch.dataset.affinity import Dataset
from neutorch.model.loss import BinomialCrossEntropyWithLogits
from neutorch.model.io import save_chkpt, log_image, log_affinity_output, load_chkpt, log_segmentation
from neutorch.model.swin_transformer3D import SwinUNet3D
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import sys
import numpy as np
import click
from tqdm import tqdm
from time import time
import os
import random


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
              type=click.Path(),
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
              type=bool, default=False, help='whether to train with mutlitask lsd'
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
@click.option('--aug',
              type=bool, default=True, help='whether to use data augmentation.'
              )
@click.option('--ddp',
              type=bool, default=True, help='whether to use distrubited data parallel vs normal data parallel.'
              )
def train_wrapper(*args, **kwargs):
    if kwargs['ddp']:
        world_size = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train_parallel, nprocs=world_size, args=(world_size, kwargs,))
    else:
        train(**kwargs)


def train_parallel(rank, world_size, kwargs):
    dist.init_process_group("nccl", init_method='env://',
                            rank=rank, world_size=world_size)

    kwargs['world_size'] = world_size
    kwargs['rank'] = rank
    train(**kwargs)


def train(path: str, seed: int, patch_size: str, batch_size: int,
          start_example: int,  num_examples: int, num_workers: int, output_dir: str,
          in_channels: int, out_channels: int, learning_rate: float,
          training_interval: int, validation_interval: int, checkpoint_interval: int,
          lsd: bool, load: str, verbose: bool, logstd: bool, aug: bool,
          ddp: bool, rank: int = 0, world_size: int = 1):

    # only print root process
    if verbose:
        print(f"init process rank {rank}")
    if rank != 0:
        verbose = False

    if rank == 0:
        # make output folder if doesnt exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # clear in case was stopped before
        tqdm._instances.clear()

    # set up
    random.seed(seed)
    t_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/train'))
    v_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/valid'))
    patch_size = eval(patch_size)
    patch_voxel_num = np.product(patch_size) * batch_size
    accumulated_loss = 0.0
    pbar = tqdm(total=num_examples)
    # total number of iterations for training
    total_itrs = num_examples // batch_size
    training_iters = 0  # number of iterations that happened since last training_interval
    dataset = Dataset(path, patch_size=patch_size,
                      length=num_examples, lsd=lsd, batch_size=batch_size, aug=aug)

    # compute automatically
    if out_channels == 0:
        if lsd:
            out_channels = 13
        else:
            out_channels = 3

    # init model
    model = SwinUNet3D(in_channels, out_channels)
    if load != '':
        model = load_chkpt(model, load)

    # handle GPU and parallelism
    pin_memory = False
    sampler = None
    if torch.cuda.is_available():
        # gpu with DistributedDataParallel
        if ddp:
            model = model.to(rank)
            DistributedDataParallel(model, device_ids=[rank])
            sampler = DistributedSampler(
                dataset, world_size, rank, seed=seed)
        # gpu with DataParallel
        else:
            model = model.cuda()
            model = nn.DataParallel(model)

        # any gpu use
        torch.backends.cudnn.benchmark = True
        pin_memory = True

    # init optimizer, loss, dataset, dataloader
    loss_module = BinomialCrossEntropyWithLogits()
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler, drop_last=True)

    if verbose:
        print("gpu: ", torch.cuda.is_available())
        print("starting... total_itrs", total_itrs)

    for step, batch in enumerate(dataloader):

        # get batch
        image, target = batch

        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

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
        cur_loss = loss.item()
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

            # log values
            t_writer.add_scalar('Loss', per_voxel_loss, example_number)
            log_affinity_output(t_writer, 'train/target',
                                target, example_number)
            log_affinity_output(t_writer, 'train/predict',
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

                # log values
                v_writer.add_scalar('Loss', per_voxel_loss, example_number)
                log_affinity_output(v_writer, 'validation/prediction',
                                    validation_predict, example_number,)
                log_affinity_output(v_writer, 'validation/target',
                                    validation_target, example_number)
                log_image(v_writer, 'validation/image',
                          validation_image, example_number)

                ping = time()
                # do aggolmoration and metrics
                metrics = {'voi_split': 0, 'voi_merge': 0,
                           'adapted_rand': 0, 'cremi_score': 0}

                # only compute over first in batch for time saving
                i = 0
                # get true segmentation and affinity map
                segmentation_truth = np.squeeze(batch.labels[i])
                affinity = validation_predict[i][0:3].cpu().numpy()

                # get predicted segmentation from affinity map
                segmentation_pred = do_agglomeration(affinity)

                # get the CREMI metrics from true segmentation vs predicted segmentation
                metric = cremi_metrics(
                    segmentation_pred, segmentation_truth)
                for m in metric.keys():
                    metrics[m] += metric[m]/batch_size

                # log the picture for first in batch
                if i == 0:
                    log_segmentation(v_writer, 'validation/seg_true',
                                     segmentation_truth, example_number)
                    log_segmentation(v_writer, 'validation/seg_pred',
                                     segmentation_pred, example_number)

                # log metrics
                for k, v in metrics.items():
                    v_writer.add_scalar(
                        f'cremi_metrics/{k}', v, example_number)

                print(f'aggo+metrics takes {round(time()-ping, 3)} seconds.')

        # save checkpoint
        if example_number // checkpoint_interval > prev_example_number // checkpoint_interval or step == total_itrs-1 and rank == 0:
            save_chkpt(model, output_dir, example_number, optimizer)

    # close all
    if rank == 0:
        t_writer.close()
        v_writer.close()
        # m_writer.close()
        pbar.close()


if __name__ == '__main__':
    train_wrapper()
