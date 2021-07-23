import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from neutorch.dataset.utils import from_h5
from neutorch.cremi.evaluate import write_output_data
from neutorch.dataset.proofread import ProofreadDataset
from neutorch.model.config import *
from neutorch.model.io import save_chkpt, log_image, log_affinity_output, load_chkpt, reassemble_img_from_cords
from torch.utils.data.dataloader import DataLoader
from neutorch.cli.test_model import test_model
import torch.cuda.amp as amp
from neutorch.model.ViT import ViT

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import numpy as np
import click
from tqdm import tqdm
import time
import os
import random
import shutil


@click.command()
@click.option('--config',
              type=str,
              help='name of the configuration defined in the config list'
              )
@click.option('--path',
              type=str, default='./data',
              help='path to the training data'
              )
@click.option('--seed',
              type=int, default=7,
              help='for reproducibility'
              )
@click.option('--batch-size', '-b',
              type=int, default=1,
              help='size of mini-batch.'
              )
@click.option('--sync-every', '-y',
              type=int, default=32,
              help='after how many iters to sync gradients across gpus'
              )
@click.option('--start_example', '-s',
              type=int, default=0,
              help='which example we are starting from if loading from checkpoint, does not affect num_examples.'
              )
@click.option('--num_workers', '-w',
              type=int, default=-1,
              help='num workers for pytorch dataloader. -1 means automatically set.'
              )
@click.option('--training-interval', '-t',
              type=int, default=2000, help='training interval in terms of examples seen to record data points.'
              )
@click.option('--load',
              type=str, default='', help='load from checkpoint, pass path to ckpt file'
              )
@click.option('--use-amp',
              type=bool, default=True, help='whether to use distrubited automatic mixed percision.'
              )
@click.option('--ddp',
              type=bool, default=True, help='whether to use distrubited data parallel vs normal data parallel.'
              )
@click.option('--fup', default=False, help='find unused parameters.'
              )
def train_wrapper(*args, **kwargs):
    if kwargs['ddp']:
        world_size = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        print(f'starting ddp world size {world_size}')
        mp.spawn(train_parallel, nprocs=world_size, args=(world_size, kwargs,))
    else:
        train(**kwargs)


def train_parallel(rank, world_size, kwargs):
    dist.init_process_group("nccl", init_method='env://',
                            rank=rank, world_size=world_size)

    kwargs['world_size'] = world_size
    kwargs['rank'] = rank

    train(**kwargs)


def train(config: str, path: str, seed: int, batch_size: int, sync_every: int,
          start_example: int,  num_workers: int,
          training_interval: int,
          load: str,
          use_amp: bool, ddp: bool, fup: bool, rank: int = 0, world_size: int = 1):

    # get config
    config = get_config(config)

    # unpack config
    epochs = 100
    patch_size = config.dataset.patch_size

    print('loading data...')

    file = 'sample_A'
    image = from_h5(
        f'./data/{file}.hdf', dataset_path='volumes/raw')
    true = from_h5(
        f'./data/{file}.hdf', dataset_path='volumes/labels/neuron_ids')
    pred = from_h5(
        f'./RSUnet_900000_run/seg_{file}_pad.hdf', dataset_path='volumes/labels/neuron_ids')
    aff = None
    dataset = ProofreadDataset(
        image, pred, true, aff, patch_size=patch_size, sort=False, shuffle=True)

    use_gpu = torch.cuda.is_available()
    gpus = torch.cuda.device_count()
    if hasattr(os, 'sched_getaffinity'):
        cpus = len(os.sched_getaffinity(0))  # gets machine cpus
    else:
        cpus = 1

    # auto set
    if num_workers == -1:
        if ddp:
            num_workers = cpus//world_size
        elif not use_gpu:
            num_workers = 1
        else:
            num_workers = cpus

    sync_every = sync_every

    output_dir = f'./{config.name}_run'

    if rank == 0:

        # rm dir if exists then create
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # write config
        f = open(f"{output_dir}/config.txt", "w")
        f.write(config.toString())
        f.write(
            f'TRAINING\nseed: {seed}, sync_every: {sync_every}, use_gpu: {use_gpu}, total_cpus: {cpus}, total_gpus: {gpus}, use_amp: {use_amp}, ddp: {ddp}, world_size: {world_size}, num_workers: {num_workers}\n')
        f.close()

        # clear in case was stopped before
        tqdm._instances.clear()
        t_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/train'))
        v_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/valid'))
        pbar = tqdm(total=len(dataset))

    # set up
    random.seed(seed)
    accumulated_loss = 0.0
    patches_seen = 0
    steps_since_training_interval = 0

    # init model
    model = ViT(
        patch_size=patch_size,
        in_channels=1,
        out_channels=3,
        patch_emb_dim=196,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    loss_module = build_loss_from_config(config.loss)
    if load != '':
        model = load_chkpt(model, load)

    # handle GPU and parallelism
    pin_memory = False
    sampler = None
    if use_gpu:
        # gpu with DistributedDataParallel
        if ddp:
            model = model.cuda(rank)
            model = DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=fup)
            sampler = DistributedSampler(
                dataset, world_size, rank, seed=seed)
            loss_module.cuda(rank)
        # gpu with DataParallel
        else:
            model = model.cuda()
            loss_module.cuda()
            model = nn.DataParallel(model)

        # any gpu use
        torch.backends.cudnn.benchmark = True
        pin_memory = True

    # init optimizer, lr_scheduler, loss, dataloader, scaler
    params = model.parameters()
    optimizer = build_optimizer_from_config(config.optimizer, params)
    # dataloader = DataLoader(
    #     dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler, drop_last=False, shuffle=True)

    scaler = amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        for step, sample in enumerate(dataset):

            # determien when to sync gradients in ddp
            if ddp:
                if step % sync_every == 0:
                    model.require_backward_grad_sync = True
                else:
                    model.require_backward_grad_sync = False

            (image, cords, true) = sample

            img_batch = np.expand_dims(image, axis=0)  # add batch dim
            cords_batch = np.expand_dims(cords, axis=0)  # add batch dim
            true_batch = np.expand_dims(true, axis=0)  # add batch dim
            img_batch = torch.from_numpy(img_batch)
            cords_batch = torch.from_numpy(cords_batch)
            true_batch = torch.from_numpy(true_batch)

            num_patches = image.shape[1]
            patches_seen += num_patches

            # Transfer Data to GPU if available
            if use_gpu:
                image = image.cuda(rank, non_blocking=True)
                target = target.cuda(rank, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # foward pass
                logits = model(img_batch)
                # compute loss
                loss = loss_module(logits, true_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # set_to_none=True here can modestly improve performance

            # record loss
            cur_loss = loss.item()
            accumulated_loss += cur_loss

            # record progress
            if rank == 0:
                pbar.update(world_size)

            # the previous number of examples the network has seen
            prev_example_number = ((step) * world_size) + \
                start_example+epoch*len(dataset)

            # the current number of examples the network has seen
            example_number = ((step+1) * world_size) + \
                start_example+epoch*len(dataset)

            steps_since_training_interval += 1

            # all io and validation done root process
            if rank == 0:

                # log for training
                if example_number // training_interval > prev_example_number // training_interval:

                    # compute loss
                    adjusted_loss = accumulated_loss / patches_seen / \
                        steps_since_training_interval

                    # compute predict
                    predict = torch.sigmoid(logits)

                    # log values
                    t_writer.add_scalar('Loss', adjusted_loss, example_number)

                    cords = cords_batch[0]
                    # reassemble rectangular image from array
                    image_img = reassemble_img_from_cords(cords, img_batch[0])
                    true_img = reassemble_img_from_cords(cords, true_batch[0])
                    predict_img = reassemble_img_from_cords(cords, predict[0])

                    log_affinity_output(t_writer, 'train/target',
                                        true_img, example_number)
                    log_affinity_output(t_writer, 'train/predict',
                                        predict_img, example_number)
                    log_image(t_writer, 'train/image',
                              image_img, example_number)

                    # reset acc loss
                    accumulated_loss = 0.0
                    steps_since_training_interval = 0

    if rank == 0:
        t_writer.close()
        v_writer.close()
        pbar.close()


if __name__ == '__main__':
    train_wrapper()
