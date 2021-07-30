import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from neutorch.cremi.evaluate import write_output_data
from neutorch.dataset.affinity import Dataset
from neutorch.model.config import *
from neutorch.model.io import save_chkpt, log_image, log_affinity_output, load_chkpt, log_segmentation
from torch.utils.data.dataloader import DataLoader
from neutorch.cli.test_model import test_model
import torch.cuda.amp as amp
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
@click.option('--output-dir', '-o',
              type=str, default='/mnt/home/jberman/ceph/runs',
              help='for output'
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
@click.option('--validation-interval', '-v',
              type=int, default=5000, help='validation interval in terms of examples seen to record validation data.'
              )
@click.option('--test-interval', '-ti',
              type=int, default=50000, help='interval when to run full test.'
              )
@click.option('--final-test', '-ft',
              type=bool, default=True, help='weather to run a final test using best performing checkpoint.'
              )
@click.option('--load',
              type=str, default='', help='load from checkpoint, pass path to ckpt file'
              )
@click.option('--verbose',
              type=bool, default=False, help='whether to print messages.'
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


def train(config: str, path: str, seed: int, output_dir: str, batch_size: int, sync_every: int,
          start_example: int,  num_workers: int,
          training_interval: int, validation_interval: int, test_interval: int, final_test: bool,
          load: str, verbose: bool,
          use_amp: bool, ddp: bool, fup: bool, rank: int = 0, world_size: int = 1):

    # get config
    config = get_config(config)

    # unpack config
    num_examples = config.dataset.num_examples
    patch_size = config.dataset.patch_size

    use_gpu = torch.cuda.is_available()
    gpus = torch.cuda.device_count()
    cpus = len(os.sched_getaffinity(0))  # gets machine cpus

    # auto set
    if num_workers == -1:
        if ddp:
            num_workers = cpus//world_size
        elif not use_gpu:
            num_workers = 1
        else:
            num_workers = cpus

    agg_threshold = 0.7
    sync_every = sync_every // batch_size

    # only print root process
    if rank != 0:
        verbose = False

    output_dir = f'{output_dir}/{config.name}_run'

    if rank == 0:

        # rm dir if exists then create
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # write config
        f = open(f"{output_dir}/config.txt", "w")
        f.write(config.toString())
        f.write(
            f'TRAINING\nseed: {seed}, batch_size: {batch_size}, sync_every: {sync_every}, use_gpu: {use_gpu}, total_cpus: {cpus}, total_gpus: {gpus}, use_amp: {use_amp}, ddp: {ddp}, world_size: {world_size}, num_workers: {num_workers}\n')
        f.close()

        # clear in case was stopped before
        tqdm._instances.clear()
        t_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/train'))
        v_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/valid'))
        pbar = tqdm(total=num_examples)

    # set up
    random.seed(seed)
    accumulated_loss = 0.0
    total_itrs = num_examples // batch_size
    dataset = build_dataset_from_config(config.dataset, path)

    patch_volume = np.product(patch_size)
    steps_since_training_interval = 0

    # metrics to keep track of
    best_avg_cremi_score = 9999
    best_example_ckpt = 0

    # init model
    model = build_model_from_config(config.model)
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
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler, drop_last=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, total_epochs, eta_min=0, last_epoch=-1, verbose=False)
    scaler = amp.GradScaler(enabled=use_amp)
    if verbose:
        print(
            f'total_itrs: {total_itrs}')
        print("starting...")

    for step, batch in enumerate(dataloader):

        # determien when to sync gradients in ddp
        if ddp:
            if step % sync_every == 0:
                model.require_backward_grad_sync = True
            else:
                model.require_backward_grad_sync = False

        # with sync_policy:
        # get batch
        image, target = batch

        # Transfer Data to GPU if available
        if use_gpu:
            image = image.cuda(rank, non_blocking=True)
            target = target.cuda(rank, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            # foward pass
            logits = model(image)

            # compute loss
            loss = loss_module(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()  # set_to_none=True here can modestly improve performance

        # record loss
        cur_loss = loss.item()
        accumulated_loss += cur_loss

        # record progress
        if rank == 0:
            pbar.set_postfix(
                {'cur_loss': round(cur_loss / patch_volume, 3)})
            pbar.update(batch_size * world_size)

        # the previous number of examples the network has seen
        prev_example_number = ((step) * batch_size*world_size)+start_example

        # the current number of examples the network has seen
        example_number = ((step+1) * batch_size*world_size)+start_example

        steps_since_training_interval += 1

        # all io and validation done root process
        if rank == 0:

            # log for training
            if example_number // training_interval > prev_example_number // training_interval:

                # compute loss
                per_voxel_loss = accumulated_loss / patch_volume / \
                    steps_since_training_interval / batch_size

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
                steps_since_training_interval = 0

            # log for validation
            if example_number // validation_interval > prev_example_number // validation_interval:

                # get validation_batch
                batch = dataset.random_validation_batch(batch_size)
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

                    per_voxel_loss = validation_loss.item() / patch_volume / batch_size

                    # log values
                    v_writer.add_scalar(
                        'Loss', per_voxel_loss, example_number)
                    log_affinity_output(v_writer, 'validation/prediction',
                                        validation_predict, example_number,)
                    log_affinity_output(v_writer, 'validation/target',
                                        validation_target, example_number)
                    log_image(v_writer, 'validation/image',
                              validation_image, example_number)

            # test checkpoint
            if example_number // test_interval > prev_example_number // test_interval:

                save_chkpt(model, output_dir, example_number, optimizer)
                sum_cremi_score = 0
                files = ['sample_A_pad', 'sample_B_pad', 'sample_C_pad']

                for file in files:
                    res = test_model(model, patch_size, f'./data/{file}.hdf', pre_crop=(10, 400, 400), threshold=agg_threshold,
                                     border_width=config.dataset.border_width,)
                    affinity, segmentation, metrics = res['affinity'], res['segmentation'], res['metrics']

                    # convert to torch, add batch dim
                    affinity = torch.unsqueeze(torch.tensor(affinity), 0)

                    # log
                    log_affinity_output(v_writer, f'test/full_affinity_{file}',
                                        affinity, example_number)
                    log_segmentation(v_writer, f'test/full_segmentation_{file}',
                                     segmentation, example_number)
                    cremi_score = metrics['cremi_score']
                    sum_cremi_score += cremi_score
                    v_writer.add_scalar(
                        f'cremi_metrics/full_cremi_score_{file}', cremi_score, example_number)

                avg_cremi_score = sum_cremi_score / len(files)

                if avg_cremi_score < best_avg_cremi_score:
                    best_avg_cremi_score = avg_cremi_score
                    best_example_ckpt = example_number

                v_writer.add_scalar(
                    f'cremi_metrics/avg_cremi_score', avg_cremi_score, example_number)

    if final_test:

        file = None
        # sleep to avoid race
        if rank == 0:
            file = 'sample_A_pad'
            time.sleep(5)
        elif rank == 1:
            file = 'sample_B_pad'
            time.sleep(10)
        elif rank == 2:
            file = 'sample_C_pad'
            time.sleep(15)

        if file is not None:
            model = load_chkpt(
                model, f'{output_dir}/chkpts/model_{best_example_ckpt}.chkpt')
            # run test
            res = test_model(model,
                             patch_size,
                             f'{path}/{file}.hdf',
                             pre_crop=None,
                             threshold=agg_threshold,
                             border_width=config.dataset.border_width,
                             full_agglomerate=True)

            affinity, segmentation, metrics = res['affinity'], res['segmentation'], res['metrics']

            write_output_data(affinity, segmentation, metrics, config_name=config.name, example_number=example_number, file=file,
                              output_dir=f'/mnt/home/jberman/ceph')

    if rank == 0:
        t_writer.close()
        v_writer.close()
        pbar.close()


if __name__ == '__main__':
    train_wrapper()
