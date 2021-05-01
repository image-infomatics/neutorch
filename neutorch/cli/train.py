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
import torchvision


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
    type=int, default=200000,
    help='the stopping index of training iteration.'
)
@click.option('--dataset-config-file', '-d',
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
    required=True,
    help='dataset configuration file path.'
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
@click.option('--sampling-distance', '-m',
    type=int, default=32, help='sampling patch around the annotated point.'
)
def train(seed: int, training_split_ratio: float, patch_size: tuple,
        iter_start: int, iter_stop: int, dataset_config_file: str, 
        output_dir: str,
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
        dataset_config_file,
        num_workers=worker_num,
        sampling_distance=sampling_distance,
        training_split_ratio=training_split_ratio,
        patch_size=patch_size,
    )

    training_patches_loader = torch.utils.data.DataLoader(
        dataset.random_training_patches,
        batch_size=1,
        prefetch_factor=2,
    )
    validation_patches_loader = torch.utils.data.DataLoader(
        dataset.random_validation_patches,
        batch_size=1,
        prefetch_factor=2,
    )

    patch_voxel_num = np.product(patch_size)
    ping = time()
    accumulated_loss = 0.
    for iter_idx in range(iter_start, iter_stop):
        training_patches_batch = next(iter(training_patches_loader))
        image = training_patches_batch['image'][tio.DATA]
        target = training_patches_batch['tbar'][tio.DATA]
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()
        logits = model(image)
        loss = loss_module(logits, target)
        accumulated_loss += loss.cpu().tolist()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx % training_interval == 0:
            per_voxel_loss = accumulated_loss / training_interval / patch_voxel_num
            accumulated_loss = 0.
            print(f'iter {iter_idx} in {round(time()-ping, 3)} seconds: training loss {round(per_voxel_loss, 3)}')
            predict = torch.sigmoid(logits)
            writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)
            log_tensor(writer, 'train/image', image, iter_idx)
            log_tensor(writer, 'train/prediction', predict, iter_idx)
            log_tensor(writer, 'train/target', target, iter_idx)

            # target2d, _ =  torch.max(target, dim=2, keepdim=False)
            # slices = torch.cat((image[:, :, 32, :, :], predict[:, :, 32, :, :], target2d))
            # image_path = os.path.expanduser('~/Downloads/patches.png')
            # print('save a batch of patches to ', image_path)
            # torchvision.utils.save_image(
            #     slices,
            #     image_path,
            #     nrow=1,
            #     normalize=True,
            #     scale_each=True,
            # )

        if iter_idx % validation_interval == 0:
            fname = os.path.join(output_dir, f'model_{iter_idx}.chkpt')
            print(f'save model to {fname}')
            save_chkpt(model, output_dir, iter_idx, optimizer)

            print('evaluate prediction: ')
            validation_patches_batch = next(iter(validation_patches_loader))
            validation_image = validation_patches_batch['image'][tio.DATA]
            validation_target = validation_patches_batch['tbar'][tio.DATA]
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                validation_image = validation_image.cuda()
                validation_target = validation_target.cuda()

            with torch.no_grad():
                validation_logits = model(validation_image)
                validation_predict = torch.sigmoid(validation_logits)
                validation_loss = loss_module(validation_logits, validation_target)
                per_voxel_loss = validation_loss.cpu().tolist() / patch_voxel_num
                print(f'iter {iter_idx}: validation loss: {round(per_voxel_loss, 3)}')
                writer.add_scalar('Loss/validation', per_voxel_loss, iter_idx)
                log_tensor(writer, 'evaluate/image', validation_image, iter_idx)
                log_tensor(writer, 'evaluate/prediction', validation_predict, iter_idx)
                log_tensor(writer, 'evaluate/target', validation_target, iter_idx)

        # reset timer
        ping = time()

    writer.close()


if __name__ == '__main__':
    train()
