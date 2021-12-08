import random
import os
from time import time

import click
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from neutorch.dataset.patch import collate_batch

from neutorch.model.IsoRSUNet import Model
from neutorch.model.io import save_chkpt, load_chkpt, log_tensor
from neutorch.loss import BinomialCrossEntropyWithLogits
from neutorch.dataset.post_synapses import Dataset, worker_init_fn



@click.command()
@click.option('--seed', 
    type=int, default=1,
    help='for reproducibility'
)
@click.option('--patch-size', '-p',
    type=int, nargs=3, default=(256, 256, 256),
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
@click.option('--training-interval', '-t',
    type=int, default=1000, help='training interval to record stuffs.'
)
@click.option('--validation-interval', '-v',
    type=int, default=10000, help='validation and saving interval iterations.'
)
@click.option('--num-workers', '-p',
    type=int, default=2, help='number of processes for data loading.'
)
def train(seed: int,  patch_size: tuple,
        iter_start: int, iter_stop: int, dataset_config_file: str, 
        output_dir: str,
        in_channels: int, out_channels: int, learning_rate: float,
        training_interval: int, validation_interval: int, num_workers: int):
    
    random.seed(seed)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))

    model = Model(in_channels, out_channels)
    model = load_chkpt(model, output_dir, iter_start)
    
    batch_size = 1
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print("Let's use ", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # we use a batch for each GPU
            batch_size = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
     
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_module = BinomialCrossEntropyWithLogits()
    training_dataset = Dataset(
        dataset_config_file,
        section_name='training',
        patch_size=patch_size,
    )
    validation_dataset = Dataset(
        config_file=dataset_config_file,
        section_name="validation",
        patch_size=patch_size,
    )

    training_data_loader = DataLoader(
        training_dataset,
        num_workers=num_workers,
        prefetch_factor=2,
        drop_last=False,
        multiprocessing_context='spawn',
        collate_fn=collate_batch,
        worker_init_fn=worker_init_fn,
        batch_size=batch_size,
    )
    
    validation_data_loader = DataLoader(
        validation_dataset,
        num_workers=1,
        prefetch_factor=1,
        drop_last=False,
        multiprocessing_context='spawn',
        collate_fn=collate_batch,
        batch_size=batch_size,
    )
    validation_data_iter = iter(validation_data_loader)

    patch_voxel_num = np.product(patch_size)
    accumulated_loss = 0.
    iter_idx = iter_start
    for image, target in training_data_loader:
        iter_idx += 1
        if iter_idx> iter_stop:
            print('exceeds the maximum iteration: ', iter_stop)
            return

        ping = time()
        print(f'preparing patch takes {round(time()-ping, 3)} seconds')
        logits = model(image)
        loss = loss_module(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accumulated_loss += loss.tolist()
        print(f'iteration {iter_idx} takes {round(time()-ping, 3)} seconds.')

        if iter_idx % training_interval == 0 and iter_idx > 0:
            per_voxel_loss = accumulated_loss / training_interval / patch_voxel_num
            print(f'training loss {round(per_voxel_loss, 3)}')
            accumulated_loss = 0.
            predict = torch.sigmoid(logits)
            writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)
            log_tensor(writer, 'train/image', image, iter_idx)
            log_tensor(writer, 'train/prediction', predict, iter_idx)
            log_tensor(writer, 'train/target', target, iter_idx)

        if iter_idx % validation_interval == 0 and iter_idx > 0:
            fname = os.path.join(output_dir, f'model_{iter_idx}.chkpt')
            print(f'save model to {fname}')
            save_chkpt(model, output_dir, iter_idx, optimizer)

            print('evaluate prediction: ')
            validation_image, validation_target = next(validation_data_iter)

            with torch.no_grad():
                validation_logits = model(validation_image)
                validation_predict = torch.sigmoid(validation_logits)
                validation_loss = loss_module(validation_logits, validation_target)
                per_voxel_loss = validation_loss.tolist() / patch_voxel_num
                print(f'iter {iter_idx}: validation loss: {round(per_voxel_loss, 3)}')
                writer.add_scalar('Loss/validation', per_voxel_loss, iter_idx)
                log_tensor(writer, 'evaluate/image', validation_image, iter_idx)
                log_tensor(writer, 'evaluate/prediction', validation_predict, iter_idx)
                log_tensor(writer, 'evaluate/target', validation_target, iter_idx)

    writer.close()


if __name__ == '__main__':
    train()
