import random
import os
from time import time

import click
import numpy as np

import torch
torch.multiprocessing.set_start_method('spawn')
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from neutorch.model.IsoRSUNet import Model
from neutorch.model.io import save_chkpt, log_tensor
from neutorch.loss import BinomialCrossEntropyWithLogits
from neutorch.data.volume import Dataset
from neutorch.data.patch import collate_batch



@click.command()
@click.option('--seed', 
    type=int, default=1,
    help='for reproducibility'
)
@click.option('--volume-path', '-v',
    type=str,
    required=True,
    help='Neuroglancer Precomputed volume path.'
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
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    required=True,
    help='the directory to save all the outputs, such as checkpoints.'
)
@click.option('--training-interval', '-t',
    type=int, default=100, help='training interval to record stuffs.'
)
@click.option('--validation-interval', '-v',
    type=int, default=1000, help='validation and saving interval iterations.'
)

def main(seed: int, volume_path : str,
        iter_start: int, iter_stop: int, output_dir: str,
        training_interval: int, validation_interval: int):
    
    random.seed(seed)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))

    model = Model(1, 1)
    if torch.cuda.is_available():
        model.share_memory()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss_module = BinomialCrossEntropyWithLogits()
    # loss_module = torch.nn.MSELoss()
    
    dataset = Dataset(volume_path)

    data_loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        multiprocessing_context='spawn',
        drop_last=True,
        # pin_memory=True,
        collate_fn=collate_batch,
    )

    patch_voxel_num = np.product(dataset.patch_size)
    accumulated_loss = 0.

    iter_idx = iter_start
    for image, target in data_loader:
        iter_idx += 1
        if iter_idx == iter_stop:
            print(f'reached stopping iteration number: {iter_stop}')
            return
        ping = time()
        # print('training patch shape: ', patch.shape)
        # image = torch.from_numpy(patch.image)
        # target = torch.from_numpy(patch.target)
        
        logits = model(image)
        loss = loss_module(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accumulated_loss += loss.cpu().tolist()
        print(f'iteration {iter_idx} takes {round(time()-ping, 3)} seconds.')

        if iter_idx % training_interval == 0:
            per_voxel_loss = accumulated_loss / training_interval / patch_voxel_num
            print(f'training loss {round(per_voxel_loss, 3)}')
            accumulated_loss = 0.
            predict = torch.sigmoid(logits)
            writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)
            log_tensor(writer, 'train/image', image, iter_idx)
            log_tensor(writer, 'train/prediction', predict, iter_idx)
            log_tensor(writer, 'train/target', target, iter_idx)

        if iter_idx % validation_interval == 0:
            fname = os.path.join(output_dir, f'model_{iter_idx}.chkpt')
            print(f'save model to {fname}')
            save_chkpt(model, output_dir, iter_idx, optimizer)

            print('evaluate prediction: ')
            validation_image, validation_target = dataset.random_sample
            
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

    writer.close()


if __name__ == '__main__':
    train()
