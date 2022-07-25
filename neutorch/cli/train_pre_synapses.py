import random
import os
from time import time

import click
import numpy as np
from tqdm import tqdm
import yaml

import torch
from torch.utils.tensorboard import SummaryWriter


from neutorch.model.IsoRSUNet import Model
from neutorch.model.io import save_chkpt, log_tensor
from neutorch.loss import BinomialCrossEntropyWithLogits
from neutorch.dataset.pre_synapses import Dataset

from chunkflow.lib.bounding_boxes import Cartesian


@click.command()
@click.option('--config-file', '-c', default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    help='configuration file path. yaml file format.')
def train(config_file: str):
    with open(config_file) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader) 

    seed = config['seed']
    dataset_path = config['dataset_path']
    # validation_names = config['validation_names']
    # test_names = config['test_names']
    iter_start = config['iter_start']
    iter_stop = config['iter_stop']
    output_dir = os.path.expanduser(config['output_dir'])
    training_interval = config['training_interval']
    validation_interval = config['validation_interval']
    patch_size = Cartesian.from_collection(config['patch_size'])

    random.seed(seed)
    patch_voxel_num = np.product(patch_size)

    writer = SummaryWriter(
        log_dir=output_dir
    )

    model = Model(
        config['in_channels'], 
        config['out_channels']
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate']
    )
    
    loss_module = BinomialCrossEntropyWithLogits()

    dataset = Dataset(
        dataset_path,
        patch_size = patch_size,
        validation_names = config['validation_names'],
        test_names = config['test_names'],
    )

    accumulated_loss = 0.
    
    for iter_idx in range(iter_start, iter_stop):
        ping = time()
        patch = dataset.random_training_patch
        # print('training patch shape: ', patch.shape)
        print(f'iteration {iter_idx}, preparing patch takes {round(time()-ping, 3)} seconds')
        image = torch.from_numpy(patch.image)
        target = torch.from_numpy(patch.target)
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
        print(f'iteration {iter_idx} takes {round(time()-ping, 3)} seconds.')

        if iter_idx % training_interval == 0 and iter_idx > 0:
            per_voxel_loss = accumulated_loss / training_interval / patch_voxel_num
            print(f'training loss {round(per_voxel_loss, 3)}')
            accumulated_loss = 0.
            predict = torch.sigmoid(logits)
            writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)
            log_tensor(writer, 'train/image', image, iter_idx, mask=target)
            log_tensor(writer, 'train/prediction', predict, iter_idx)
            log_tensor(writer, 'train/target', target, iter_idx)

        if iter_idx % validation_interval == 0 and iter_idx > 0:
            fname = os.path.join(output_dir, f'model_{iter_idx}.chkpt')
            print(f'save model to {fname}')
            save_chkpt(model, output_dir, iter_idx, optimizer)

            print('evaluate prediction: ')
            patch = dataset.random_validation_patch
            # print('evaluation patch shape: ', patch.shape)
            validation_image = torch.from_numpy(patch.image)
            validation_target = torch.from_numpy(patch.target)
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
                log_tensor(writer, 'validation/image', validation_image, iter_idx, mask=validation_target)
                log_tensor(writer, 'validation/prediction', validation_predict, iter_idx)
                log_tensor(writer, 'validation/target', validation_target, iter_idx)

    writer.close()


if __name__ == '__main__':
    train()
