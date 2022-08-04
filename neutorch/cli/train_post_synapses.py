import random
import os
from time import time
from glob import glob

from yacs.config import CfgNode
import click
import numpy as np

from chunkflow.lib.bounding_boxes import Cartesian

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from neutorch.dataset.patch import collate_batch

from neutorch.model.IsoRSUNet import Model
from neutorch.model.io import save_chkpt, load_chkpt, log_tensor
from neutorch.loss import BinomialCrossEntropyWithLogits
from neutorch.dataset.post_synapses import PostSynapsesDataset, worker_init_fn



@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.'
)
def main(config_file: str):
    with open(config_file) as file:
        cfg = CfgNode.load_cfg(file)
    cfg.freeze()
    
    patch_size=Cartesian.from_collection(cfg.train.patch_size)

    # split path list
    glob_path = os.path.expanduser(cfg.dataset.glob_path)
    path_list = glob(glob_path, recursive=True)
    path_list = sorted(path_list)
    print(f'path_list \n: {path_list}')
    assert len(path_list) > 1
    assert len(path_list) % 2 == 0, \
        "the image and synapses should be paired."

    training_path_list = []
    validation_path_list = []
    for path in path_list:
        assignment_flag = False
        for validation_name in cfg.dataset.validation_names:
            if validation_name in path:
                validation_path_list.append(path)
                assignment_flag = True
        
        for test_name in cfg.dataset.test_names:
            if test_name in path:
                assignment_flag = True

        if not assignment_flag:
            training_path_list.append(path)

    print(f'split {len(path_list)} ground truth samples to {len(training_path_list)} training samples, {len(validation_path_list)} validation samples, and {len(path_list)-len(training_path_list)-len(validation_path_list)} test samples.')

    random.seed(cfg.system.seed)
    writer = SummaryWriter(log_dir=cfg.train.output_dir)

    model = Model(cfg.model.in_channels, cfg.model.out_channels)

    batch_size = 1
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_num = torch.cuda.device_count()
        print("Let's use ", gpu_num, " GPUs!")
        model = torch.nn.DataParallel(
            model, 
            device_ids=list(range(gpu_num)), 
            dim=0,
        )
        # we use a batch for each GPU
        batch_size = gpu_num
    else:
        device = torch.device("cpu")

    # note that we have to wrap the nn.DataParallel(model) before 
    # loading the model since the dictionary is changed after the wrapping 
    model = load_chkpt(model, cfg.train.output_dir, cfg.train.iter_start)
    print('send model to device: ', device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    
    loss_module = BinomialCrossEntropyWithLogits()
    training_dataset = PostSynapsesDataset(
        training_path_list,
        cfg.dataset.image_dirs,
        patch_size=patch_size,
    )
    validation_dataset = PostSynapsesDataset(
        validation_path_list,
        cfg.dataset.image_dirs,
        patch_size=patch_size,
    )
  
    training_data_loader = DataLoader(
        training_dataset,
        num_workers=cfg.system.cpus,
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
        prefetch_factor=2,
        drop_last=False,
        multiprocessing_context='spawn',
        collate_fn=collate_batch,
        batch_size=batch_size,
    )
    validation_data_iter = iter(validation_data_loader)

    voxel_num = np.product(patch_size) * batch_size
    accumulated_loss = 0.
    iter_idx = cfg.train.iter_start
    for image, target in training_data_loader:
        iter_idx += 1
        if iter_idx> cfg.train.iter_stop:
            print('exceeds the maximum iteration: ', cfg.train.iter_stop)
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

        if iter_idx % cfg.train.training_interval == 0 and iter_idx > 0:
            per_voxel_loss = accumulated_loss / cfg.train.training_interval / voxel_num
            print(f'training loss {round(per_voxel_loss, 3)}')
            accumulated_loss = 0.
            predict = torch.sigmoid(logits)
            writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)
            log_tensor(writer, 'train/image', image, iter_idx)
            log_tensor(writer, 'train/prediction', predict, iter_idx)
            log_tensor(writer, 'train/target', target, iter_idx)

        if iter_idx % cfg.train.validation_interval == 0 and iter_idx > 0:
            fname = os.path.join(cfg.train.output_dir, f'model_{iter_idx}.chkpt')
            print(f'save model to {fname}')
            save_chkpt(model, cfg.train.output_dir, iter_idx, optimizer)

            print('evaluate prediction: ')
            validation_image, validation_target = next(validation_data_iter)

            with torch.no_grad():
                validation_logits = model(validation_image)
                validation_predict = torch.sigmoid(validation_logits)
                validation_loss = loss_module(validation_logits, validation_target)
                per_voxel_loss = validation_loss.tolist() / voxel_num
                print(f'iter {iter_idx}: validation loss: {round(per_voxel_loss, 3)}')
                writer.add_scalar('Loss/validation', per_voxel_loss, iter_idx)
                log_tensor(writer, 'evaluate/image', validation_image, iter_idx)
                log_tensor(writer, 'evaluate/prediction', validation_predict, iter_idx)
                log_tensor(writer, 'evaluate/target', validation_target, iter_idx)

    writer.close()


if __name__ == '__main__':
    main()
