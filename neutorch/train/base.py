from abc import ABC, abstractproperty
from functools import cached_property
from glob import glob

import random
import os
from time import time

from yacs.config import CfgNode
import numpy as np

from chunkflow.lib.cartesian_coordinate import Cartesian

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from neutorch.dataset.patch import collate_batch

from neutorch.model.IsoRSUNet import Model
from neutorch.model.io import save_chkpt, load_chkpt, log_tensor
from neutorch.loss import BinomialCrossEntropyWithLogits
from neutorch.dataset.base import worker_init_fn


class TrainerBase(ABC):
    def __init__(self, cfg: CfgNode, 
            batch_size: int = 1) -> None:
        if isinstance(cfg, str) and os.path.exists(cfg):
            with open(cfg) as file:
                cfg = CfgNode.load_cfg(file)
        cfg.freeze()
        
        if cfg.seed is not None:
            random.seed(cfg.seed)
        
        self.cfg = cfg
        self.batch_size = batch_size
        self.patch_size=Cartesian.from_collection(cfg.train.patch_size)

        self._split_path_list()

    @cached_property
    def path_list(self):
        glob_path = os.path.expanduser(self.cfg.dataset.glob_path)
        path_list = glob(glob_path, recursive=True)
        path_list = sorted(path_list)
        print(f'path_list \n: {path_list}')
        assert len(path_list) > 1
        assert len(path_list) % 2 == 0, \
            "the image and synapses should be paired."
        return path_list

    def _split_path_list(self):
        training_path_list = []
        validation_path_list = []
        for path in self.path_list:
            assignment_flag = False
            for validation_name in self.cfg.dataset.validation_names:
                if validation_name in path:
                    validation_path_list.append(path)
                    assignment_flag = True
            
            for test_name in self.cfg.dataset.test_names:
                if test_name in path:
                    assignment_flag = True

            if not assignment_flag:
                training_path_list.append(path)

        print(f'split {len(self.path_list)} ground truth samples to {len(training_path_list)} training samples, {len(validation_path_list)} validation samples, and {len(self.path_list)-len(training_path_list)-len(validation_path_list)} test samples.')
        self.training_path_list = training_path_list
        self.validation_path_list = validation_path_list

    @cached_property
    def model(self):
        model = Model(self.cfg.model.in_channels, self.cfg.model.out_channels)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_num = torch.cuda.device_count()
            print("Let's use ", gpu_num, " GPUs!")
            model = torch.nn.DataParallel(
                model, 
                device_ids=list(range(gpu_num)), 
                dim=0,
            )
            # we normally use one batch for each GPU
            self.batch_size *= gpu_num
        else:
            device = torch.device("cpu")

        # note that we have to wrap the nn.DataParallel(model) before 
        # loading the model since the dictionary is changed after the wrapping 
        model = load_chkpt(
            model, 
            self.cfg.train.output_dir, 
            self.cfg.train.iter_start)
        print('send model to device: ', device)
        model = model.to(device)
        return model

    @cached_property
    def optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.cfg.train.learning_rate
        )


    @cached_property
    def loss_module(self):
        return BinomialCrossEntropyWithLogits()

    @cached_property
    @abstractproperty
    def training_dataset(self):
        pass

    @cached_property
    @abstractproperty
    def validation_dataset(self):
        pass
        
    @cached_property
    def training_data_loader(self):
        training_data_loader = DataLoader(
            self.training_dataset,
            #num_workers=self.cfg.system.cpus,
            num_workers=1,
            prefetch_factor=1,
            drop_last=False,
            multiprocessing_context='spawn',
            collate_fn=collate_batch,
            worker_init_fn=worker_init_fn,
            batch_size=self.batch_size,
        )
        return training_data_loader
    
    @cached_property
    def validation_data_loader(self):
        validation_data_loader = DataLoader(
            self.validation_dataset,
            num_workers=1,
            prefetch_factor=2,
            drop_last=False,
            multiprocessing_context='spawn',
            collate_fn=collate_batch,
            batch_size=self.batch_size,
        )
        return validation_data_loader

    @cached_property
    def validation_data_iter(self):
        validation_data_iter = iter(self.validation_data_loader)
        return validation_data_iter

    @cached_property
    def voxel_num(self):
        return np.product(self.patch_size) * self.batch_size

    def __call__(self) -> None:
        writer = SummaryWriter(log_dir=self.cfg.train.output_dir)
        accumulated_loss = 0.
        iter_idx = self.cfg.train.iter_start
        for image, target in self.training_data_loader:
            iter_idx += 1
            if iter_idx> self.cfg.train.iter_stop:
                print('exceeds the maximum iteration: ', self.cfg.train.iter_stop)
                return

            ping = time()
            # print(f'preparing patch takes {round(time()-ping, 3)} seconds')
            logits = self.model(image)
            loss = self.loss_module(logits, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            accumulated_loss += loss.tolist()
            print(f'iteration {iter_idx} takes {round(time()-ping, 3)} seconds.')

            if iter_idx % self.cfg.train.training_interval == 0 and iter_idx > 0:
                per_voxel_loss = accumulated_loss / \
                    self.cfg.train.training_interval / \
                    self.voxel_num

                print(f'training loss {round(per_voxel_loss, 3)}')
                accumulated_loss = 0.
                predict = torch.sigmoid(logits)
                writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)
                log_tensor(writer, 'train/image', image, iter_idx)
                log_tensor(writer, 'train/prediction', predict, iter_idx)
                log_tensor(writer, 'train/target', target, iter_idx)

            if iter_idx % self.cfg.train.validation_interval == 0 and iter_idx > 0:
                fname = os.path.join(self.cfg.train.output_dir, f'model_{iter_idx}.chkpt')
                print(f'save model to {fname}')
                save_chkpt(self.model, self.cfg.train.output_dir, iter_idx, self.optimizer)

                print('evaluate prediction: ')
                validation_image, validation_target = next(self.validation_data_iter)

                with torch.no_grad():
                    validation_logits = self.model(validation_image)
                    validation_predict = torch.sigmoid(validation_logits)
                    validation_loss = self.loss_module(validation_logits, validation_target)
                    per_voxel_loss = validation_loss.tolist() / self.voxel_num
                    print(f'iter {iter_idx}: validation loss: {round(per_voxel_loss, 3)}')
                    writer.add_scalar('Loss/validation', per_voxel_loss, iter_idx)
                    log_tensor(writer, 'evaluate/image', validation_image, iter_idx)
                    log_tensor(writer, 'evaluate/prediction', validation_predict, iter_idx)
                    log_tensor(writer, 'evaluate/target', validation_target, iter_idx)

        writer.close()
