import os
import random
from abc import ABC, abstractproperty
from functools import cached_property
from time import time
from glob import glob

import numpy as np
from yacs.config import CfgNode
from chunkflow.lib.cartesian_coordinate import Cartesian

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from neutorch.data.patch import collate_batch
from neutorch.loss import BinomialCrossEntropyWithLogits, MeanSquareErrorLoss
from neutorch.model.io import load_chkpt, log_tensor, save_chkpt
from neutorch.model.IsoRSUNet import Model
from neutorch.data.dataset import worker_init_fn

def setup():
    dist.init_process_group('nccl')

def cleanup():
    dist.destroy_process_group()

class TrainerBase(ABC):
    def __init__(self, cfg: CfgNode #, 
            device: torch.DeviceObjType = None,
            local_rank: int = int(os.getenv('LOCAL_RANK', -1)),
            ) -> None:
        if isinstance(cfg, str) and os.path.exists(cfg):
            with open(cfg) as file:
                cfg = CfgNode.load_cfg(file)
        cfg.freeze()
        
        if cfg.system.seed is not None:
            random.seed(cfg.system.seed)
        
        self.cfg = cfg
        self.device = device
        self.local_rank = local_rank
        if cfg.system.gpus < 0:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = cfg.system.gpus
        self.patch_size=Cartesian.from_collection(cfg.train.patch_size)

    @cached_property
    def batch_size(self):
        # return self.num_gpus * self.cfg.train.batch_size
        # this batch size is for a single GPU rather than the total number!
        return self.cfg.train.batch_size * self.cfg.train.batch_size

    # @cached_property
    # def path_list(self):
    #     glob_path = os.path.expanduser(self.cfg.dataset.glob_path)
    #     path_list = glob(glob_path, recursive=True)
    #     path_list = sorted(path_list)
    #     print(f'path_list \n: {path_list}')
    #     assert len(path_list) > 1
        
    #     # sometimes, the path list only contains the label without corresponding image!
    #     # assert len(path_list) % 2 == 0, \
    #         # "the image and synapses should be paired."
    #     return path_list

    # def _split_path_list(self):
    #     training_path_list = []
    #     validation_path_list = []
    #     for path in self.path_list:
    #         assignment_flag = False
    #         for validation_name in self.cfg.dataset.validation_names:
    #             if validation_name in path:
    #                 validation_path_list.append(path)
    #                 assignment_flag = True
            
    #         for test_name in self.cfg.dataset.test_names:
    #             if test_name in path:
    #                 assignment_flag = True

    #         if not assignment_flag:
    #             training_path_list.append(path)

    #     print(f'split {len(self.path_list)} ground truth samples to {len(training_path_list)} training samples, {len(validation_path_list)} validation samples, and {len(self.path_list)-len(training_path_list)-len(validation_path_list)} test samples.')
    #     self.training_path_list = training_path_list
    #     self.validation_path_list = validation_path_list

    @cached_property
    def model(self):
        model = Model(self.cfg.model.in_channels, self.cfg.model.out_channels)
                           
        if 'preload' in self.cfg.train:
            fname = self.cfg.train.preload
        else:
            fname = os.path.join(self.cfg.train.output_dir, 
                f'model_{self.cfg.train.iter_start}.chkpt')
            
        model = torch.nn.SyncBatchNorm.convert_sync_batchmore(model)

        if os.path.exists(fname) and self.local_rank==0:
            model = load_chkpt(model, fname)
        
        # note that we have to wrap the nn.DataParallel(model) before 
        # loading the model since the dictionary is changed after the wrapping
        if self.num_gpus > 1:
            print(f'use {self.num_gpus} gpus!')
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank],
                output_device=self.local_rank)
       
        model.to('cuda')
        #if torch.cuda.is_available():
        #    gpu_num = torch.cuda.device_count()
        #    print("Let's use", gpu_num, " GPUs!")
        #    model = torch.nn.parallel.DataParallel(
        #            model,
        #            device_ids=list(range(torch.cuda.device_count())),
        #    )
        
        #model = load_chkpt(
        #    model,
        #    self.cfg.train.output_dir,
        #    self.cfg.train.iter_start)
        
        return model

    @cached_property
    def optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.cfg.train.learning_rate
        )
        # Potential optimizers
        # return torch.optim.SGD(
            # self.model.parameters(),
            # lr=self.cfg.train.learning_rate
        # )

    @cached_property
    def loss_module(self):
        return BinomialCrossEntropyWithLogits()
        #return MeanSquareErrorLoss()


    @cached_property
    @abstractproperty
    def training_dataset(self):
        pass

    @cached_property
    @abstractproperty
    def validation_dataset(self):
        pass
    
    @cached_property
    def LOCAL_RANK(self):
        return int(os.getenv('LOCAL_RANK', -1))
       
    @cached_property
    def training_data_loader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.training_dataset,
            shuffle = False,
        )

        if self.cfg.system.cpus > 0:
            prefetch_factor = None
            multiprocessing_context='spawn'
        else:
            prefetch_factor = None
            multiprocessing_context=None
        
        dataloader = torch.utils.data.DataLoader(
            self.training_dataset,
             shuffle=False, 
            num_workers = self.cfg.system.cpus,
            prefetch_factor = self.cfg.system.cpus,
            collate_fn=collate_batch,
            worker_init_fn=worker_init_fn,
            batch_size=self.batch_size,
            multiprocessing_context='spawn',
            # pin_memory = True, # only dense tensor can be pinned. To-Do: enable it.
            sampler=sampler
        )

        return dataloader
        #training_data_loader = DataLoader(
        #    self.training_dataset,
        #    num_workers=0,
        #    prefetch_factor=None,
        #    drop_last=False,
        #    # multiprocessing_context='spawn', 
        #    collate_fn=collate_batch,
        #    worker_init_fn=worker_init_fn,
        #    batch_size=self.batch_size,
        #) 
        #return training_data_loader

    
    @cached_property
    def validation_data_loader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.validation_dataset,
            shuffle = False,
        )
        dataloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            shuffle=False, 
            num_workers = self.cfg.system.cpus,
            prefetch_factor = self.cfg.system.cpus,
            collate_fn=collate_batch,
            batch_size=self.batch_size,
            multiprocessing_context='spawn',
            pin_memory = True, # only dense tensor can be pinned. To-Do: enable it.
            sampler=sampler
        )

        return dataloader 
        #validation_data_loader = DataLoader(
        #    self.validation_dataset,
        #    num_workers=0,
        #    prefetch_factor=None,
        #    drop_last=False,
        #    collate_fn=collate_batch,
        #    batch_size=self.batch_size,
        #) 
        #return validation_data_loader

    @cached_property
    def validation_data_iter(self):
        validation_data_iter = iter(self.validation_data_loader)
        return validation_data_iter

    @cached_property
    def voxel_num(self):
        return np.product(self.patch_size) * self.batch_size

    def label_to_target(self, label: torch.Tensor):
        return label.cuda()

    def post_processing(self, prediction: torch.Tensor):
        if isinstance(self.loss_module, BinomialCrossEntropyWithLogits):
        #if isinstance(self.loss_module, MeanSquareErrorLoss): 
            return torch.sigmoid(prediction)
        else:
            return prediction

    def __call__(self) -> None:
        writer = SummaryWriter(log_dir=self.cfg.train.output_dir) 
        accumulated_loss = 0.
        iter_idx = self.cfg.train.iter_start
        
        for iter_idx in range(self.cfg.train.iter_start, self.cfg.train.iter.stop):
            image, label = next(iter(self.training_data_loader))
            target = self.label_to_target(label)

            #iter_idx += 1
            #if iter_idx > self.cfg.train.iter_stop:
            #    print('exceeds the maximum iteration: ', self.cfg.train.iter_stop)
            #    return
                

            ping = time()
            # print(f'preparing patch takes {round(time()-ping, 3)} seconds')
            # image.to(self.device)
            # self.model.to(self.device)
            #            predict = self.model(image)
            #breakpoint() 
            predict = self.model(image)
            predict = self.post_processing(predict)
            loss = self.loss_module(predict, target)
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
                predict = self.post_processing(predict)
                writer.add_scalar('Loss/train', per_voxel_loss, iter_idx)
                log_tensor(writer, 'train/image', image, 'image', iter_idx)
                log_tensor(writer, 'train/prediction', predict.detach(), 'image', iter_idx)
                log_tensor(writer, 'train/target', target, 'image', iter_idx)

            if iter_idx % self.cfg.train.validation_interval == 0 and self.is_main_process and iter_idx > 0:

                    #if self.LOCAL_RANK <= 0:
                    # only save model on master
                fname = os.path.join(self.cfg.train.output_dir, f'model_{iter_idx}.chkpt')
                print(f'save model to {fname}')
                if iter_idx >= self.cfg.train.start_saving:
                    print(f'save model to {fname}')
                    save_chkpt(self.model, self.cfg.train.output_dir, iter_idx, self.optimizer)

                print('evaluate prediction: ')
                validation_image, validation_label = next(self.validation_data_iter)
                validation_target = self.label_to_target(validation_label)

                with torch.no_grad():
                    validation_predict = self.model(validation_image)
                    validation_loss = self.loss_module(validation_predict, validation_target)
                    validation_predict = self.post_processing(validation_predict)
                    per_voxel_loss = validation_loss.tolist() / self.voxel_num
                    print(f'iter {iter_idx}: validation loss: {round(per_voxel_loss, 3)}')
                    writer.add_scalar('Loss/validation', per_voxel_loss, iter_idx)
                    log_tensor(writer, 'evaluate/image', validation_image, 'image', iter_idx)
                    log_tensor(writer, 'evaluate/prediction', validation_predict, 'image', iter_idx)
                    log_tensor(writer, 'evaluate/target', validation_target, 'image', iter_idx)
        
        writer.close()
        cleanup()

