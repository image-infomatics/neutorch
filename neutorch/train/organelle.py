import os
from functools import cached_property
from time import time

import numpy as np
import click
from yacs.config import CfgNode

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from .base import TrainerBase
from neutorch.dataset.organelle import OrganelleDataset, to_tensor
from neutorch.model.io import save_chkpt, log_tensor


class OrganelleTrainer(TrainerBase):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
    
        self.num_classes = self.cfg.model.out_channels

    @cached_property
    def training_dataset(self):
        return OrganelleDataset(
            self.training_path_list,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            skip_classes=self.cfg.dataset.skip_classes
        )
       
    @cached_property
    def validation_dataset(self):
        return OrganelleDataset(
            self.validation_path_list,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            skip_classes=self.cfg.dataset.skip_classes
        )
    
    @cached_property
    def loss_module(self):
        class_counts = self.training_dataset.class_counts
        # the equation is from scikit-learn
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        class_weights = self.training_dataset.voxel_num / (self.num_classes * class_counts)
        class_weights = class_weights.astype(np.float32)
        print(f'class weights: {class_weights}')
        # send weights to device as a pytorch Tensor
        class_weights = to_tensor(class_weights)
        return CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def post_processing(self, predict: torch.Tensor):
        predict = torch.argmax(predict, dim=1, keepdim=False)
        # predict = predict.to(torch.int64)
        return predict

    def __call__(self) -> None:
        writer = SummaryWriter(log_dir=self.cfg.train.output_dir)
        accumulated_loss = 0.
        iter_idx = self.cfg.train.iter_start
        for image, label in self.training_data_loader:
            iter_idx += 1
            if iter_idx> self.cfg.train.iter_stop:
                print('exceeds the maximum iteration: ', self.cfg.train.iter_stop)
                return

            ping = time()
            # print(f'preparing patch takes {round(time()-ping, 3)} seconds')
            predict = self.model(image)
            # predict = self.post_processing(predict)
            loss = self.loss_module(predict, label)
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
                writer.add_scalar('loss/train', per_voxel_loss, iter_idx)
                log_tensor(writer, 'train/image', image, 'image', iter_idx)
                log_tensor(writer, 'train/prediction', predict, 'segmentation', iter_idx)
                log_tensor(writer, 'train/label', label, 'segmentation', iter_idx)

            if iter_idx % self.cfg.train.validation_interval == 0 and iter_idx > 0:
                fname = os.path.join(self.cfg.train.output_dir, f'model_{iter_idx}.chkpt')
                print(f'save model to {fname}')
                save_chkpt(self.model, self.cfg.train.output_dir, iter_idx, self.optimizer)

                print('evaluate prediction: ')
                validation_image, validation_label = next(self.validation_data_iter)

                with torch.no_grad():
                    validation_predict = self.model(validation_image)
                    validation_loss = self.loss_module(validation_predict, validation_label)
                    validation_predict = self.post_processing(validation_predict)
                    per_voxel_loss = validation_loss.tolist() / self.voxel_num
                    print(f'iter {iter_idx}: validation loss: {round(per_voxel_loss, 3)}')
                    writer.add_scalar('loss/validation', per_voxel_loss, iter_idx)
                    log_tensor(writer, 'validation/image', validation_image, 'image', iter_idx)
                    log_tensor(writer, 'validation/prediction', validation_predict, 'segmentation', iter_idx)
                    log_tensor(writer, 'validation/label', validation_label, 'segmentation', iter_idx)

        writer.close()


@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.'
)
def main(config_file: str):
    trainer = OrganelleTrainer(config_file)
    trainer()