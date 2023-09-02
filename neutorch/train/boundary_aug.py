from functools import cached_property

#import os
import click
from yacs.config import CfgNode
import torch
import torch.distributed as dist

from neutorch.data.dataset import BoundaryAugmentationDataset
from .base import TrainerBase, setup, cleanup

class BoundaryAugTrainer(TrainerBase):
    def __init__(self, cfg: CfgNode) -> None:
        assert isinstance(cfg, CfgNode) 
        super().__init__(cfg) 

        self.cfg = cfg
        #self.num_classes = self.cfg.model.out_channels

    @cached_property
    def training_dataset(self):
        return BoundaryAugmentationDataset.from_config(self.cfg, is_train=True)
       
    @cached_property
    def validation_dataset(self):
        return BoundaryAugmentationDataset.from_config(self.cfg, is_train=False)
    
    def label_to_target(self, label: torch.Tensor):
        return label

@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.'
)
def main(config_file: str):  
    from neutorch.data.dataset import load_cfg
    cfg = load_cfg(config_file)
    trainer = BoundaryAugTrainer(cfg) #, device=device)
    trainer()
