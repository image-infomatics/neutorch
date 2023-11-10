import os
from functools import cached_property

import click
from yacs.config import CfgNode

from neutorch.data.dataset import BoundaryAugmentationDataset, AffinityMapDataset
from neutorch.train.base import TrainerBase, setup, cleanup

from .base import TrainerBase

import torch
import torch.distributed as dist

class BoundaryAugTrainer(TrainerBase):
    def __init__(self, cfg: CfgNode,
                 #device: torch.DeviceObjType=None,
                 #local_rank: int = int(os.getenv('LOCAL_RANK', -1)) 
        ) -> None:
        assert isinstance(cfg, CfgNode)
        super().__init__(cfg) #, device=device)
        #self.cfg = cfg

    @cached_property
    def training_dataset(self):
        return AffinityMapDataset.from_config(self.cfg, mode='training')
        # return BoundaryAugmentationDataset.from_config(training_data, mode='training')
       
    @cached_property
    def validation_dataset(self):
        return AffinityMapDataset.from_config(self.cfg, mode='validation')
        # return BoundaryAugmentationDataset.from_config(validation_data, mode='validation')

    def label_to_target(self, label: torch.Tensor):
        return label

@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.'
)
#@click.option('--local-rank', '-r',
#    type=click.INT, default=int(os.getenv('LOCAL_RANK', -1)),
#    help='rank of local process. It is used to assign batches and GPU devices.'
#)
def main(config_file: str): # local_rank: int):  
    #if local_rank != -1:
    #    dist.init_process_group(backend="nccl", init_method='env://')
    #    print(f'local rank of processes: {local_rank}')
    #    torch.cuda.set_device(local_rank)
    #    device=torch.device("cuda", local_rank)

    from neutorch.data.dataset import load_cfg
    cfg = load_cfg(config_file)
    trainer = BoundaryAugTrainer(cfg) #, device=device)
    trainer()
    #cleanup()
