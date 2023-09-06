#/bin/env python

import os
from functools import cached_property

import click
from yacs.config import CfgNode

from neutorch.data.dataset import VolumeWithMask
from neutorch.train.base import TrainerBase, setup, cleanup

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
# torch.multiprocessing.set_start_method('spawn')


class WholeBrainTrainer(TrainerBase):
    def __init__(self, cfg: CfgNode, 
            device: torch.DeviceObjType=None,
            local_rank: int = int(os.getenv('LOCAL_RANK', -1))
        ) -> None:
        super().__init__(cfg, device=device, local_rank=local_rank)
        assert isinstance(cfg, CfgNode)

    @cached_property
    def training_dataset(self):
        return VolumeWithMask.from_config(self.cfg, mode='training')
       
    @cached_property
    def validation_dataset(self):
        return VolumeWithMask.from_config(self.cfg, mode='validation')



@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.'
)
@click.option('--local-rank', '-r',
    type=click.INT, default=int(os.getenv('LOCAL_RANK', -1)),
    help='rank of local process. It is used to assign batches and GPU devices.'
)
@record
def main(config_file: str, local_rank: int):
    print(f'local rank: {local_rank}')
    if local_rank != -1:
        dist.init_process_group(backend="nccl", init_method='env://')
        print(f'local rank of processes: {local_rank}')
        torch.cuda.set_device(local_rank)
        device=torch.device("cuda", local_rank)
    else:
        setup()

    from neutorch.data.dataset import load_cfg
    cfg = load_cfg(config_file)
    trainer = WholeBrainTrainer(cfg, device=device)
    trainer()
    cleanup()
