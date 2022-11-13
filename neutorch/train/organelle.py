from functools import cached_property

import click
from yacs.config import CfgNode

from .base import TrainerBase
from neutorch.dataset.organelle import OrganelleDataset


class OrganelleTrainer(TrainerBase):
    def __init__(self, cfg: CfgNode, batch_size: int = 1) -> None:
        super().__init__(cfg, batch_size)

    @cached_property
    def training_dataset(self):
        return OrganelleDataset(
            self.training_path_list,
            patch_size=self.patch_size,
        )
    
    @cached_property
    def validation_dataset(self):
        return OrganelleDataset(
            self.validation_path_list,
            patch_size=self.patch_size,
        )


@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.'
)
def main(config_file: str):
    trainer = OrganelleTrainer(config_file)
    trainer()