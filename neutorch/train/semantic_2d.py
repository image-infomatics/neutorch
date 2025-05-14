#/bin/env python


import click

import lightning.pytorch as pl

from neutorch.model.lightning import LitSemanticRSUNet
from neutorch.model.RSUNet import Model as RSUNet
from neutorch.data.module import IncucyteDataModule

from neutorch.data.dataset import load_cfg


@click.command()
@click.option('--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True), 
    default='./config.yaml', 
    help = 'configuration file containing all the parameters.'
)
@click.option('--devices', '-d', default="auto", 
    help='number of devices')
@click.option('--accelerator', '-a', type=click.Choice(['auto', 'gpu', 'cpu', 'cuda', 'hpu', 'ipu', 'mps', 'tpu']), default='auto',
    help='accelerator to use, [auto, cpu, cuda, hpu, ipu, mps, tpu]')
@click.option('--strategy', '-s', 
    type=click.Choice(['ddp', 'ddp_spawn', 'auto']), default='auto')
@click.option('--log-every-n-steps', '-n', default=10000, help='log every n steps')
@click.option('--chkpt-file', '-k',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True, resolve_path=True),
    default= None,
    help = 'trained model checkpoint file'
)
def main(config_file: str, devices: int, accelerator: str, 
         strategy: str, log_every_n_steps: int, chkpt_file: str):
    cfg = load_cfg(config_file)

    datamodule = IncucyteDataModule(cfg)

    # torch.distributed.init_process_group(backend="nccl")
    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=devices, 
        strategy=strategy,
        max_epochs=-1, # unlimited training epochs
        log_every_n_steps = log_every_n_steps,
    )

    
    if chkpt_file is not None:
        model = LitSemanticRSUNet.load_from_checkpoint(
            chkpt_file,
            model=RSUNet(cfg.model.in_channels, cfg.model.out_channels),
        )
    else:
        model = LitSemanticRSUNet(cfg)

    trainer.fit(
        model = model, 
        datamodule=datamodule,
    )



