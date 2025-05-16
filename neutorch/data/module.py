from functools import cached_property
from typing import List

import torch
import lightning as L

from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import Cartesian
from neutorch.data.dataset import *

class IncucyteSample(SemanticSample):
    def __init__(self,
            inputs: List[Chunk | AbstractVolume],
            label: Chunk | AbstractVolume,
            output_patch_size: Cartesian = Cartesian(1, 128, 128),
            ) -> None:
        super().__init__(
            inputs,
            label,
            output_patch_size=output_patch_size,
        )

    def label_to_target(self, label: Chunk ) -> Chunk:
        # convert label to target
        target = np.concatenate([
            (label.array <= 1).astype(np.float32),
            (label.array == 2).astype(np.float32),
            (label.array == 3).astype(np.float32),
        ], axis=0)
        assert target.ndim == 4
        assert target.shape[0] == 3, 'label should have 3 channels'
        target = Chunk(target)
        target.set_properties(label.properties)
        return target

class IncucyteDataModule(L.LightningDataModule):
    def __init__(self, 
            cfg: str | CfgNode,
            dataset_type: str = 'SemanticDataset',
            ) -> None:
        super().__init__()

        if isinstance(cfg, str):
            cfg = load_cfg(cfg)
        self.cfg = cfg
        self.inputs = cfg.data.inputs
        self.labels = cfg.data.labels
        self.dataset_type = eval(dataset_type)

        print('loading samples ...')
        self.samples = dict()
        for sample_config_file in self.cfg.data.samples:
            sample_cfg = load_cfg(sample_config_file)
            for sample_name, node in sample_cfg.items():
                self.samples[sample_name] = IncucyteSample.from_config_v6(
                    node,
                    output_patch_size=self.cfg.train.patch_size,
                )

        # To-Do: training and validation, test split using 0.-1. number 
        # for example: training=0.8, validation=0.2, test=0.0
        # if isinstance(self.cfg.data.training, float):
        #     self.cfg.data.training = [self.cfg.data.training]
    
    @cached_property
    def training_dataset(self):
        samples = []
        if isinstance(self.cfg.data.training, list):
            for sample_name in self.cfg.data.training:
                samples.append(self.samples[sample_name])
        elif isinstance(self.cfg.data.validation, float):
            raise NotImplementedError(
                'Validation split by float is not implemented yet.')

        return self.dataset_type(samples)
 
    @cached_property
    def validation_dataset(self):
        samples = []
        if isinstance(self.cfg.data.validation, list):
            for sample_name in self.cfg.data.validation:
                samples.append(self.samples[sample_name])
        elif isinstance(self.cfg.data.validation, float):
            raise NotImplementedError(
                'Validation split by float is not implemented yet.')

        return self.dataset_type(samples)


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        num_workers = self.cfg.data.num_workers
        if num_workers == 0:
            dataloader = torch.utils.data.DataLoader(
                self.training_dataset,
                batch_size=self.cfg.train.batch_size,
                num_workers = num_workers,
                # prefetch_factor=2,
                drop_last=False,
                persistent_workers=False,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self.training_dataset,
                batch_size=self.cfg.train.batch_size,
                num_workers = num_workers,
                # prefetch_factor=2,
                drop_last=False,
                multiprocessing_context='spawn',
                persistent_workers=True,
            )
        return dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        num_workers = self.cfg.data.num_workers
        if num_workers == 0:
            dataloader = torch.utils.data.DataLoader(
                self.validation_dataset,
                batch_size=self.cfg.train.batch_size,
                num_workers = num_workers,
                # prefetch_factor=2,
                drop_last=False,
                persistent_workers=False,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self.validation_dataset,
                batch_size=self.cfg.train.batch_size,
                num_workers = num_workers,
                # prefetch_factor=2,
                drop_last=False,
                multiprocessing_context='spawn',
                persistent_workers=True,
            )
        return dataloader


class Wasp(L.LightningDataModule):
    def __init__(self, 
            sample_config_files: List[str], 
            inputs: List[str] = ['image'],
            labels: List[str] = ['label'],
            output_patch_size: Cartesian = Cartesian(128, 128, 128),
            batch_size: int = 1,
            dataset_type: str = 'SemanticDataset',
            ) -> None:
        super().__init__()

        self.sample_config_files = sample_config_files
        self.inputs = inputs
        self.labels = labels
        self.output_patch_size = output_patch_size
        self.batch_size = batch_size
        self.dataset_type = eval(dataset_type)

        # # define the variable first
        # self.training_dataset = None
        # self.validation_dataset = None
        assert self.training_dataset is not None
        assert self.validation_dataset is not None
    
    @cached_property
    def training_dataset(self):
        return self.dataset_type.from_config_v5(
            self.sample_config_files, 
            mode='training', 
            inputs = self.inputs,
            labels = self.labels,
            output_patch_size=self.output_patch_size
        )

    @cached_property
    def validation_dataset(self):
        return self.dataset_type.from_config_v5(
            self.sample_config_files, 
            mode='validation',
            inputs = self.inputs,
            labels = self.labels,
            output_patch_size=self.output_patch_size,
        )

    def train_dataloader(self) :
        return torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            num_workers = 8,
            # prefetch_factor=2,
            drop_last=False,
            multiprocessing_context='spawn',
        )

    def val_dataloader(self) :
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers = 8,
            # prefetch_factor=2,
            drop_last=False,
            multiprocessing_context='spawn',
        )