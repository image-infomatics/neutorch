import json
import os
import math
import random
from typing import Union
from time import time, sleep

import numpy as np
import h5py

from chunkflow.chunk import Chunk

import torch
import toml

from .ground_truth_volume import GroundTruthVolume
from .transform import *


def image_reader(path: str):
    with h5py.File(path, 'r') as file:
        img = np.asarray(file['main'])
    # the last one is affine transformation matrix in torchio image type
    return img, None


class CremiDataset(torch.utils.data.Dataset):
    def __init__(self, 
            training_split_ratio: float = 0.9,
            patch_size: Union[int, tuple]=(64, 64, 64), 
            ):
        """
        Parameters:
            config_file (str): file_path to provide metadata of all the ground truth data.
            training_split_ratio (float): split the datasets to training and validation sets.
            patch_size (int or tuple): the patch size we are going to provide.
        """

        super().__init__()
        assert training_split_ratio > 0.5
        assert training_split_ratio < 1.

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3

        self._prepare_transform()
        assert isinstance(self.transform, Compose)

        self.patch_size = patch_size
        # we oversample the patch to create buffer for any transformation
        over_sample = 4
        patch_size_oversized = tuple(x+over_sample for x in patch_size)

        # load all the datasets
        fileA = './data/cremi/sample_A.hdf'
        fileB = './data/cremi/sample_B.hdf'
        fileC = './data/cremi/sample_C.hdf'

        files = [fileA]#, fileB, fileC]
        volumes = []
        for file in files:
            image = Chunk.from_h5(file, dataset_path='volumes/raw')
            label = Chunk.from_h5(file, dataset_path='volumes/labels/neuron_ids')

            image = image.astype(np.float32) / 255.
            ground_truth_volume = GroundTruthVolume(image,label,patch_size=patch_size_oversized)
            volumes.append(ground_truth_volume)
        
        self.training_volumes = volumes # volumes[1:]
        self.validation_volumes = [volumes[0]]


    @property
    def random_training_patch(self):
        return self.select_random_patch(self.training_volumes)

    @property
    def random_validation_patch(self):
        return self.select_random_patch(self.validation_volumes)


    def select_random_patch(self, collection):
        volume = random.choice(collection)
        patch = volume.random_patch
        self.transform(patch)
        patch.compute_target()
        print('patch shape: ', patch.shape)
        # assert patch.shape[-3:] == self.patch_size, f'patch shape: {patch.shape}'
        return patch
           
    def _prepare_transform(self):
        self.transform = Compose([
            NormalizeTo01(probability=1.0),

            # Intensity Transforms
            # AdjustBrightness(),
            # AdjustContrast(),
            # Gamma(),
            # OneOf([
            #     Noise(),
            #     GaussianBlur2D(),
            # ]),
            # BlackBox(),

            # Spatial Transforms
            # RotateScale(probability=1.),

            # MissAlignment(),
            # DropAlongAxis(),
            # Flip(),
            # Transpose(),
            Perspective2D(),

        ])


if __name__ == '__main__':
    dataset = CremiDataset(training_split_ratio=0.99)

    from torch.utils.tensorboard import SummaryWriter
    from neutorch.model.io import log_tensor
    writer = SummaryWriter(log_dir='/tmp/log')

    import h5py
    
    model = torch.nn.Identity()
    print('start generating random patches...')
    for n in range(10000):
        ping = time()
        patch = dataset.random_training_patch
        print(f'generating a patch takes {round(time()-ping, 3)} seconds.')
        image = patch.image
        label = patch.label
        with h5py.File('/tmp/image.h5', 'w') as file:
            file['main'] = image[0,0, ...]
        with h5py.File('/tmp/label.h5', 'w') as file:
            file['main'] = label[0,0, ...]

        print('number of nonzero voxels: ', np.count_nonzero(label))
        # assert np.count_nonzero(tbar) == 8
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        log_tensor(writer, 'train/image', image, n)
        log_tensor(writer, 'train/label', label, n)

        # # print(patch)
        # logits = model(image)
        # image = image[:, :, 32, :, :]
        # tbar, _ = torch.max(tbar, dim=2, keepdim=False)
        # slices = torch.cat((image, tbar))
        # image_path = os.path.expanduser('~/Downloads/patches.png')
        # print('save a batch of patches to ', image_path)
        # torchvision.utils.save_image(
        #     slices,
        #     image_path,
        #     nrow=8,
        #     normalize=True,
        #     scale_each=True,
        # )
        sleep(1)

