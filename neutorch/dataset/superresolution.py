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

from neutorch.dataset.ground_truth_volume import GroundTruthVolume
from neutorch.dataset.transform import *


def image_reader(path: str):
    with h5py.File(path, 'r') as file:
        img = np.asarray(file['main'])
    # the last one is affine transformation matrix in torchio image type
    return img, None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config_file: str, 
            training_split_ratio: float = 0.9,
            patch_size: Union[int, tuple]=(64, 64, 64)):
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

        config_file = os.path.expanduser(config_file)
        assert config_file.endswith('.toml'), "we use toml file as configuration format."
        with open(config_file, 'r') as file:
            meta = toml.load(file)
        config_dir = os.path.dirname(config_file)

        self._prepare_transform()
        assert isinstance(self.transform, Compose)

        self.patch_size = patch_size
        patch_size_before_transform = tuple(
            p + s0 + s1 for p, s0, s1 in zip(
                patch_size, 
                self.transform.shrink_size[:3], 
                self.transform.shrink_size[-3:]
            )
        )
        
        # load all the datasets
        volumes = []
        for gt in meta.values():
            image_path = gt['image']
            assert image_path.endswith('.h5')
            image_path = os.path.join(config_dir, image_path)

            image = Chunk.from_h5(image_path)
            image = image.astype(np.float32) / 255.
            ground_truth_volume = GroundTruthVolume(
                image,
                image,
                patch_size=patch_size_before_transform,
            )
            volumes.append(ground_truth_volume)
        
        # shuffle the volume list and then split it to training and test
        # random.shuffle(volumes)

        # use the number of candidate patches as volume sampling weight
        volume_weights = []
        for volume in volumes:
            volume_weights.append(volume.volume_sampling_weight)

        self.training_volume_num = math.floor(len(volumes) * training_split_ratio)
        self.validation_volume_num = len(volumes) - self.training_volume_num
        self.training_volumes = volumes[:self.training_volume_num]
        self.validation_volumes = volumes[-self.validation_volume_num:]
        self.training_volume_weights = volume_weights[:self.training_volume_num]
        self.validation_volume_weights = volume_weights[-self.validation_volume_num]
        
    @property
    def random_training_patch(self):
        # only sample one subject, so replacement option could be ignored
        if self.training_volume_num == 1:
            volume_index = 0
        else:
            volume_index = random.choices(
                range(self.training_volume_num),
                weights=self.training_volume_weights,
                k=1,
            )[0]
        volume = self.training_volumes[volume_index]
        patch = volume.random_patch
        self.transform(patch)
        patch.apply_delayed_shrink_size()
        print('patch shape: ', patch.shape)
        assert patch.shape[-3:] == self.patch_size, f'patch shape: {patch.shape}'
        return patch

    @property
    def random_validation_patch(self):
        if self.validation_volume_num == 1:
            volume_index = 0
        else:
            volume_index = random.choices(
                range(self.validation_volume_num),
                weights=self.validation_volume_weights,
                k=1,
            )[0]
        volume = self.validation_volumes[volume_index]
        patch = volume.random_patch
        self.transform(patch)
        patch.apply_delayed_shrink_size()
        return patch
           
    def _prepare_transform(self):
        self.transform = Compose([
            NormalizeTo01(),
            AdjustBrightness(),
            AdjustContrast(),
            Gamma(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            BlackBox(),
        ])


if __name__ == '__main__':
    dataset = Dataset(
        "~/Dropbox (Simons Foundation)/40_gt/tbar.toml",
        max_sampling_distance=24,
        training_split_ratio=0.99,
    )

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

        assert np.any(label > 0)
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

