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

from neutorch.dataset.ground_truth_volume import GroundTruthVolumeWithPointAnnotation
from neutorch.dataset.transform import *


def image_reader(path: str):
    with h5py.File(path, 'r') as file:
        img = np.asarray(file['main'])
    # the last one is affine transformation matrix in torchio image type
    return img, None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config_file: str, 
            training_split_ratio: float = 0.9,
            patch_size: Union[int, tuple]=(64, 64, 64), 
            max_sampling_distance: Union[int, tuple] = None):
        """
        Parameters:
            config_file (str): file_path to provide metadata of all the ground truth data.
            training_split_ratio (float): split the datasets to training and validation sets.
            patch_size (int or tuple): the patch size we are going to provide.
            max_sampling_distance (int or tuple, optional): sampling patches around the annotated T-bar point 
                limited by a maximum distance.
        """
        super().__init__()
        assert training_split_ratio > 0.5
        assert training_split_ratio < 1.

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3

        if max_sampling_distance is None:
            max_sampling_distance = tuple(p // 2 for p in patch_size)
        elif isinstance(max_sampling_distance, int):
            max_sampling_distance = (max_sampling_distance,) * 3
        assert len(max_sampling_distance) == 3

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
            synapse_path = gt['ground_truth']
            assert image_path.endswith('.h5')
            assert synapse_path.endswith('.json')
            image_path = os.path.join(config_dir, image_path)
            synapse_path = os.path.join(config_dir, synapse_path)

            image = Chunk.from_h5(image_path)
            voxel_offset = image.voxel_offset
            image = image.astype(np.float32) / 255.
            # use the voxel number as the sampling weights
            # subject_weights.append(len(img))
            with open(synapse_path, 'r') as file:
                synapses = json.load(file)
                assert synapses['order'] == ['x', 'y', 'z']
            # use the number of T-bars as subject sampling weights
            # subject_weights.append(len(synapses['presynapses']))
            presynapses = synapses['presynapses']
            assert len(presynapses) > 0
            tbar_points = np.zeros((len(presynapses), 3), dtype=np.int64)
            for idx, point in  enumerate(presynapses.values()):
                # transform xyz to zyx
                tbar_points[idx, :] = point[::-1] 
                # tbar_points[idx, 0] = point[2]
                # tbar_points[idx, 1] = point[1]
                # tbar_points[idx, 2] = point[0]
            tbar_points -= voxel_offset
            print(f'min offset: {np.min(tbar_points, axis=0)}')
            print(f'max offset: {np.max(tbar_points, axis=0)}')
            # all the points should be inside the image
            np.testing.assert_array_less(np.max(tbar_points, axis=0), image.shape)

            ground_truth_volume = GroundTruthVolumeWithPointAnnotation(
                image,
                annotation_points=tbar_points,
                patch_size=patch_size_before_transform,
                max_sampling_distance=max_sampling_distance,
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
            NormalizeTo01(probability=1.),
            AdjustBrightness(),
            AdjustContrast(),
            Gamma(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            BlackBox(),
            Perspective2D(),
            # RotateScale(probability=1.),
            DropSection(),
            Flip(),
            Transpose(),
            MissAlignment(),
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

