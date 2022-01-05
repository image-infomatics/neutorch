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

from neutorch.dataset.ground_truth_sample import GroundTruthVolumeWithPointAnnotation
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
            synapse_path = gt['synapses']
            image_path = os.path.expanduser(image_path)
            synapse_path = os.path.expanduser(synapse_path)
            print(f'image path: {image_path}')
            assert h5py.is_hdf5(image_path)
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
                assert synapses['order'] == ['z', 'y', 'x']
            # use the number of T-bars as subject sampling weights
            # subject_weights.append(len(synapses['presynapses']))
            del synapses['order']
            del synapses['resolution']

            assert len(synapses) > 0
            tbar_points = np.zeros((len(synapses), 3), dtype=np.int64)
            for idx, synapse in enumerate(synapses.values()):
                # transform xyz to zyx
                tbar_points[idx, :] = synapse['coord'] 
            tbar_points -= voxel_offset
            print(f'min offset: {np.min(tbar_points, axis=0)}')
            print(f'max offset: {np.max(tbar_points, axis=0)}')
            # all the points should be inside the image
            # np.testing.assert_array_less(np.max(tbar_points, axis=0), image.shape)

            ground_truth_sample = GroundTruthVolumeWithPointAnnotation(
                image,
                annotation_points=tbar_points,
                patch_size=patch_size_before_transform,
            )
            volumes.append(ground_truth_sample)
        
        # shuffle the volume list and then split it to training and test
        # random.shuffle(volumes)

        # use the number of candidate patches as volume sampling weight
        sample_weights = []
        for volume in volumes:
            sample_weights.append(int(volume.volume_sampling_weight))

        self.training_sample_num = int( math.floor(len(volumes) * training_split_ratio) )
        self.validation_sample_num = int(len(volumes) - self.training_sample_num)
        self.training_samples = volumes[:self.training_sample_num]
        self.validation_samples = volumes[-self.validation_sample_num:]
        self.training_sample_weights = sample_weights[:self.training_sample_num]
        self.validation_sample_weights = sample_weights[-self.validation_sample_num:]
        
    @property
    def random_training_patch(self):
        # only sample one subject, so replacement option could be ignored
        if self.training_sample_num == 1:
            sample_index = 0
        else:
            sample_index = random.choices(
                range(self.training_sample_num),
                weights=self.training_sample_weights,
                k=1,
            )[0]
        volume = self.training_samples[sample_index]
        patch = volume.random_patch
        self.transform(patch)
        patch.apply_delayed_shrink_size()
        print('patch shape: ', patch.shape)
        assert patch.shape[-3:] == self.patch_size, f'patch shape: {patch.shape}'
        return patch

    @property
    def random_validation_patch(self):
        if self.validation_sample_num == 1:
            sample_index = 0
        else:
            sample_index = random.choices(
                range(self.validation_sample_num),
                weights=self.validation_sample_weights,
                k=1,
            )[0]
        volume = self.validation_samples[sample_index]
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
            #DropSection(),
            Flip(),
            Transpose(),
            MissAlignment(),
        ])


if __name__ == '__main__':
    dataset = Dataset(
        "~/Dropbox (Simons Foundation)/40_gt/tbar.toml",
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
        target = patch.target
        with h5py.File('/tmp/image.h5', 'w') as file:
            file['main'] = image[0,0, ...]
        with h5py.File('/tmp/target.h5', 'w') as file:
            file['main'] = target[0,0, ...]

        print('number of nonzero voxels: ', np.count_nonzero(target))
        # assert np.count_nonzero(tbar) == 8
        image = torch.from_numpy(image)
        target = torch.from_numpy(target)
        log_tensor(writer, 'train/image', image, n)
        log_tensor(writer, 'train/target', target, n)

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

