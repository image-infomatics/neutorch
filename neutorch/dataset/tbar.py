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
from torch.utils.data import random_split
import torchvision
import torchio as tio
import toml

from .ground_truth_volume import GroundTruthVolume


def image_reader(path: str):
    with h5py.File(path, 'r') as file:
        img = np.asarray(file['main'])
    # the last one is affine transformation matrix in torchio image type
    return img, None

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config_file: str, training_split_ratio: float = 0.9,
            patch_size: Union[int, tuple]=64, sampling_distance: int = 22):
        """
        Parameters:
            config_file (str): file_path to provide metadata of all the ground truth data.
            training_split_ratio (float): split the datasets to training and validation sets.
            patch_size (int or tuple): the patch size we are going to provide.
            sampling_distance (int): sampling patches around the annotated T-bar point 
                limited by a maximum distance.
        """
        super().__init__()
        assert training_split_ratio > 0.5
        assert training_split_ratio < 1.
        config_file = os.path.expanduser(config_file)
        assert config_file.endswith('.toml'), "we use toml file as configuration format."

        with open(config_file, 'r') as file:
            meta = toml.load(file)

        config_dir = os.path.dirname(config_file)
        
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
            image = image.astype(np.float32) / 255.
            # use the voxel number as the sampling weights
            # subject_weights.append(len(img))
            with open(synapse_path, 'r') as file:
                synapses = json.load(file)
                assert synapses['order'] = ['x', 'y', 'z']
            # use the number of T-bars as subject sampling weights
            # subject_weights.append(len(synapses['presynapses']))
            presynapses = synapses['presynapses']
            tbar_points = np.zeros((len(presynapses), 3), dtype=np.unit32)
            for idx, point in  enumerate(presynapses.values()):
                # transform xyz to zyx
                tbar_points[idx, :] = point[::-1]
                # tbar_points[idx, 0] = point[2]
                # tbar_points[idx, 1] = point[1]
                # tbar_points[idx, 2] = point[0]

            target = self._annotation_to_target_volume(
                image, tbar_points
            )

            ground_truth_volume = GroundTruthVolume(image, target,
                patch_size=patch_size,
                annotation_points=tbar_points
            )
            volumes.appen(ground_truth_volume)
        
        # shuffle the volume list and then split it to training and test
        volumes = random.shuffle(self.volumes)

        # use the number of candidate patches as volume sampling weight
        volume_weights = []
        for volume in volumes:
            volume_weights.append(volume.candidate_patch_num)

        training_volume_num = math.floor(len(volumes) * training_split_ratio)
        validation_volume_num = len(volumes) - training_volume_num
        self.training_volumes = volumes[:training_volume_num]
        self.validation_volumes = volumes[-validation_volume_num:]
        self.training_volume_weights = volume_weights[:training_volume_num]
        self.validation_volume_weights = volume_weights[-validation_volume_num]
        
    @property
    def random_training_patch(self):
        # only sample one subject, so replacement option could be ignored
        volume_index = torch.utils.data.WeightedRandomSampler(self.training_volume_weights, 1)
        volume = self.training_volumes[volume_index]
        return volume.random_patch
    
    @property
    def random_validation_patch(self):
        volume_index = torch.utils.data.WeightedRandomSampler(self.validation_volume_weights, 1)
        volume = self.validation_volumes[volume_index]
        return volume.random_patch
           
    def _annotation_to_target_volume(self, image: Chunk, tbar_points: list,
            expand_distance: int = 2) -> tuple:
        """transform point annotation to volumes

        Args:
            img (np.ndarray): image volume
            voxel_offset (np.ndarray): offset of image volume
            synapses (dict): the annotated synapses
            sampling_distance (int, optional): the maximum distance from the annotated point to 
                the center of sampling patch. Defaults to 22.
            expand_distance (int): expand the point annotation to a cube. 
                This will help to got more positive voxels.
                The expansion should be small enough to ensure that all the voxels are inside T-bar.

        Returns:
            bin_presyn: binary label of annotated position.
            sampling_probability_map: the probability map of sampling
        """
        # assert synapses['resolution'] == [8, 8, 8]
        bin_presyn = np.zeros_like(image.array, dtype=np.float32)
        voxel_offset = np.asarray(image.voxel_offset, dtype=np.uint32)
        for coordinate in tbar_points:
            # transform coordinate from xyz order to zyx
            coordinate = coordinate[::-1]
            coordinate = np.asarray(coordinate, dtype=np.uint32)
            coordinate -= voxel_offset
            bin_presyn[
                coordinate[0]-expand_distance : coordinate[0]+expand_distance,
                coordinate[1]-expand_distance : coordinate[1]+expand_distance,
                coordinate[2]-expand_distance : coordinate[2]+expand_distance,
            ] = 1.
        return bin_presyn
    
    @property
    def transform(self):
        return tio.Compose([
            # tio.RandomMotion(p=0.2),
            tio.RandomBiasField(p=0.3),
            tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            # tio.OneOf({
            #     tio.RandomAffine(): 0.3,
            #     tio.RandomElasticDeformation(): 0.7
            # }),
            tio.RandomGamma(p=0.1),
            tio.RandomGhosting(p=0.1),
            tio.RandomAnisotropy(p=0.2),
            tio.RandomSpike(p=0.1),
        ])

if __name__ == '__main__':
    dataset = Dataset(
        "~/Dropbox (Simons Foundation)/40_gt/tbar.toml",
        num_workers=1,
        sampling_distance=4,
        training_split_ratio=0.99,
    )
    # we only left one subject as validation set
    training_batch_size = dataset.training_subjects_num
    patches_loader = torch.utils.data.DataLoader(
        dataset.random_training_patches,
        batch_size=training_batch_size
    )
    
    model = torch.nn.Identity()
    print('start generating random patches...')
    ping = time()
    for n in range(100):
        patches_batch = next(iter(patches_loader))
        print(f'generating a patch takes {int(time()-ping)} seconds.')
        # print(patch)
        image = patches_batch['image'][tio.DATA]
        logits = model(image)
        assert image.shape[0] == training_batch_size
        image = image[:, :, 32, :, :]
        tbar = patches_batch['tbar'][tio.DATA]
        tbar, _ = torch.max(tbar, dim=2, keepdim=False)
        # breakpoint()
        slices = torch.cat((image, tbar))
        image_path = os.path.expanduser('~/Downloads/patches.png')
        print('save a batch of patches to ', image_path)
        torchvision.utils.save_image(
            slices,
            image_path,
            nrow=training_batch_size,
            normalize=True,
            scale_each=True,
        )

        ping = time()
