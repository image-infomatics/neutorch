from .utils import from_h5
import random
from typing import Union
from time import time

import numpy as np

import torch
from .tio_transforms import *
from .utils import from_h5
from .ground_truth_volume import GroundTruthVolume
from .patch import AffinityBatch
import torchio as tio


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 path: str,
                 patch_size: Union[int, tuple] = (64, 64, 64),
                 batch_size=1,
                 ):
        """
        Parameters:
            path (str): file_path to the ground truth data.
            patch_size (int or tuple): the patch size we are going to provide.
            batch_size (int): the number of batches in each batch
        """

        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3

        self.batch_size = batch_size
        self.patch_size = patch_size
        # we oversample the patch to create buffer for any transformation
        self.over_sample = 4
        patch_size_oversized = tuple(x+self.over_sample for x in patch_size)
        # number of slices used to make validation volume
        # use the z dim of the patch size plus how ever much we oversample plus a little extra
        val_slices_bonus = 10
        self.validation_slices = patch_size_oversized[0] + val_slices_bonus

        self._prepare_transform()

        # load all the datasets
        fileA = 'sample_A'
        fileB = 'sample_B'
        fileC = 'sample_C'

        # temporary for testing
        files = [fileA, fileB, fileC]
        training_volumes = []
        validation_volumes = []

        for i in range(len(files)):
            file = files[i]
            image = from_h5(f'{path}/{file}.hdf', dataset_path='volumes/raw')
            label = from_h5(
                f'{path}/{file}.hdf', dataset_path='volumes/labels/neuron_ids')

            lsd_label = np.load(f'{path}/{file}_lsd.npy')
            image = image.astype(np.float32) / 255.

            # we just trim the last slice because it is duplicate in the data
            # due to quirk of lsd algo, in future, should just fix data
            lsd_label = lsd_label[:, :-1, :, :]

            # here we take some slices of a volume to build a valiation volume
            # for now we only take from the first training volume
            if i == 0:
                vs = self.validation_slices

                val_image = image[..., :vs, :, :]
                val_label = label[..., :vs, :, :]
                val_lsd_label = lsd_label[..., :vs, :, :]

                image = image[..., vs:, :, :]
                label = label[..., vs:, :, :]
                lsd_label = lsd_label[..., vs:, :, :]

                val_ground_truth_volume = GroundTruthVolume(
                    val_image, val_label, patch_size=patch_size_oversized, lsd_label=val_lsd_label)
                validation_volumes.append(val_ground_truth_volume)

            train_ground_truth_volume = GroundTruthVolume(
                image, label, patch_size=patch_size_oversized, lsd_label=lsd_label)
            training_volumes.append(train_ground_truth_volume)

        self.training_volumes = training_volumes
        self.validation_volumes = validation_volumes

    @property
    def random_training_batch(self):
        patches = []
        for _ in range(self.batch_size):
            p = self.random_training_patch
            patches.append(p)

        return AffinityBatch(patches)

    @property
    def random_validation_batch(self):
        patches = []
        for _ in range(self.batch_size):
            p = self.random_validation_patch
            patches.append(p)

        return AffinityBatch(patches)

    @property
    def random_validation_batch(self):
        return self.select_random_patch(self.validation_volumes)

    @property
    def random_training_patch(self):
        return self.select_random_patch(self.training_volumes)

    @property
    def random_validation_patch(self):
        return self.select_random_patch(self.validation_volumes)

    def select_random_patch(self, collection):
        volume = random.choice(collection)
        patch = volume.random_patch
        ping = time()
        patch.subject = self.transform(patch.subject)
        # print(f'transform takes {round(time()-ping, 4)} seconds.')
        patch.compute_affinity()
        # crop down from over sample to true patch size, crop after compute affinity
        crop = tio.Crop(bounds_parameters=self.over_sample//2)
        patch.subject = crop(patch.subject)

        return patch

    def _prepare_transform(self):

        # normalization
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))

        # orient
        transposeXY = Transpose(axes=(2, 3))  # only do XY for simplicity

        # spacial
        flip = tio.RandomFlip(axes=(0, 1, 2))
        slip = SlipAlongAxis(p=0.5)
        affine = Perspective2D()
        spacial = tio.Compose([slip, flip, affine])

        # loss
        drop_section = DropSections(drop_amount=(1, 30))
        drop_axis = ZeroAlongAxis()
        dropZ = DropZSlices()
        # one data loss transform
        loss = tio.OneOf({drop_section: 0.4, drop_axis: 0.3, dropZ: 0.2})

        bias = ApplyIntensityAlongZ(tio.RandomBiasField(coefficients=0.24))
        gamma = ApplyIntensityAlongZ(tio.RandomGamma(
            log_gamma=(-1, 2)))
        brightness = ApplyIntensityAlongZ(Brightness(amount=(-0.3, 0.3)))

        noise = ApplyIntensityAlongZ(tio.RandomNoise(std=(0, 0.04)))
        blur = ApplyIntensityAlongZ(tio.RandomBlur(std=(0.5, 3)))

        intensity = tio.Compose([bias, gamma, brightness, noise, blur, loss])

        # clip
        clip = Clip(min_max=(0.2, 0.98))

        transforms = [rescale, transposeXY,
                      spacial, intensity, transposeXY, clip]
        self.transform = tio.Compose(transforms)
