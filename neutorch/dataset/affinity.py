from .utils import from_h5
import random
from typing import Union
from time import time
import math
import numpy as np

import torch
from .tio_transforms import *
from .utils import from_h5
from .ground_truth_volume import GroundTruthVolume
from .patch import AffinityBatch
import torchio as tio
from skimage.transform import rescale


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 path: str,
                 length: int,
                 lsd: bool = True,
                 patch_size: Union[int, tuple] = (26, 256, 256),
                 aug: bool = True,
                 border_width=1,
                 affinity_offsets=[(1, 1, 1)],
                 float16: bool = False,
                 downsample: float = 1.0,
                 ):
        """
        Parameters:
            path (str): file_path to the ground truth data.
            length (int): number of examples
            lsd (bool): whether to use multiask lsd target
            patch_size (int or tuple): the patch size we are going to provide.
            aug (bool): whether to use data augmentation
            affinity_offsets (List of tuple), amount of offset in (x, y, z) for each affinity map, used for Long range affinity
            border_width (int): border width for affinty map
            downsample (int): factor to downsample volumes by
        """

        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3

        # keep track of total length and current index
        self.length = length
        self.aug = aug
        self.patch_size = patch_size
        self.float16 = float16

        # we oversample the patch to create buffer for any transformation
        mz, my, mx = (2, 2, 2)
        for os in affinity_offsets:
            mz, my, mx = max(os[0], mz), max(os[1], my), max(os[2], mx)
        self.over_sample = (mz*2, my*2, mx*2)
        patch_size_oversized = tuple(
            p+o for p, o in zip(patch_size, self.over_sample))

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
            print(f'loading file {file}...')
            image = from_h5(f'{path}/{file}.hdf', dataset_path='volumes/raw')
            label = from_h5(
                f'{path}/{file}.hdf', dataset_path='volumes/labels/neuron_ids')

            lsd_label = None
            if lsd:
                lsd_label = np.load(f'{path}/{file}_lsd.npy')

                # we just trim the last slice because it is duplicate in the data
                # due to quirk of lsd algo, in future, should just fix data
                lsd_label = lsd_label[:, :-1, :, :]

            # # here we take some slices of a volume to build a valiation volume
            # # for now we only take from the first training volume
            # if i == 0:
            #     vs = self.validation_slices

            #     val_image = image[..., :vs, :, :]
            #     val_label = label[..., :vs, :, :]

            #     image = image[..., vs:, :, :]
            #     label = label[..., vs:, :, :]

            #     val_lsd_label = None
            #     if lsd:
            #         val_lsd_label = lsd_label[..., :vs, :, :]
            #         lsd_label = lsd_label[..., vs:, :, :]

            #     val_ground_truth_volume = GroundTruthVolume(
            #         val_image, val_label, patch_size=patch_size_oversized,
            #         affinity_offsets=affinity_offsets,
            #         lsd_label=val_lsd_label,
            #         border_width=border_width, name=f'{file}_val')
            #     validation_volumes.append(val_ground_truth_volume)

            if downsample != 1.0:
                image = rescale(image, downsample, anti_aliasing=False)
                label = rescale(label, downsample, anti_aliasing=False)
                if lsd:
                    lsd_label = rescale(
                        lsd_label, downsample, anti_aliasing=False)

            train_ground_truth_volume = GroundTruthVolume(
                image, label, patch_size=patch_size_oversized,
                affinity_offsets=affinity_offsets,
                lsd_label=lsd_label, border_width=border_width, name=f'{file}_train')

            training_volumes.append(train_ground_truth_volume)

        self.training_volumes = training_volumes
        self.validation_volumes = training_volumes

    def random_validation_batch(self, batch_size):
        patches = []
        for i in range(batch_size):
            patches.append(self.random_validation_patch)
        return AffinityBatch(patches)

    def random_training_batch(self, batch_size):
        patches = []
        for i in range(batch_size):
            patches.append(self.random_training_patch)
        return AffinityBatch(patches)

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
        # print(f'transform takes {round(time()-ping, 3)} seconds.')
        patch.compute_affinity()
        # crop down from over sample to true patch size, crop after compute affinity
        crop_bds = tuple([x//2 for x in self.over_sample])
        crop = tio.Crop(bounds_parameters=crop_bds)
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

        # light
        bias = ApplyIntensityAlongZ(tio.RandomBiasField(coefficients=0.24))
        gamma = ApplyIntensityAlongZ(tio.RandomGamma(
            log_gamma=(-1, 2)))
        brightness = ApplyIntensityAlongZ(Brightness(amount=(-0.3, 0.3)))

        # focus
        noise = ApplyIntensityAlongZ(tio.RandomNoise(std=(0, 0.04)))
        blur = ApplyIntensityAlongZ(tio.RandomBlur(std=(0.5, 3)))

        # all intensities
        intensity = tio.Compose([bias, gamma, brightness, noise, blur, loss])

        # clip
        clip = Clip(min_max=(0.0, 1.0))

        # compose transforms
        transforms = [rescale, transposeXY,
                      spacial, intensity, transposeXY, clip]

        self.transform = tio.Compose(transforms)

        if not self.aug:
            self.transform = tio.Compose([rescale, clip])

    def __getitem__(self, idx):
        patch = self.random_training_patch
        X = patch.image
        y = patch.target
        if self.float16:
            X = X.astype(np.float16)
            y = y.astype(np.float16)
        return X, y

    def __len__(self):
        return self.length
