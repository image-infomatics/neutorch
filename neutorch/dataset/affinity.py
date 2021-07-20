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


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 path: str,
                 length: int,
                 lsd: bool = True,
                 patch_size: Union[int, tuple] = (26, 256, 256),
                 aug: bool = True,
                 border_width=1,
                 affinity_offsets=[(1, 1, 1)]
                 ):
        """
        Parameters:
            path (str): file_path to the ground truth data.
            length (int): number of examples
            lsd (bool): whether to use multiask lsd target
            patch_size (int or tuple): the patch size we are going to provide.
            aug (bool): whether to use data augmentation
            border_width (int): border width for affinty map
        """

        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3

        # keep track of total length and current index
        self.length = length
        self.aug = aug
        self.patch_size = patch_size
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

                image = image[..., vs:, :, :]
                label = label[..., vs:, :, :]

                val_lsd_label = None
                if lsd:
                    val_lsd_label = lsd_label[..., :vs, :, :]
                    lsd_label = lsd_label[..., vs:, :, :]

                val_ground_truth_volume = GroundTruthVolume(
                    val_image, val_label, patch_size=patch_size_oversized,
                    affinity_offsets=affinity_offsets,
                    lsd_label=val_lsd_label,
                    border_width=border_width, name=f'{file}_val')
                validation_volumes.append(val_ground_truth_volume)

            train_ground_truth_volume = GroundTruthVolume(
                image, label, patch_size=patch_size_oversized,
                affinity_offsets=affinity_offsets,
                lsd_label=lsd_label, border_width=border_width, name=f'{file}_train')

            training_volumes.append(train_ground_truth_volume)

        self.training_volumes = training_volumes
        self.validation_volumes = validation_volumes


    def random_validation_batch(self, batch_size):
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
        return X, y

    def __len__(self):
        return self.length


class TestDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path: str,
                 patch_size: tuple,
                 stride: tuple,
                 with_label: bool = False,
                 crop_offset: bool = False,
                 ):
        """
        Parameters:
            path (str): file_path to the test  data.
            patch_size (tuple): the patch size we are going to provide.
            with_label (bool): get label also
            stride (tuple): amount to over sampling area each example
            crop_offset (bool): weather to crop the input from offest
        """

        super().__init__()

        print(f'loading from {path}...')
        volume = from_h5(path, dataset_path='volumes/raw')
        volume = volume.astype(np.float32) / 255.
        if with_label:
            self.label, self.label_offset = from_h5(
                path, dataset_path='volumes/labels/neuron_ids', get_offset=True)

        (pz, py, px) = patch_size
        
        # true shape is what we finially crop to from 0:true_shape
        self.true_shape = volume.shape

        if crop_offset:
            (shz, shy, shx) = self.label.shape
            self.true_shape = self.label.shape
            (oz, oy, ox) = self.label_offset
            # also add patch size for better context on edges
            volume = volume[oz:oz+shz+pz, oy:oy+shy+py, ox:ox+shx+px]
            


        (z, y, x) = volume.shape
    
        (sz, sy, sx) = stride

        # add padding for overlap
        volume = np.pad(volume, ((0, pz), (0, py), (0, px)))

        # full shape is with all the padding
        self.full_shape = volume.shape

        self.stride = stride
        self.patch_size = patch_size
        self.volume = volume
        self.z_len = int(math.ceil(z / sz))
        self.y_len = int(math.ceil(y / sy))
        self.x_len = int(math.ceil(x / sx))
        self.length = (self.z_len * self.y_len * self.x_len)

        # pregenerate all sampling indices
        self.all_indices = self._gen_indices()
        self.length = len(self.all_indices)

    def _gen_indices(self):
        (pz, py, px) = self.stride
        (iz, iy, ix) = (0, 0, 0)
        all_indices = [(iz, iy, ix)]
        for _ in range(self.length):
            nx = ix + px
            ny = iy
            nz = iz
            if(nx + px > self.x_len*px):
                nx = 0
                ny += py
                if(ny + py > self.y_len*py):
                    ny = 0
                    nz += pz
            all_indices.append((nz, ny, nx))
            (iz, iy, ix) = (nz, ny, nx)
        return all_indices

    def get_indices(self, idx):
        return self.all_indices[idx]

    def __getitem__(self, idx):

        (iz, iy, ix) = self.all_indices[idx]
        (pz, py, px) = self.patch_size
        patch = self.volume[iz:iz+pz, iy:iy+py, ix:ix+px]

        # convert to torch and add dim for channel
        patch = torch.Tensor(patch)
        patch = torch.unsqueeze(patch, 0)

        return patch

    def __len__(self):
        return self.length
