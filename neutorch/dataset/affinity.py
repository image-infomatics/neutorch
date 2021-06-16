from matplotlib.pyplot import axes
from .utils import from_h5
import random
from typing import Union
from time import time

import numpy as np
import h5py

import torch
from .tio_transforms import *
from .utils import from_h5
from .ground_truth_volume import GroundTruthVolume
from .patch import AffinityBatch
import torchio as tio


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 path: str,
                 training_split_ratio: float = 0.9,
                 patch_size: Union[int, tuple] = (64, 64, 64),
                 batch_size=1,
                 ):
        """
        Parameters:
            path (str): file_path to the ground truth data.
            training_split_ratio (float): split the datasets to training and validation sets.
            patch_size (int or tuple): the patch size we are going to provide.
            batch_size (int): the number of batches in each batch
        """

        super().__init__()
        assert training_split_ratio > 0.5
        assert training_split_ratio < 1.

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3

        self.batch_size = batch_size
        self.patch_size = patch_size
        # we oversample the patch to create buffer for any transformation
        self.over_sample = 4
        patch_size_oversized = tuple(x+self.over_sample for x in patch_size)

        self._prepare_transform()

        # load all the datasets
        fileA = 'sample_A'
        fileB = 'sample_B'
        fileC = 'sample_C'

        # temporary for testing
        files = [fileA]  # , fileB, fileC]
        volumes = []

        for file in files:
            image = from_h5(f'{path}/{file}.hdf', dataset_path='volumes/raw')
            label = from_h5(
                f'{path}/{file}.hdf', dataset_path='volumes/labels/neuron_ids')

            lsd_label = np.load(f'{path}/{file}_lsd.npy')

            image = image.astype(np.float32) / 255.

            # we just trim the last slice because it is duplicate in the data
            # due to quirk of lsd algo, in future, should just fix data
            lsd_label = lsd_label[:, :-1, :, :]

            ground_truth_volume = GroundTruthVolume(
                image, label, patch_size=patch_size_oversized, lsd_label=lsd_label)
            volumes.append(ground_truth_volume)

        self.training_volumes = volumes  # volumes[1:]
        self.validation_volumes = [volumes[0]]

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

        bias = ApplyIntensityAlongZ(tio.RandomBiasField(coefficients=0.25))
        gamma = ApplyIntensityAlongZ(tio.RandomGamma(
            log_gamma=(-1, 2)))
        brightness = ApplyIntensityAlongZ(Brightness(amount=(-0.4, 0.4)))

        noise = ApplyIntensityAlongZ(tio.RandomNoise(std=(0, 0.05)))
        blur = ApplyIntensityAlongZ(tio.RandomBlur(std=(0.5, 3.5)))

        intensity = tio.Compose([bias, gamma, brightness, noise, blur, loss])

        # clip
        clip = Clip(min_max=(0.2, 0.98))

        transforms = [rescale, transposeXY,
                      spacial, intensity, transposeXY, clip]
        self.transform = tio.Compose(transforms)


if __name__ == '__main__':
    dataset = Dataset(training_split_ratio=0.99)

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
            file['main'] = image[0, 0, ...]
        with h5py.File('/tmp/label.h5', 'w') as file:
            file['main'] = label[0, 0, ...]

        print('number of nonzero voxels: ', np.count_nonzero(label))
        # assert np.count_nonzero(tbar) == 8
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        log_tensor(writer, 'train/image', image, n)
        log_tensor(writer, 'train/label', label, n)

        # # print(patch)
        # logits = UNetModel(image)
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
