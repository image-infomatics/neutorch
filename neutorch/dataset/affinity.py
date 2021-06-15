from matplotlib.pyplot import axes
from .utils import from_h5
import random
from typing import Union
from time import time

import numpy as np
import h5py

import torch
from .tio_transforms import DropAlongAxis, ZeroAlongAxis, DropSections, Transpose
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
        self.over_sample = 16
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

        rescale = tio.RescaleIntensity(
            out_min_max=(0, 1),
            # percentiles=(0.5, 0.95) # these percentiles might be desireable as in https://arxiv.org/abs/1809.10486
        )

        # orient
        transpose = Transpose(axes=(2, 3))  # only do XY for simplicity

        # spacial
        flip = tio.RandomFlip(axes=(0, 1, 2))
        axis = DropAlongAxis()
        affine = tio.RandomAffine(
            center='image',
            scales=(1.2, 1.5),
            translation=(-5, 5),
            degrees=(-8, 8),
            default_pad_value='otsu'
        )
        spacial = tio.Compose([axis, affine, flip])

        # intensity
        section = DropSections()
        zero = ZeroAlongAxis()
        loss = tio.OneOf([section, zero])  # one data loss transform

        bias = tio.RandomBiasField(coefficients=0.22)
        gamma = tio.RandomGamma(log_gamma=(-0.25, 0.25))
        contrast = tio.OneOf([gamma, bias])  # one contrast transform

        noise = tio.RandomNoise(std=(0, 0.05))
        blur = tio.RandomBlur(std=(0, 0.15))
        # either blur or noise, both would counter act
        noise_blur = tio.OneOf([noise, blur])

        intensity = tio.Compose([contrast, noise_blur, loss])

        # pipeline
        # we apply two transpose tranforms on either end,
        # this means that AlongAxis transforms will happen on either axis
        # in both the input and output space
        # should rescale be at the beginning or end?
        transforms = [transpose, spacial, intensity, transpose, rescale]
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
