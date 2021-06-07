import random
from typing import Union
import time

import numpy as np
import h5py

from chunkflow.chunk import Chunk

import torch

from .ground_truth_volume import GroundTruthVolume
import torchio as tio


class CremiDataset(torch.utils.data.Dataset):
    def __init__(self,
                 training_split_ratio: float = 0.9,
                 patch_size: Union[int, tuple] = (64, 64, 64),
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

        self.patch_size = patch_size
        # we oversample the patch to create buffer for any transformation
        self.over_sample = 4
        patch_size_oversized = tuple(x+self.over_sample for x in patch_size)

        self._prepare_transform()

        # load all the datasets
        fileA = './data/cremi/sample_A.hdf'
        fileB = './data/cremi/sample_B.hdf'
        fileC = './data/cremi/sample_C.hdf'

        files = [fileA]  # , fileB, fileC]
        volumes = []
        for file in files:
            image = Chunk.from_h5(file, dataset_path='volumes/raw')
            label = Chunk.from_h5(
                file, dataset_path='volumes/labels/neuron_ids')

            image = image.astype(np.float32) / 255.
            ground_truth_volume = GroundTruthVolume(
                image, label, patch_size=patch_size_oversized)
            volumes.append(ground_truth_volume)

        self.training_volumes = volumes  # volumes[1:]
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
        patch.subject = self.transform(patch.subject)
        return patch

    def _prepare_transform(self):

        # Normalization
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))

        # Spatial
        elastic = tio.RandomElasticDeformation(locked_borders=2)
        flip = tio.RandomFlip(axes=(0, 1, 2))

        none = (0, 0.1)
        affine = tio.RandomAffine(
            center='image')
        spatial = tio.OneOf({
            affine: 0.2,
            # elastic: 0.2, this is a very slow transformation
            flip: 0.8,
        },
            p=0.25,
        )
        # Intensity
        intensity = tio.OneOf({
            tio.RandomBiasField(): 0.3,
            tio.RandomNoise(): 0.2,
            tio.RandomGamma(): 0.1,
            tio.RandomSpike(): 0.2,
            tio.RandomGhosting(): 0.1,

        },
            p=0.25,
        )

        # Crop down to true patch size
        crop = tio.Crop(bounds_parameters=self.over_sample//2)

        transforms = [rescale, intensity, spatial, crop]
        transforms = [rescale, affine, crop]

        self.transform = tio.Compose(transforms)


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
