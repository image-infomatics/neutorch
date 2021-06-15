from typing import Tuple, Optional, List
import numpy as np
import random
import torch

from torchio.data.subject import Subject
from torchio.transforms.spatial_transform import SpatialTransform
from torchio.transforms.intensity_transform import IntensityTransform
from torchio.transforms.augmentation.random_transform import RandomTransform
from torchio import DATA


class DropAlongAxis(RandomTransform, SpatialTransform):

    def __init__(
            self,
            drop_amount: Tuple = (1, 2),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.drop_amount = random.randint(drop_amount[0], drop_amount[1])

    def apply_transform(self, sample: Subject) -> dict:
        shape = sample.get_first_image().shape
        z0 = shape[2]
        z = random.randint(1, z0-1)
        random_parameters_dict = {'drop_position':  z}
        random_parameters_dict = {
            'drop_position':  z, 'drop_amount': self.drop_amount}

        for image in self.get_images(sample):
            data = image.numpy()
            new_data = np.zeros(data.shape)
            new_data[..., :, :z, :] = data[..., :, :z, :]
            new_data[..., :, z:z0-self.drop_amount,
                     :] = data[..., :, z+self.drop_amount:, :]
            image[DATA] = torch.from_numpy(new_data)

        sample.add_transform(self, random_parameters_dict)
        return sample


class ZeroAlongAxis(RandomTransform, IntensityTransform):

    def __init__(
            self,
            drop_amount: Tuple = (1, 2),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.drop_amount = random.randint(drop_amount[0], drop_amount[1])

    def apply_transform(self, sample: Subject) -> dict:
        shape = sample.get_first_image().shape
        z0 = shape[2]
        z = random.randint(1, z0-1)
        random_parameters_dict = {
            'drop_position':  z, 'drop_amount': self.drop_amount}

        for image in self.get_images(sample):
            data = image.numpy()
            data[..., :, z:z+self.drop_amount,
                 :] = np.random.rand(*data[..., :, z:z+self.drop_amount, :].shape)
            image[DATA] = torch.from_numpy(data)

        sample.add_transform(self, random_parameters_dict)
        return sample


class DropSections(RandomTransform, IntensityTransform):

    def __init__(
            self,
            slices: Tuple = (1, 5),
            drop_amount: Tuple = (1, 20),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.slices = random.randint(slices[0], slices[1])
        self.drop_amount = random.randint(drop_amount[0], drop_amount[1])

    # drops a rectangular section from *slices* slides in the patch
    # of WxH within *drop_amount*
    # fills the section with noise [0,1]
    def apply_transform(self, sample: Subject) -> dict:

        shape = sample.get_first_image().shape
        z0 = shape[-3]
        y0 = shape[-2]
        x0 = shape[-1]

        for image in self.get_images(sample):
            data = image.numpy()

            for _ in range(self.slices):
                max = self.drop_amount
                h = random.randint(2, max)
                w = random.randint(2, max)
                z = random.randint(0, z0-1)
                y = random.randint(1, y0-max-1)
                x = random.randint(1, x0-max-1)
                data[..., z, y:y+h, x:x+w] = np.random.rand(1, h, w)
            image[DATA] = torch.from_numpy(data)

        return sample


class Transpose(RandomTransform, SpatialTransform):
    """
    Args:
        axes: List of indices of the spatial dimensions along which
            the image might be transposed.
        p: Probability that this transform will be applied.
    """

    def __init__(
            self,
            axes: List,
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.axes = np.array(axes)

    def apply_transform(self, sample: Subject) -> dict:

        shape = sample.get_first_image().shape
        dims = len(shape)
        axii = np.arange(dims)
        end_axii = axii[self.axes]
        random.shuffle(end_axii)
        axii[self.axes] = end_axii

        for image in self.get_images(sample):
            data = image.numpy()
            data = np.transpose(data, axii)
            image[DATA] = torch.from_numpy(data)

        return sample
