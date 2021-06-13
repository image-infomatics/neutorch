from typing import Tuple, Optional, List
import numpy as np
import random

from torchio.data.subject import Subject
from torchio.transforms.spatial_transform import SpatialTransform
from torchio.transforms.intensity_transform import IntensityTransform
from torchio.transforms.augmentation.random_transform import RandomTransform
from torchio import DATA


class DropAlongAxis(RandomTransform, SpatialTransform):

    def __init__(
            self,
            drop_amount: Tuple = (1, 3),
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
            image[DATA] = new_data

        sample.add_transform(self, random_parameters_dict)
        return sample


class ZeroAlongAxis(RandomTransform, IntensityTransform):

    def __init__(
            self,
            drop_amount: Tuple = (1, 3),
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
            data[..., :, z:z+self.drop_amount, :] = 0
            image[DATA] = data

        sample.add_transform(self, random_parameters_dict)
        return sample
