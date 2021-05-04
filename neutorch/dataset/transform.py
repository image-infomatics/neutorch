from abc import ABC, abstractmethod
import random

import numpy as np

from .patch import Patch


class AbstractTransform(ABC):
    def __init__(self, probability: float = 1.):
        assert probability > 0.
        assert probability <= 1.
        self.probability = probability

    @property
    def name(self):
        return self.__class__.__name__

    def is_invertible(self):
        return hasattr(self, 'invert')

    def __call__(self, patch):
        if random.random() < self.probability:
            return self.transform(patch)
        else:
            # do nothing if outside of probability 
            return patch

    @abstractmethod
    def transform(self, patch: tuple):
        """perform the real transform of image and label

        Args:
            patch (tuple): image and label
        """
        pass

    
class SpatialTransform(AbstractTransform):
    """Modify image voxel position and reinterprete."""
    def __init__(self, probability: float = 1.):
        super().__init__(probability=probability)

    @abstractmethod
    def transform(self, patch: Patch):
        """transform the image and label together

        Args:
            patch (tuple): image and label pair
        """
        pass
    
    @property
    @abstractmethod
    def shrinking_size(self):
        """this transform might shrink the patch size.
        for example, droping a section will shrink the z axis.

        Return:
            shrinking_size (tuple): z0,y0,x0,z1,y1,x1
        """
        return (0, 0, 0, 0, 0, 0)

class IntensityTransform(AbstractTransform):
    """change image intensity only"""
    def __init__(self, probability: float = 1.):
        super().__init__(probability=probability)

    @abstractmethod
    def transform(self, patch: tuple):
        pass


class SectionTransform(AbstractTransform):
    """change a random section only."""
    def __init__(self, probability: float = 1. ):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        self.selected_axis = random.randrange(3)
        self.selected_idx = random.randrange(
            patch.image.shape[self.selected_axis]
        )
        patch = self.transform_section(patch)
        return patch
    
    @abstractmethod
    def transform_section(self, patch: Patch):
        pass

class Compose(object):
    def __init__(self, transforms: list):
        """compose multiple transforms

        Args:
            transforms (list): list of transform instances
        """
        self.transforms = transforms
        shrinking_size = np.zeros((6,), dtype=np.int64)
        for transform in transforms:
            if isinstance(transform, SpatialTransform):
                shrinking_size += np.asarray(transform.shrinking_size)
        self.shrinking_size = tuple(x for x in shrinking_size)

    def __call__(self, patch: tuple):
        for transform in self.transforms:
            patch = transform(patch)
        return patch



class DropSection(SpatialTransform):
    def __init__(self, probability: float = 1.):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        breakpoint()
        # since this transform really removes information
        # we do not delay the shrinking
        # make the first and last section missing is meaning less
        b0, c0, z0, y0, x0 = patch.image.shape
        z = random.randrange(1, z0-1)
        image = np.zeros((b0, c0, z0-1, y0, x0), dtype=patch.image.dtype)
        label = np.zeros((b0, c0, z0-1, y0, x0), dtype=patch.label.dtype)
        image[..., :z, :, :] = patch.image[..., :z, :, :]
        label[..., :z, :, :] = patch.label[..., :z, :, :]
        image[..., z:, :, :] = patch.image[..., z+1, :, :]
        label[..., z:, :, :] = patch.label[..., z+1, :, :]

    @property
    def shrinking_size(self):
        return (0, 0, 0, 1, 0, 0)