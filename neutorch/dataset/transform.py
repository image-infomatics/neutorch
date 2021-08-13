from abc import ABC, abstractmethod
import random
from functools import lru_cache
from copy import deepcopy

import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import affine_transform

import cv2

from skimage.util import random_noise
from skimage.transform import swirl

from .patch import Patch


DEFAULT_PROBABILITY = .5


class AbstractTransform(ABC):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        assert probability > 0.
        assert probability <= 1.
        self.probability = probability

    @property
    def name(self):
        return self.__class__.__name__

    def is_invertible(self):
        return hasattr(self, 'invert')

    def __call__(self, patch: Patch):
        if random.random() < self.probability:
            self.transform(patch)
        else:
            # for spatial transform, we need to correct the size
            # to make sure that the final patch size is correct 
            if hasattr(self, 'shrink_size'):
                patch.accumulate_delayed_shrink_size(self.shrink_size) 

    @abstractmethod
    def transform(self, patch: Patch):
        """perform the real transform of image and label

        Args:
            patch (Patch): image and label
        """
        pass

    
class SpatialTransform(AbstractTransform):
    """Modify image voxel position and reinterprete."""
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    @abstractmethod
    def transform(self, patch: Patch):
        """transform the image and label together

        Args:
            patch (tuple): image and label pair
        """
        pass
    
    @property
    def shrink_size(self):
        """this transform might shrink the patch size.
        for example, droping a section will shrink the z axis.

        Return:
            shrink_size (tuple): z0,y0,x0,z1,y1,x1
        """
        return (0, 0, 0, 0, 0, 0)

class IntensityTransform(AbstractTransform):
    """change image intensity only"""
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    @abstractmethod
    def transform(self, patch: Patch):
        pass


class SectionTransform(AbstractTransform):
    """change a random section only."""
    def __init__(self, probability: float = DEFAULT_PROBABILITY ):
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
        shrink_size = np.zeros((6,), dtype=np.int64)
        for transform in transforms:
            if isinstance(transform, SpatialTransform):
                shrink_size += np.asarray(transform.shrink_size)
        self.shrink_size = tuple(x for x in shrink_size)

    def __call__(self, patch: Patch):
        for transform in self.transforms:
            transform(patch)
        # after the transformation, the stride of array
        # could be negative, and pytorch could not tranform
        # the array to Tensor. Copy can fix it.
        patch.image = patch.image.copy()
        patch.label = patch.label.copy()


class OneOf(AbstractTransform):
    def __init__(self, transforms: list, 
            probability: float = DEFAULT_PROBABILITY) -> None:
        super().__init__(probability=probability)
        assert len(transforms) > 1
        self.transforms = transforms

        shrink_size = np.zeros((6,), dtype=np.int64)
        for transform in transforms:
            if isinstance(transform, SpatialTransform):
                shrink_size += np.asarray(transform.shrink_size)
        self.shrink_size = tuple(x for x in shrink_size)

    def transform(self, patch: Patch):
        # select one of the transforms
        transform = random.choice(self.transforms)
        transform(patch)


class DropSection(SpatialTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        # since this transform really removes information
        # we do not delay the shrinking
        # make the first and last section missing is meaning less
        b0, c0, z0, y0, x0 = patch.shape
        z = random.randint(1, z0-1)
        image = np.zeros((b0, c0, z0-1, y0, x0), dtype=patch.image.dtype)
        label = np.zeros((b0, c0, z0-1, y0, x0), dtype=patch.label.dtype)
        image[..., :z, :, :] = patch.image[..., :z, :, :]
        label[..., :z, :, :] = patch.label[..., :z, :, :]
        image[..., z:, :, :] = patch.image[..., z+1:, :, :]
        label[..., z:, :, :] = patch.label[..., z+1:, :, :]
        patch.image = image
        patch.label = label

    @property
    def shrink_size(self):
        return (0, 0, 0, 1, 0, 0)


class BlackBox(IntensityTransform):
    def __init__(self,
            probability: float = DEFAULT_PROBABILITY,
            max_box_size: tuple = (8,8,8),
            max_box_num: int = 3):
        """make some black cubes in image patch

        Args:
            probability (float, optional): probability of triggering this augmentation. Defaults to 1..
            max_box_size (tuple, optional): maximum cube size. Defaults to (4,4,4).
            max_box_num (int, optional): maximum number of black boxes. Defaults to 2.
        """
        super().__init__(probability=probability)
        assert len(max_box_size) == 3
        self.max_box_size = max_box_size
        self.max_box_num = max_box_num

    def transform(self, patch: Patch):
        box_num = random.randint(1, self.max_box_num)
        for _ in range(box_num):
            box_size = tuple(random.randint(1, s) for s in self.max_box_size)
            start = tuple(random.randint(1, t-b) for t, b in zip(patch.shape[-3:], box_size))
            patch.image[
                ...,
                start[0] : start[0] + box_size[0],
                start[1] : start[1] + box_size[1],
                start[2] : start[2] + box_size[2],
            ] = 0

class NormalizeTo01(IntensityTransform):
    def __init__(self, probability: float = 1.):
        super().__init__(probability=probability)
        
    def transform(self, patch: Patch):
        if np.issubdtype(patch.image.dtype, np.uint8):
            patch.image = patch.image.astype(np.float32) / 255.

class AdjustBrightness(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            min_factor: float = 0.05,
            max_factor: float = 0.3):
        super().__init__(probability=probability)
        max_factor = np.clip(max_factor, 0, 2)
        self.min_factor = min_factor
        self.max_factor = max_factor
    
    def transform(self, patch: Patch):
        patch.image += random.uniform(-0.5, 0.5) * random.uniform(
            self.min_factor, self.max_factor)
        np.clip(patch.image, 0., 1., out=patch.image)

class AdjustContrast(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            factor_range: tuple = (0.05, 2.)):
        super().__init__(probability=probability)
        # factor_range = np.clip(factor_range, 0., 2.)
        self.factor_range = factor_range

    def transform(self, patch: Patch):
        #factor = 1 + random.uniform(-0.5, 0.5) * random.uniform(
        #    self.factor_range[0], self.factor_range[1])
        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        patch.image *= factor
        np.clip(patch.image, 0., 1., out=patch.image)


class Gamma(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        # gamma = random.random() * 2. - 1.
        gamma = random.uniform(-1., 1.)
        patch.image **= 2.** gamma


class GaussianBlur2D(IntensityTransform):
    def __init__(self, probability: float=DEFAULT_PROBABILITY, 
            sigma: float = 1.5):
        super().__init__(probability=probability)
        self.sigma = sigma

    def transform(self, patch: Patch):
        sigma = random.uniform(0.2, self.sigma)
        gaussian_filter(patch.image, sigma=sigma, output=patch.image)


class GaussianBlur3D(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            max_sigma: tuple = (1.5, 1.5, 1.5)):
        super().__init__(probability=probability)
        self.max_sigma = max_sigma

    def transform(self, patch: Patch):
        sigma = tuple(random.uniform(0.2, s) for s in self.max_sigma)
        gaussian_filter(patch.image, sigma=sigma, output=patch.image)


class Noise(IntensityTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY,
            mode: str='gaussian', max_variance: float = 0.02):
        super().__init__(probability=probability)
        self.mode = mode  
        self.max_variance = max_variance

    def transform(self, patch: Patch):
        variance = random.uniform(0.01, self.max_variance)
        random_noise(patch.image, mode=self.mode, var=variance)
        np.clip(patch.image, 0., 1., out=patch.image)


class Flip(SpatialTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        axis_num = random.randint(1, 3)
        axis = random.sample(range(3), axis_num)
        # the image and label is 5d
        # the first two axises are batch and channel
        axis5d = tuple(2+x for x in axis)
        patch.image = np.flip(patch.image, axis=axis5d)
        patch.label = np.flip(patch.label, axis=axis5d)

        shrink = list(patch.delayed_shrink_size)
        for ax in axis:
            # swap the axis to be shrinked
            shrink[3+ax], shrink[ax] = shrink[ax], shrink[3+ax]
        patch.delayed_shrink_size = tuple(shrink)


class Transpose(SpatialTransform):
    def __init__(self, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)

    def transform(self, patch: Patch):
        axis = [2,3,4]
        random.shuffle(axis)
        axis5d = (0, 1, *axis,)
        patch.image = np.transpose(patch.image, axis5d)
        patch.label = np.transpose(patch.label, axis5d)

        shrink = list(patch.delayed_shrink_size)
        for ax0, ax1 in enumerate(axis):
            ax1 -= 2
            # swap the axis to be shrinked
            shrink[ax0] = patch.delayed_shrink_size[ax1]
            shrink[3+ax0] = patch.delayed_shrink_size[3+ax1]
        patch.delayed_shrink_size = tuple(shrink) 

    
class MissAlignment(SpatialTransform):
    def __init__(self, probability: float=DEFAULT_PROBABILITY,
            max_displacement: int=2):
        """move part of volume alone x axis
        We'll alwasy select a position alone z axis, and move the bottom part alone X axis.
        By combining with transpose, flip and rotation, we can get other displacement automatically.

        Args:
            probability (float, optional): probability of this augmentation. Defaults to DEFAULT_PROBABILITY.
            max_displacement (int, optional): maximum displacement. Defaults to 2.
        """
        super().__init__(probability=probability)
        assert max_displacement > 0
        assert max_displacement < 8
        self.max_displacement = max_displacement

    def transform(self, patch: Patch):
        axis = random.randint(2, 4)
        displacement = random.randint(1, self.max_displacement)
        # random direction
        # no need to use random direction because we can combine with rotation and flipping
        # displacement *= random.choice([-1, 1])
        _,_, sz, sy, sx = patch.shape
        if axis == 2:
            zloc = random.randint(1, sz-1)
            patch.image[..., zloc:, 
                self.max_displacement : sy-self.max_displacement,
                self.max_displacement : sx-self.max_displacement,
                ] = patch.image[..., zloc:, 
                    self.max_displacement+displacement : sy+displacement-self.max_displacement,
                    self.max_displacement+displacement : sx+displacement-self.max_displacement,
                    ]
            patch.label[..., zloc:, 
                self.max_displacement : sy-self.max_displacement,
                self.max_displacement : sx-self.max_displacement,
                ] = patch.label[..., zloc:, 
                    self.max_displacement+displacement : sy+displacement-self.max_displacement,
                    self.max_displacement+displacement : sx+displacement-self.max_displacement,
                    ]
        elif axis == 3:
            yloc = random.randint(1, sy-1)
            
            # print('right side shape: ', patch.image[...,  self.max_displacement+displacement : sz+displacement-self.max_displacement,yloc:,self.max_displacement+displacement : sx+displacement-self.max_displacement,].shape)

            patch.image[..., 
                self.max_displacement : sz-self.max_displacement,
                yloc:, 
                self.max_displacement : sx-self.max_displacement,
                ] = patch.image[..., 
                    self.max_displacement+displacement : sz+displacement-self.max_displacement, 
                    yloc:, 
                    self.max_displacement+displacement : sx+displacement-self.max_displacement,
                    ]
            patch.label[..., 
                self.max_displacement : sz-self.max_displacement,
                yloc:,
                self.max_displacement : sx-self.max_displacement,
                ] = patch.label[..., 
                    self.max_displacement+displacement : sz+displacement-self.max_displacement,
                    yloc:, 
                    self.max_displacement+displacement : sx+displacement-self.max_displacement,
                    ]
        elif axis == 4:
            xloc = random.randint(1, sx-1)
            patch.image[..., 
                self.max_displacement : sz-self.max_displacement,
                self.max_displacement : sy-self.max_displacement,
                xloc:, 
                ] = patch.image[...,  
                    self.max_displacement+displacement : sz+displacement-self.max_displacement,
                    self.max_displacement+displacement : sy+displacement-self.max_displacement,
                    xloc:
                    ]
            patch.label[..., 
                self.max_displacement : sz-self.max_displacement,
                self.max_displacement : sy-self.max_displacement,
                xloc:,
                ] = patch.label[..., 
                    self.max_displacement+displacement : sz+displacement-self.max_displacement,
                    self.max_displacement+displacement : sy+displacement-self.max_displacement,
                    xloc:, 
                    ] 

        # only keep the central region  
        patch.shrink(self.shrink_size)       
    
    @property
    @lru_cache
    def shrink_size(self):
        # return (0, 0, 0, 0, 0, self.max_displacement)
        return (self.max_displacement,) * 6


class Perspective2D(SpatialTransform):
    def __init__(self, probability: float=DEFAULT_PROBABILITY,
            corner_ratio: float=0.2):
        """Warp image using Perspective transform

        Args:
            probability (float, optional): probability of this transformation. Defaults to DEFAULT_PROBABILITY.
            corner_ratio (float, optional): We split the 2D image to four equal size rectangles.
                For each axis in rectangle, we further divid it to four rectangles using this ratio.
                The rectangle containing the image corner was used as a sampling point region.
                This idea is inspired by this example:
                https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#perspective-transformation
                Defaults to 0.5.
        """
        super().__init__(probability=probability)
        self.corner_ratio = corner_ratio

    def transform(self, patch: Patch):
        # matrix = np.eye(3)
        # offset = tuple(-ps // 2 for ps in patch.shape[-3:] )
        for batch in range(patch.shape[0]):
            for channel in range(patch.shape[1]):
                for z in range(patch.shape[2]):
                    patch.image[batch,channel,z,...] = self._transform2d(
                        patch.image[batch, channel, z, ...], cv2.INTER_LINEAR
                    )
                    patch.image[batch,channel,z,...] = self._transform2d(
                        patch.image[batch, channel, z, ...], cv2.INTER_NEAREST
                    )
                
        patch.shrink(self.shrink_size)

    def _transform2d(self, arr: np.ndarray, interpolation: int):
        assert arr.ndim == 2
        corner_ratio = random.uniform(0.02, self.corner_ratio)
        # corner_ratio = self.corner_ratio
        sy, sx = arr.shape
        upper_left_point = [
            random.randint(0, round(sy*corner_ratio/2)), 
            random.randint(0, round(sx*corner_ratio/2))
        ]
        upper_right_point = [
            random.randint(0, round(sy*corner_ratio/2)),
            random.randint(sx-round(sx*corner_ratio/2), sx-1)
        ]
        lower_left_point = [
            random.randint(sy-round(sy*corner_ratio/2), sy-1),
            random.randint(0, round(sx*corner_ratio/2))
        ]
        lower_right_point = [
            random.randint(sy-round(sy*corner_ratio/2), sy-1),
            random.randint(sx-round(sx*corner_ratio/2), sx-1)
        ]
        pts1 = [
            upper_left_point, 
            upper_right_point, 
            lower_left_point, 
            lower_right_point
        ]
        # push the list order to get rotation effect
        # for example, push one position will rotate about 90 degrees
        # push_index = random.randint(0, 3)
        # if push_index > 0:
        #     tmp = deepcopy(pts1)
        #     pts1[push_index:] = tmp[:4-push_index]
        #     # the pushed out elements should be reversed
        #     pts1[:push_index] = tmp[4-push_index:][::-1]

        pts1 = np.asarray(pts1, dtype=np.float32)
        
        pts2 =np.float32([[0, 0], [0, sx], [sy, 0], [sy, sx]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(arr, M, (sy, sx), flags=interpolation)
        return dst


# class RotateScale(SpatialTransform):
#     def __init__(self, probability: float=DEFAULT_PROBABILITY,
#             max_scaling: float=1.3):
#         super().__init__(probability=probability)
#         raise NotImplementedError('this augmentation is not working correctly yet. The image and label could have patchy effect.We are not sure why.')
#         self.max_scaling = max_scaling

#     def transform(self, patch: Patch):
#         # because we do not know the rotation angle
#         # we should apply the shrinking first
#         patch.apply_delayed_shrink_size()

#         # if the rotation is close to diagnal, for example 45 degree
#         # the label could be outside the volume and be black!
#         # angle = random.choice([0, 90, 180, -90, -180]) + random.randint(-5, 5)
#         angle = random.randint(0, 180)
#         scale = random.uniform(1.1, self.max_scaling)
#         center = patch.center[-2:]
#         mat = cv2.getRotationMatrix2D( center, angle, scale )
        
#         for batch in range(patch.shape[0]):
#             for channel in range(patch.shape[1]):
#                 for z in range(patch.shape[2]):
#                     patch.image[batch, channel, z, ...] = cv2.warpAffine(
#                         patch.image[batch, channel, z, ...],
#                         mat, patch.shape[-2:], flags=cv2.INTER_LINEAR
#                     ) 
#                     patch.label[batch, channel, z, ...] = cv2.warpAffine(
#                         patch.label[batch, channel, z, ...],
#                         mat, patch.shape[-2:], flags=cv2.INTER_NEAREST
#                     ) 


class Swirl(SpatialTransform):
    def __init__(self, max_rotation: int = 5, max_strength: int = 3, probability: float = DEFAULT_PROBABILITY):
        super().__init__(probability=probability)
        self.max_strength = max_strength
        self.max_rotation = max_rotation
    
    def transform(self, patch: Patch):
        for z in range(patch.shape[-3]):
            patch.image[..., z, :, :] = swirl(
                patch.image[..., z, :, :],
                rotation=random.randint(1, self.max_rotation),
                strength=random.randint(1, self.max_strength),
                radius = (patch.shape[-1] + patch.shape[-2]) // 4,
            )

class LabelAsTarget()

class Label2Affinity