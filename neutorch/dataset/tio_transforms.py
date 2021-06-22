import cv2
from typing import Tuple, Optional, List
import numpy as np
import random
import torch

from torchio.data.subject import Subject
from torchio.transforms.spatial_transform import SpatialTransform
from torchio.transforms.intensity_transform import IntensityTransform
from torchio.transforms.augmentation.random_transform import RandomTransform
from torchio import DATA


class SlipAlongAxis(RandomTransform, SpatialTransform):

    def __init__(
            self,
            drop_amount: Tuple = (1, 6),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.drop_amount = drop_amount

    def apply_transform(self, sample: Subject) -> dict:
        drop = random.randint(self.drop_amount[0], self.drop_amount[1])
        shape = sample.get_first_image().shape
        z0 = shape[2]
        z = random.randint(1, z0-1)

        for image in self.get_images(sample):
            data = image.numpy()
            new_data = np.zeros(data.shape)
            new_data[..., :, :z, :] = data[..., :, :z, :]
            new_data[..., :, z:z0-drop,
                     :] = data[..., :, z+drop:, :]
            image[DATA] = torch.from_numpy(new_data)

        return sample


class ZeroAlongAxis(RandomTransform, IntensityTransform):

    def __init__(
            self,
            drop_amount: Tuple = (1, 6),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.drop_amount = drop_amount

    def apply_transform(self, sample: Subject) -> dict:
        drop = random.randint(self.drop_amount[0], self.drop_amount[1])
        shape = sample.get_first_image().shape
        z0 = shape[2]
        z = random.randint(1, z0-1)

        for image in self.get_images(sample):
            data = image.numpy()
            data[..., :, z:z+drop,
                 :] = np.random.rand(*data[..., :, z:z+drop, :].shape)
            image[DATA] = torch.from_numpy(data)

        return sample


class DropSections(RandomTransform, IntensityTransform):

    def __init__(
            self,
            slices: Tuple = (1, 8),
            drop_amount: Tuple = (1, 25),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.slices = slices
        self.drop_amount = drop_amount

    # drops a rectangular section from *slices* slides in the patch
    # of WxH within *drop_amount*
    # fills the section with noise [0,1]
    def apply_transform(self, sample: Subject) -> dict:

        slice_range = random.randint(self.slices[0], self.slices[1])

        shape = sample.get_first_image().shape
        z0 = shape[-3]
        y0 = shape[-2]
        x0 = shape[-1]

        for image in self.get_images(sample):
            data = image.numpy()

            for _ in range(slice_range):
                min = self.drop_amount[0]
                max = self.drop_amount[1]
                h = random.randint(min, max)
                w = random.randint(min, max)
                z = random.randint(0, z0-1)
                y = random.randint(1, y0-h-1)
                x = random.randint(1, x0-w-1)
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


class ApplyIntensityAlongZ(RandomTransform, IntensityTransform):

    def __init__(
            self,
            transform: IntensityTransform,
            slice_range: Tuple = (1, 8),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.slice_range = slice_range
        self.transform = transform

    def apply_transform(self, sample: Subject) -> dict:
        num_slices = random.randint(self.slice_range[0], self.slice_range[1])
        shape = sample.get_first_image().shape
        z0 = shape[-3]
        z_s = [random.randint(0, z0-1) for _ in range(num_slices)]

        for image in self.get_images(sample):
            data = image.numpy()
            for z in z_s:
                slice = data[..., z, :, :]
                transformed_slice = self.transform(
                    np.expand_dims(slice, axis=0))
                data[..., z, :, :] = np.squeeze(transformed_slice, axis=0)
            image[DATA] = torch.from_numpy(data)

        return sample


class DropZSlices(RandomTransform, IntensityTransform):

    def __init__(
            self,
            slice_range: Tuple = (1, 1),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.slice_range = slice_range

    def apply_transform(self, sample: Subject) -> dict:
        num_slices = random.randint(self.slice_range[0], self.slice_range[1])
        shape = sample.get_first_image().shape
        z0 = shape[-3]
        z_s = [random.randint(0, z0-1) for _ in range(num_slices)]

        for image in self.get_images(sample):
            data = image.numpy()
            for z in z_s:
                data[..., z, :, :] = np.random.rand(1, shape[-2], shape[-1])
            image[DATA] = torch.from_numpy(data)

        return sample


class Perspective2D(RandomTransform, SpatialTransform):

    def __init__(
            self,
            corner_ratio: float = 0.8,
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
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
        super().__init__(p=p, seed=seed, keys=keys)
        self.corner_ratio = corner_ratio

    def apply_transform(self, sample: Subject) -> dict:
        # get just z,y,x dims which should be same for all images and labels
        image = sample.get_first_image()
        (_, z0,  y0, x0) = image.shape
        R = self.gen_random_transformtion(y0, x0)
        for image in self.get_images(sample):
            # get channel which is different for some labels
            c0 = image.shape[0]
            data = image.numpy()
            for c in range(c0):
                for z in range(z0):
                    slice = np.squeeze(data[..., c, z, :, :])
                    transformed_slice = self._transform2d(
                        slice, R, cv2.INTER_NEAREST)
                    data[..., c, z, :, :] = transformed_slice

            image[DATA] = torch.from_numpy(data)

        return sample

    def gen_random_transformtion(self, sy: int, sx: int):
        corner_ratio = random.uniform(0.05, self.corner_ratio)
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
        return pts1

    def _transform2d(self, arr: np.ndarray, pts1: np.ndarray, interpolation: int):
        sy, sx = arr.shape
        assert arr.ndim == 2
        # push the list order to get rotation effect
        # for example, push one position will rotate about 90 degrees
        # push_index = random.randint(0, 3)
        # if push_index > 0:
        #     tmp = deepcopy(pts1)
        #     pts1[push_index:] = tmp[:4-push_index]
        #     # the pushed out elements should be reversed
        #     pts1[:push_index] = tmp[4-push_index:][::-1]

        pts1 = np.asarray(pts1, dtype=np.float32)
        pts2 = np.float32([[0, 0], [0, sx], [sy, 0], [sy, sx]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(arr, M, (sy, sx), flags=interpolation)
        return dst


class Clip(IntensityTransform):

    def __init__(
            self,
            min_max: Tuple = (0, 1),
            keys: Optional[List[str]] = None,
    ):
        super().__init__(keys=keys)
        self.min_max = min_max

    def apply_transform(self, sample: Subject) -> dict:
        for image in self.get_images(sample):
            data = image.numpy()
            clipped = np.clip(data, self.min_max[0], self.min_max[1])
            image[DATA] = torch.from_numpy(clipped)

        return sample


class Brightness(RandomTransform, IntensityTransform):

    def __init__(
            self,
            amount: Tuple = (-0.3, 0.3),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
    ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.amount = amount

    def apply_transform(self, sample: Subject) -> dict:

        for image in self.get_images(sample):
            data = image.numpy()
            amt = random.uniform(self.amount[0],  self.amount[1])
            data += amt
            data = np.clip(data, 0.0, 1.0)
            image[DATA] = torch.from_numpy(data)

        return sample
