import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.lib.synapses import Synapses

from .patch import Patch


class AbstractGroundTruthSample(ABC):
    def __init__(self, patch_size: Cartesian = Cartesian(256, 256, 256)):

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        else:
            assert len(patch_size) == 3
        self.patch_size = patch_size

    @property
    @abstractmethod
    def random_patch(self):
        pass

    @property
    def sampling_weight(self) -> int:
        """the weight to sample 

        Returns:
            int: the relative weight. The default is 1, 
                so all the sample have the same weight.
        """
        return 1 


class GroundTruthSample(AbstractGroundTruthSample):
    def __init__(self, 
            images: List[Chunk], target: np.ndarray,
            patch_size: Cartesian = Cartesian(128, 128, 128), 
            forbidden_distance_to_boundary: tuple = None) -> None:
        """Image sample with ground truth annotations

        Args:
            images (List[Chunk]): different versions of image chunks normalized to 0-1
            target (np.ndarray): training target
            patch_size (Cartesian): output patch size
            forbbiden_distance_to_boundary (Union[tuple, int]): 
                the distance from patch center to sample boundary that is not allowed to sample 
                the order is z,y,x,-z,-y,-x
                if this is an integer, then all dimension is the same.
                if this is a tuple of three integers, the positive and negative is the same
                if this is a tuple of six integers, the positive and negative 
                direction is defined separately. 
        """
        super().__init__(patch_size=patch_size)
        assert len(images) > 0
        assert images[0].ndim == 3
        assert target.ndim >= 3
        assert images[0].shape == target.shape[-3:]
        assert isinstance(patch_size, Cartesian)

        
        if forbidden_distance_to_boundary is None:
            forbidden_distance_to_boundary = patch_size // 2 
        assert len(forbidden_distance_to_boundary) == 3 or len(forbidden_distance_to_boundary)==6

        for idx in range(3):
            # the center of random patch should not be too close to boundary
            # otherwise, the patch will go outside of the volume
            assert forbidden_distance_to_boundary[idx] >= patch_size[idx] // 2
            assert forbidden_distance_to_boundary[-idx] >= patch_size[-idx] // 2
        
        self.images = images
        self.target = target
        self.center_start = forbidden_distance_to_boundary[:3]
        self.center_stop = tuple(s - d for s, d in zip(
            images[0].shape, forbidden_distance_to_boundary[-3:]))
    
    def _expand_to_5d(self, array: np.ndarray):
        if array.ndim == 3:
            return np.expand_dims(array, axis=(0, 1))
        elif array.ndim == 4:
            return np.expand_dims(array, axis=0)
        elif array.ndim == 5:
            return array
        else:
            raise ValueError('only support 3 to 5 dimensional array.')

    @property
    def random_patch(self):
        patch = self.random_patch_from_center_range(self.center_start, self.center_stop)
        return patch
    
    def random_patch_from_center_range(self, center_start: tuple, center_stop: tuple):
        cz = random.randint(center_start[0], center_stop[0])
        cy = random.randint(center_start[1], center_stop[1])
        cx = random.randint(center_start[2], center_stop[2])
        return self.patch_from_center((cz, cy, cx)) 

    def patch_from_center(self, center: tuple):
        bz = center[0] - self.patch_size[-3] // 2
        by = center[1] - self.patch_size[-2] // 2
        bx = center[2] - self.patch_size[-1] // 2

        image = random.choice(self.images).array
        image_patch = image[...,
            bz : bz + self.patch_size[-3],
            by : by + self.patch_size[-2],
            bx : bx + self.patch_size[-1]
        ]
        target_patch = self.target[...,
            bz : bz + self.patch_size[-3],
            by : by + self.patch_size[-2],
            bx : bx + self.patch_size[-1]
        ]
        # if we do not copy here, the augmentation will change our 
        # image and target sample!
        image_patch = self._expand_to_5d(image_patch).copy()
        target_patch = self._expand_to_5d(target_patch).copy()
        return Patch(image_patch, target_patch)
    
    @property
    def sampling_weight(self):
        return int(np.product(tuple(e-b for b, e in zip(
            self.center_start, self.center_stop))))
    

class GroundTruthSampleWithPointAnnotation(GroundTruthSample):
    def __init__(self, 
            images: List[Chunk], 
            annotation_points: np.ndarray,
            patch_size: Cartesian = Cartesian(256, 256, 256), 
            forbidden_distance_to_boundary: tuple = None) -> None:
        """Image sample with ground truth annotations

        Args:
            image (np.ndarray): image normalized to 0-1
            annotation_points (np.ndarray): point annotations with zyx order.
            patch_size (Cartesian): output patch size
            forbbiden_distance_to_boundary (tuple, optional): sample patches far away 
                from sample boundary. Defaults to None.
        """

        assert annotation_points.shape[1] == 3
        self.annotation_points = annotation_points
        target = np.zeros_like(images[0].array, dtype=np.float32)
        target = self._points_to_target(target)
        super().__init__(
            images, target, 
            patch_size = patch_size,
            forbbiden_distance_to_boundary=forbidden_distance_to_boundary
        )

    @property
    def sampling_weight(self):
        """use number of annotated points as weight to sample volume."""
        return int(self.annotation_points.shape[0])

    def _points_to_target(self, target: np.ndarray,
            expand_distance: int = 2) -> tuple:
        """transform point annotation to volumes

        Args:
            expand_distance (int): expand the point annotation to a cube. 
                This will help to got more positive voxels.
                The expansion should be small enough to ensure that all the voxels are inside T-bar.

        Returns:
            bin_presyn: binary target of annotated position.
        """
        # assert synapses['resolution'] == [8, 8, 8]
        # target = np.zeros_like(image, dtype=np.float32)
        # adjust target to 0.05-0.95 for better regularization
        # the effect might be similar with Focal loss!
        target += 0.05
        for idx in range(self.annotation_points.shape[0]):
            coordinate = self.annotation_points[idx, :]
            target[...,
                coordinate[0]-expand_distance : coordinate[0]+expand_distance,
                coordinate[1]-expand_distance : coordinate[1]+expand_distance,
                coordinate[2]-expand_distance : coordinate[2]+expand_distance,
            ] = 0.95
        assert np.any(target > 0.5)
        return target


class PostSynapseGroundTruth(AbstractGroundTruthSample):
    def __init__(self,
            synapses: Synapses,
            images: List[Chunk], 
            patch_size: Cartesian = Cartesian(256, 256, 256), 
            point_expand: int = 2,
        ):
        """Ground Truth for post synapses

        Args:
            synapses (Synapses): including both presynapses and postsynapses
            images (List[Chunk]): several image chunk versions covering the whole synapses
            patch_size (Cartesian): image patch size covering the whole synapse
            point_expand (int): expand the point. range from 1 to half of patch size.
        """
        if not isinstance(patch_size, Cartesian):
            patch_size = Cartesian.from_collection(patch_size)
        super().__init__(patch_size=patch_size)

        self.images = images
        self.synapses = synapses
        self.pre_index2post_indices = synapses.pre_index2post_indices
        self.point_expand = point_expand

    @property
    def random_patch(self):
        pre_index = random.randint(0, self.synapses.pre_num - 1)
        pre = self.synapses.pre[pre_index, :]
        
        post_indices = self.pre_index2post_indices[pre_index]
        assert len(post_indices) > 0

        bbox = BoundingBox.from_center(
            Cartesian(*pre), 
            extent=self.patch_size // 2
        )

        image = random.choice(self.images)
        
        # Note that image is 4D array, the first dimension size is 1
        image = image.cutout(bbox)
        assert image.dtype == np.uint8
        image = image.astype(np.float32)
        image /= 255.
        # pre_target = np.zeros_like(image)
        # pre_target[
            
        #     pre[0] - self.point_expand : pre[0] + self.point_expand,
        #     pre[1] - self.point_expand : pre[1] + self.point_expand,
        #     pre[2] - self.point_expand : pre[2] + self.point_expand,
        # ] = 0.95

        # stack them together in the channel dimension
        # image = np.expand_dims(image, axis=0)
        # pre_target = np.expand_dims(pre_target, axis=0)
        # image = np.concatenate((image, pre_target), axis=0)

        target = np.zeros(image.shape, dtype=np.float32)
        target = Chunk(target, voxel_offset=image.voxel_offset)
        target += 0.05
        for post_index in post_indices:
            assert post_index < self.synapses.post_num
            coord = self.synapses.post_coordinates[post_index, :]
            coord = coord - target.voxel_offset
            target[...,
                coord[0] - self.point_expand : coord[0] + self.point_expand,
                coord[1] - self.point_expand : coord[1] + self.point_expand,
                coord[2] - self.point_expand : coord[2] + self.point_expand,
            ] = 0.95
        assert np.any(target > 0.5)
        return Patch(image, target)