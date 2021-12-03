from abc import ABC, abstractmethod
import random
from typing import Union

import numpy as np

from chunkflow.lib.bounding_boxes import BoundingBox, Cartesian
from .patch import Patch

from chunkflow.chunk import Chunk
from chunkflow.lib.synapses import Synapses


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
    def __init__(self, image: np.ndarray, label: np.ndarray,
            patch_size: Cartesian = Cartesian(256, 256, 256), 
            forbbiden_distance_to_boundary: tuple = None) -> None:
        """Image sample with ground truth annotations

        Args:
            image (np.ndarray): image normalized to 0-1
            label (np.ndarray): training label
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

        assert image.ndim == 3
        assert label.ndim >= 3
        assert image.shape == label.shape[-3:]

        
        if forbbiden_distance_to_boundary is None:
            forbbiden_distance_to_boundary = patch_size // 2 
        assert len(forbbiden_distance_to_boundary) == 3 or len(forbbiden_distance_to_boundary)==6

        for idx in range(3):
            # the center of random patch should not be too close to boundary
            # otherwise, the patch will go outside of the volume
            assert forbbiden_distance_to_boundary[idx] >= patch_size[idx] // 2
            assert forbbiden_distance_to_boundary[-idx] >= patch_size[-idx] // 2
        
        self.image = image
        self.label = label
        self.center_start = forbbiden_distance_to_boundary[:3]
        self.center_stop = tuple(s - d for s, d in zip(image.shape, forbbiden_distance_to_boundary[-3:]))
    
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
        # breakpoint()
        cz = random.randint(center_start[0], center_stop[0])
        cy = random.randint(center_start[1], center_stop[1])
        cx = random.randint(center_start[2], center_stop[2])
        return self.patch_from_center((cz, cy, cx)) 

    def patch_from_center(self, center: tuple):
        bz = center[0] - self.patch_size[-3] // 2
        by = center[1] - self.patch_size[-2] // 2
        bx = center[2] - self.patch_size[-1] // 2
        print('center: ', center)
        image_patch = self.image[...,
            bz : bz + self.patch_size[-3],
            by : by + self.patch_size[-2],
            bx : bx + self.patch_size[-1]
        ]
        label_patch = self.label[...,
            bz : bz + self.patch_size[-3],
            by : by + self.patch_size[-2],
            bx : bx + self.patch_size[-1]
        ]
        # if we do not copy here, the augmentation will change our 
        # image and label sample!
        image_patch = self._expand_to_5d(image_patch).copy()
        label_patch = self._expand_to_5d(label_patch).copy()
        return Patch(image_patch, label_patch)
    
    @property
    def sampling_weight(self):
        return int(np.product(tuple(e-b for b, e in zip(self.center_start, self.center_stop))))
    

class GroundTruthSampleWithPointAnnotation(GroundTruthSample):
    def __init__(self, image: np.ndarray, 
            annotation_points: np.ndarray,
            patch_size: Cartesian = Cartesian(256, 256, 256), 
            forbbiden_distance_to_boundary: tuple = None) -> None:
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
        label = self._points_to_label(image)
        super().__init__(
            image, label, 
            patch_size = patch_size,
            forbbiden_distance_to_boundary=forbbiden_distance_to_boundary
        )


    # it turns out that this sampling is biased to patches containing T-bar
    # the net will always try to find at least one T-bar in the input patch.
    # the result will have a low precision containing a lot of false positive prediction.
    # @property
    # def random_patch(self):
    #     point_num = self.annotation_points.shape[0]
    #     idx = random.randint(0, point_num-1)
    #     point = self.annotation_points[idx, :]
    #     center_start = tuple(p - d for p, d in zip(point, self.max_sampling_distance))
    #     center_stop = tuple(p + d for p, d in zip(point, self.max_sampling_distance))
    #     center_start = tuple(
    #         max(c1, c2) for c1, c2 in zip(center_start, self.center_start)
    #     )
    #     center_stop = tuple(
    #         min(c1, c2) for c1, c2 in zip(center_stop, self.center_stop)
    #     )
    #     for ct, cp in zip(center_start, center_stop):
    #         if ct >= cp:
    #             breakpoint()

    #     return self.random_patch_from_center_range(center_start, center_stop)

    @property
    def sampling_weight(self):
        # use number of annotated points as weight
        # to sample volume
        return self.annotation_points.shape[0]

    def _points_to_label(self, image: np.ndarray,
            expand_distance: int = 2) -> tuple:
        """transform point annotation to volumes

        Args:
            expand_distance (int): expand the point annotation to a cube. 
                This will help to got more positive voxels.
                The expansion should be small enough to ensure that all the voxels are inside T-bar.

        Returns:
            bin_presyn: binary label of annotated position.
        """
        # assert synapses['resolution'] == [8, 8, 8]
        label = np.zeros_like(image, dtype=np.float32)
        # adjust target to 0.05-0.95 for better regularization
        # the effect might be similar with Focal loss!
        label += 0.05
        for idx in range(self.annotation_points.shape[0]):
            coordinate = self.annotation_points[idx, :]
            label[...,
                coordinate[0]-expand_distance : coordinate[0]+expand_distance,
                coordinate[1]-expand_distance : coordinate[1]+expand_distance,
                coordinate[2]-expand_distance : coordinate[2]+expand_distance,
            ] = 0.95
        return label


class PostSynapseGroundTruth(AbstractGroundTruthSample):
    def __init__(self, 
            image: Chunk, 
            synapses: Synapses,
            patch_size: Cartesian = Cartesian(256, 256, 256), 
            point_expand: int = 2,
        ):
        """Ground Truth for post synapses

        Args:
            image (Chunk): image chunk covering the whole synapses
            synapses (Synapses): including both presynapses and postsynapses
            patch_size (Cartesian): image patch size covering the whole synapse
            point_expand (int): expand the point. range from 1 to half of patch size.
        """
        if isinstance(patch_size, tuple):
            patch_size = Cartesian(*patch_size)
        super().__init__(patch_size=patch_size)

        self.image = image
        self.synapses = synapses
        self.pre_index2post_indices = synapses.pre_index2post_indices
        self.point_expand = point_expand

    @property
    def random_patch(self):
        pre_index = random.randint(0, self.synapses.pre_num - 1)
        pre = self.synapses.pre[pre_index, :]
        
        post_indices = self.pre_index2post_indices[pre_index]

        bbox = BoundingBox.from_center(
            Cartesian(*pre), 
            extent=self.patch_size // 2
        )
        # shrink the right side to make the shape as even numbers
        # our network model only works with even number shape!
        bbox.adjust((0, 0, 0, -1, -1, -1))
        # Note that image is 4D array, the first dimension size is 1
        image = self.image.cutout(bbox)
        assert image.dtype == np.uint8
        image = image.astype(np.float32)
        image /= 255.
        # pre_label = np.zeros_like(image)
        # pre_label[
            
        #     pre[0] - self.point_expand : pre[0] + self.point_expand,
        #     pre[1] - self.point_expand : pre[1] + self.point_expand,
        #     pre[2] - self.point_expand : pre[2] + self.point_expand,
        # ] = 0.95

        # stack them together in the channel dimension
        # image = np.expand_dims(image, axis=0)
        # pre_label = np.expand_dims(pre_label, axis=0)
        # image = np.concatenate((image, pre_label), axis=0)

        label = np.zeros(image.shape, dtype=np.float32)
        label = Chunk(label, voxel_offset=image.voxel_offset)
        label += 0.05
        for post_index in post_indices:
            coord = self.synapses.post_coordinates[post_index, :]
            coord = coord - label.voxel_offset
            label[...,
                coord[0] - self.point_expand : coord[0] + self.point_expand,
                coord[1] - self.point_expand : coord[1] + self.point_expand,
                coord[2] - self.point_expand : coord[2] + self.point_expand,
            ] = 0.95
        assert np.any(label > 0.5)
        return Patch(image, label)