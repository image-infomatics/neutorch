import random
from typing import Union

import numpy as np


class GroundTruthVolume(object):
    def __init__(self, image: np.ndarray, 
            patch_size: Union[tuple, int], 
            target: np.ndarray = None,
            forbbiden_distance_to_boundary: tuple = None) -> None:
        """Image volume with ground truth annotations

        Args:
            image (np.ndarray): image normalized to 0-1
            target (np.ndarray): training target
            patch_size (Union[tuple, int]): output patch size
        """
        assert image.ndim == 3
        if target is not None:
            assert target.ndim >= 3
        assert image.shape == target.shape[-3:]
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        else:
            assert len(patch_size) == 3
        
        if forbbiden_distance_to_boundary is None:
            forbbiden_distance_to_boundary = tuple(ps // 2 for ps in patch_size)
        assert len(forbbiden_distance_to_boundary) == 3 or len(forbbiden_distance_to_boundary)==6
        
        for idx in range(3):
            # the center of random patch should not be too close to boundary
            # otherwise, the patch will go outside of the volume
            assert forbbiden_distance_to_boundary[idx] >= patch_size[idx] // 2
            assert forbbiden_distance_to_boundary[-idx] >= patch_size[-idx] // 2
        
        self.image = image
        self.target = target
        self.patch_size = patch_size
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
        return self.random_patch_from_center_range(self.center_start, self.center_stop)
    
    def random_patch_from_center_range(self, center_start: tuple, center_stop: tuple):
        breakpoint()
        cz = random.randrange(center_start[0], center_stop[0])
        cy = random.randrange(center_start[1], center_stop[1])
        cx = random.randrange(center_start[2], center_stop[2])
        return self.patch_from_center((cz, cy, cx)) 

    def patch_from_center(self, center: tuple):
        bz = center[0] - self.patch_size[0] // 2
        by = center[1] - self.patch_size[1] // 2
        bx = center[2] - self.patch_size[2] // 2
        image_patch = self.image[
            bz : bz + self.patch_size[0],
            by : by + self.patch_size[1],
            bx : bx + self.patch_size[2]
        ]
        target_patch = self.target[...,
            bz : bz + self.patch_size[0],
            by : by + self.patch_size[1],
            bx : bx + self.patch_size[2]
        ]
        image_patch = self._expand_to_5d(image_patch)
        target_patch = self._expand_to_5d(target_patch)
        return image_patch, target_patch
    
    @property
    def volume_sampling_weight(self):
        return np.product(tuple(e-b for b, e in zip(self.center_start, self.center_stop)))



class GroundTruthVolumeWithPointAnnotation(GroundTruthVolume):
    def __init__(self, image: np.ndarray, 
            patch_size: Union[tuple, int], 
            target: np.ndarray = None,
            forbbiden_distance_to_boundary: tuple = None,
            annotation_points: np.ndarray = None,
            max_sampling_distance: Union[int, tuple] = None) -> None:
        """Image volume with ground truth annotations

        Args:
            image (np.ndarray): image normalized to 0-1
            patch_size (Union[tuple, int]): output patch size
            target (np.ndarray): training target
            forbbiden_distance_to_boundary (tuple, optional): sample patches far away 
                from volume boundary. Defaults to None.
            annotation_points (np.ndarray, optional): point annotations. Defaults to None.
            max_sampling_distance (int, tuple): maximum distance from the annotated point 
                to the center of random patch.
        """
        super().__init__(
            image, patch_size,
            forbbiden_distance_to_boundary=forbbiden_distance_to_boundary
        )
        
        if max_sampling_distance is None:
            max_sampling_distance = tuple(ps // 2 for ps in patch_size)
        if isinstance(max_sampling_distance, int):
            max_sampling_distance = (max_sampling_distance, ) * 3
        for idx in range(3):
            max_sampling_distance[idx] <= patch_size[idx] // 2

        if annotation_points is not None:
            assert annotation_points.shape[1] == 3
        self.annotation_points = annotation_points
        self.max_sampling_distance = max_sampling_distance
        if target is None:
            self._points_to_target()

    @property
    def random_patch(self):
        point_num = self.annotation_points.shape[0]
        idx = random.randrange(point_num)
        point = self.annotation_points[idx, :]
        center_start = tuple(p - d for p, d in zip(point, self.max_sampling_distance))
        center_stop = tuple(p + d for p, d in zip(point, self.max_sampling_distance))
        center_start = tuple(
            max(c1, c2) for c1, c2 in zip(center_start, self.center_start)
        )
        center_stop = tuple(
            min(c1, c2) for c1, c2 in zip(center_stop, self.center_stop)
        )
        return self.random_patch_from_center_range(center_start, center_stop)

    @property
    def volume_sampling_weight(self):
        # use number of annotated points as weight
        # to sample volume
        return self.annotation_points.shape[0]

    def _points_to_target(self, expand_distance: int = 2) -> tuple:
        """transform point annotation to volumes

        Args:
            expand_distance (int): expand the point annotation to a cube. 
                This will help to got more positive voxels.
                The expansion should be small enough to ensure that all the voxels are inside T-bar.

        Returns:
            bin_presyn: binary label of annotated position.
        """
        # assert synapses['resolution'] == [8, 8, 8]
        self.target = np.zeros_like(self.image.array, dtype=np.float32)
        voxel_offset = np.asarray(self.image.voxel_offset, dtype=np.int64)
        for coordinate in self.annotation_points:
            # transform coordinate from xyz order to zyx
            coordinate = coordinate[::-1]
            coordinate = np.asarray(coordinate, dtype=np.int64)
            coordinate -= voxel_offset
            self.target[
                coordinate[0]-expand_distance : coordinate[0]+expand_distance,
                coordinate[1]-expand_distance : coordinate[1]+expand_distance,
                coordinate[2]-expand_distance : coordinate[2]+expand_distance,
            ] = 1.
        return

