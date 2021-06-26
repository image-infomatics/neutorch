from abc import ABC, abstractmethod
import random
from typing import Union, Optional

import numpy as np
from scipy.ndimage.measurements import label
from .patch import Patch
from .patch import AffinityPatch


class AbstractGroundTruthVolume(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def random_patch(self):
        pass

    @property
    @abstractmethod
    def volume_sampling_weight(self):
        pass


class GroundTruthVolume(AbstractGroundTruthVolume):
    def __init__(self, image: np.ndarray, label: np.ndarray,
                 patch_size: Union[tuple, int],
                 forbbiden_distance_to_boundary: tuple = None,
                 lsd_label: Optional[np.ndarray] = None,
                 name: str = '') -> None:
        """Image volume with ground truth annotations

        Args:
            image (np.ndarray): image normalized to 0-1
            label (np.ndarray): training label
            patch_size (Union[tuple, int]): output patch size in real space
            forbbiden_distance_to_boundary (Union[tuple, int]): 
                the distance from patch center to volume boundary that is not allowed to sample 
                the order is z,y,x,-z,-y,-x
                if this is an integer, then all dimension is the same.
                if this is a tuple of three integers, the positive and negative is the same
                if this is a tuple of six integers, the positive and negative 
                direction is defined separately. 
            lsd_label Optional[np.ndarray]:
                an auxiliary label such as LSD which is treated similarly to normal label 
            name (str): name of volume
        """

        self.name = name
        assert image.ndim == 3
        assert label.ndim >= 3
        assert image.shape == label.shape[-3:]
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        else:
            assert len(patch_size) == 3

        if forbbiden_distance_to_boundary is None:
            forbbiden_distance_to_boundary = tuple(
                ps // 2 for ps in patch_size)
        assert len(forbbiden_distance_to_boundary) == 3 or len(
            forbbiden_distance_to_boundary) == 6

        for idx in range(3):
            # the center of random patch should not be too close to boundary
            # otherwise, the patch will go outside of the volume
            assert forbbiden_distance_to_boundary[idx] >= patch_size[idx] // 2
            assert forbbiden_distance_to_boundary[-idx] >= patch_size[-idx] // 2

        self.image = image
        self.label = label
        self.lsd_label = lsd_label
        self.patch_size = patch_size
        self.center_start = forbbiden_distance_to_boundary[:3]
        self.center_stop = tuple(
            s - d for s, d in zip(image.shape, forbbiden_distance_to_boundary[-3:]))

    @property
    def random_patch(self):
        patch = self.random_patch_from_center_range(
            self.center_start, self.center_stop)
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
        image_patch = self.image[...,
                                 bz: bz + self.patch_size[-3],
                                 by: by + self.patch_size[-2],
                                 bx: bx + self.patch_size[-1]
                                 ]
        label_patch = self.label[...,
                                 bz: bz + self.patch_size[-3],
                                 by: by + self.patch_size[-2],
                                 bx: bx + self.patch_size[-1]
                                 ]
        lsd_label = self.lsd_label
        if lsd_label is not None:
            lsd_label = self.lsd_label[...,
                                       bz: bz + self.patch_size[-3],
                                       by: by + self.patch_size[-2],
                                       bx: bx + self.patch_size[-1]
                                       ]

        return AffinityPatch(image_patch, label_patch, lsd_label=lsd_label)

    @property
    def volume_sampling_weight(self):
        return np.product(tuple(e-b for b, e in zip(self.center_start, self.center_stop)))


class GroundTruthVolumeWithPointAnnotation(GroundTruthVolume):
    def __init__(self, image: np.ndarray,
                 annotation_points: np.ndarray,
                 patch_size: Union[tuple, int],
                 forbbiden_distance_to_boundary: tuple = None) -> None:
        """Image volume with ground truth annotations

        Args:
            image (np.ndarray): image normalized to 0-1
            annotation_points (np.ndarray): point annotations with zyx order.
            patch_size (Union[tuple, int]): output patch size
            forbbiden_distance_to_boundary (tuple, optional): sample patches far away 
                from volume boundary. Defaults to None.
        """

        assert annotation_points.shape[1] == 3
        self.annotation_points = annotation_points
        label = self._points_to_label(image)
        super().__init__(
            image, label, patch_size,
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

    # @property
    # def volume_sampling_weight(self):
    #     # use number of annotated points as weight
    #     # to sample volume
    #     return self.annotation_points.shape[0]

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
                  coordinate[0]-expand_distance: coordinate[0]+expand_distance,
                  coordinate[1]-expand_distance: coordinate[1]+expand_distance,
                  coordinate[2]-expand_distance: coordinate[2]+expand_distance,
                  ] = 0.95
        return label
