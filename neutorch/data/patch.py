from functools import cached_property

import numpy as np
import torch

from chunkflow.lib.cartesian_coordinate import Cartesian
from chunkflow.chunk import Chunk

def expand_to_4d(arr: np.ndarray):
    if arr.ndim == 4:
        return arr
    elif arr.ndim == 3:
        return np.expand_dims(arr, axis=0)
    else:
        raise ValueError(f'only support array dimension of 3,4,5, but get {arr.ndim}')

class Patch(object):
    def __init__(self, image: Chunk, target: Chunk,
            mask: Chunk = None):
        """A patch of volume containing both image and target. 

        Args:
            image (Chunk): image
            target (Chunk): target
        """
        assert image.shape[-3:] == target.shape[-3:], \
            f'image shape: {image.shape}, target shape: {target.shape}'
        assert image.voxel_offset == target.voxel_offset
        if mask is not None:
            mask.shape == target.shape
            assert mask.ndim == 3
            assert mask.voxel_offset == image.voxel_offset
        
        image.array = expand_to_4d(image.array)
        target.array = expand_to_4d(target.array)
        
        self.image = image
        self.target = target  
        self.mask = mask

    @cached_property
    def has_mask(self):
        return self.mask is not None

    def shrink(self, size: tuple):
        self.image.shrink(size)
        self.target.shrink(size)
        if self.has_mask:
            self.mask.shrink(size)
            
    @property
    def shape(self):
        return self.image.shape

    @property
    def ndim(self):
        return self.image.ndim

    @cached_property
    def center(self):
        return Cartesian.from_collection(self.shape[-3:]) // 2

    def normalize(self):
        def _normalize(arr):
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32)
                arr /= 255.
            elif arr.dtype == torch.uint8:
                arr = arr.type(torch.float32)
                arr /= 255.
            return arr
        self.image.array = _normalize(self.image.array)
        self.target.array = _normalize(self.target.array)



def collate_batch(batch):
    patch_list = []
    for patch in batch:
        patch_list.append(patch)
    return patch