from functools import lru_cache
import numpy as np

# from torch import tensor, device
import torch
# torch.multiprocessing.set_start_method('spawn')

from chunkflow.lib.cartesian_coordinate import Cartesian


class Patch(object):
    def __init__(self, image: np.ndarray, label: np.ndarray,
        delayed_shrink_size: tuple = (0, 0, 0, 0, 0, 0)):
        """A patch of volume containing both image and label

        Args:
            image (np.ndarray): image
            label (np.ndarray): label
            delayed_shrink_size (tuple): delayed shrinking size.
                some transform might shrink the patch size, but we
                would like to delay it to keep a little bit more 
                information. For exampling, warping the image will
                make boundary some black region.
        """
        assert image.shape == label.shape

        image = self._expand_to_5d(image)
        label = self._expand_to_5d(label)

        self.image = image
        self.label = label
        self.delayed_shrink_size = delayed_shrink_size

    def _expand_to_5d(self, arr: np.ndarray):
        if arr.ndim == 4:
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3:
            arr = np.expand_dims(arr, axis=(0,1))
        elif arr.ndim == 5:
            pass
        else:
            raise ValueError(f'only support array dimension of 3,4,5, but get {arr.ndim}')
        return arr

    def accumulate_delayed_shrink_size(self, shrink_size: tuple):
        self.delayed_shrink_size = tuple(
            d + s for d, s in zip(self.delayed_shrink_size, shrink_size))

    def apply_delayed_shrink_size(self):
        if self.delayed_shrink_size is None or not np.any(self.delayed_shrink_size):
            return
        # elif len(self.delayed_shrink_size) == 3:
        #     margin1 = tuple(s // 2 for s in self.delayed_shrink_size)
        #     margin2 = tuple(s - m1 for s, m1 in zip(self.delayed_shrink_size, margin1))
        # elif len(self.delayed_shrink_size) == 6:
        #     margin
        self.shrink(self.delayed_shrink_size)
        
        # reset the shrink size to 0
        self.delayed_shrink_size = (0,) * 6

    def shrink(self, size: tuple):
        assert len(size) == 6
        _, _, z, y, x = self.shape
        self.image = self.image[
            ...,
            size[0]:z-size[3],
            size[1]:y-size[4],
            size[2]:x-size[5],
        ]
        self.label = self.label[
            ...,
            size[0]:z-size[3],
            size[1]:y-size[4],
            size[2]:x-size[5],
        ]


    @property
    def shape(self):
        return self.image.shape



    @property
    @lru_cache
    def center(self):
        return tuple(ps // 2 for ps in self.shape)

    def normalize(self):
        def _normalize(arr):
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32)
                arr /= 255.
            elif arr.dtype == torch.uint8:
                arr = arr.type(torch.float32)
                arr /= 255.
            return arr
        self.image = _normalize(self.image)
        self.label = _normalize(self.label)

def collate_batch(batch):
   
    patch_list = []
   
    for patch in batch:
        patch_list.append(patch)
    
    return patch