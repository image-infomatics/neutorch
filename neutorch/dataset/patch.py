from functools import lru_cache
import numpy as np

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
        self.image = image
        self.label = label
        self.delayed_shrink_size = delayed_shrink_size

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


class AffinityPatch(object):
    def __init__(self, image: np.ndarray, label: np.ndarray,):
        """A patch of volume containing both image and label

        Args:
            image (np.ndarray): image
            label (np.ndarray): label
        """
    

        assert image.shape == label.shape
        self.image = image # EM volume
        self.label = label # segmentation label
        self.target = label # init target to current label


    # def shrink(self, size: tuple):
    #     assert len(size) == 6
    #     _, _, z, y, x = self.shape
    #     self.image = self.image[
    #         ...,
    #         size[0]:z-size[3],
    #         size[1]:y-size[4],
    #         size[2]:x-size[5],
    #     ]
    #     self.label = self.label[
    #         ...,
    #         size[0]:z-size[3],
    #         size[1]:y-size[4],
    #         size[2]:x-size[5],
    #     ]
    #     self.target = self.target[
    #         ...,
    #         size[0]:z-size[3],
    #         size[1]:y-size[4],
    #         size[2]:x-size[5],
    #     ]

    def compute_target(self):
        print("cur label shape", self.label.shape)
        affinity = self.compute_affinity(self.label) # affinity map
        # self.LSD # local shape descriptor

        self.target = affinity # learning target

    # segmentation label into affinty map
    def compute_affinity(self, label):
        b0, c0, z0, y0, x0 = self.shape

        # along some axis X, affinity is 1 or 0 based on if voxel x === x-1
        cur_label = label[..., 1:, 1:, 1:]
        offset_label = label[..., 0:-1, 0:-1, 0:-1]

        newlabel = np.zeros((b0, c0, z0-1, y0-1, x0-1), dtype=label.dtype)
        newlabel[..., :, :, :] = cur_label == offset_label

        return newlabel

    @property
    def shape(self):
        return self.image.shape

    @property
    @lru_cache
    def center(self):
        return tuple(ps // 2 for ps in self.shape)