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

        assert len(self.delayed_shrink_size) == 6
        _, _, z, y, x = self.image.shape
        self.image = self.image[
            ...,
            self.delayed_shrink_size[0]:z-self.delayed_shrink_size[3],
            self.delayed_shrink_size[1]:y-self.delayed_shrink_size[4],
            self.delayed_shrink_size[2]:x-self.delayed_shrink_size[5],
        ]
        self.label = self.label[
            ...,
            self.delayed_shrink_size[0]:z-self.delayed_shrink_size[3],
            self.delayed_shrink_size[1]:y-self.delayed_shrink_size[4],
            self.delayed_shrink_size[2]:x-self.delayed_shrink_size[5],
        ]