import numpy as np


class Patch(object):
    def __init__(self, image: np.ndarray, label: np.ndarray,
        delayed_shrink_size: tuple = None):
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

    def apply_delayed_shrink_size(self):
        if self.delayed_shrink_size is None or not np.any(self.delayed_shrink_size):
            return
        # elif len(self.delayed_shrink_size) == 3:
        #     margin1 = tuple(s // 2 for s in self.delayed_shrink_size)
        #     margin2 = tuple(s - m1 for s, m1 in zip(self.delayed_shrink_size, margin1))
        # elif len(self.delayed_shrink_size) == 6:
        #     margin

        assert len(self.delayed_shrink_size) == 6
        self.image = self.image[
            ...,
            self.delayed_shrink_size[0]:-self.delayed_shrink_size[3],
            self.delayed_shrink_size[1]:-self.delayed_shrink_size[4],
            self.delayed_shrink_size[2]:-self.delayed_shrink_size[5],
        ]
        self.label = self.label[
            ...,
            self.delayed_shrink_size[0]:-self.delayed_shrink_size[3],
            self.delayed_shrink_size[1]:-self.delayed_shrink_size[4],
            self.delayed_shrink_size[2]:-self.delayed_shrink_size[5],
        ]