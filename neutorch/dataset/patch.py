import numpy as np
import torchio as tio
from typing import Optional

from .border_mask import create_border_mask

# (int): the amount of border added to the affinity maps (for thicker lines)
AFF_BORDER_WIDTH = 1


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
    def center(self):
        return tuple(ps // 2 for ps in self.shape)


class AffinityPatch(object):
    def __init__(self, image: np.ndarray, label: np.ndarray,
                 lsd_label: Optional[np.ndarray] = None):
        """A patch of volume containing both image and label

        Args:
            image (np.ndarray): image
            label (np.ndarray): label
            lsd_label (np.ndarray): auxiliary label such as LSD
        """
        assert image.shape == label.shape

        image_tensor = np.expand_dims(image, axis=0)
        label_tensor = np.expand_dims(label, axis=0)

        tio_image = tio.ScalarImage(tensor=image_tensor)
        tio_label = tio.LabelMap(tensor=label_tensor)
        subject = tio.Subject(
            image=tio_image, label=tio_label)

        if lsd_label is not None:
            tio_lsd = tio.LabelMap(tensor=lsd_label)
            subject.add_image(tio_lsd, 'lsd')

        self.subject = subject

    # segmentation label into affinty map
    def compute_affinity(self):

        label = np.squeeze(self.subject.label.tensor.numpy())
        z0, y0, x0 = label.shape

        # add background mask
        masked_label = np.zeros(label.shape, dtype=np.uint64)
        create_border_mask(label, masked_label, AFF_BORDER_WIDTH, 0)

        # along some axis X, affinity is 1 or 0 based on if voxel x === x-1
        affinity = np.zeros((3, z0, y0, x0))
        affinity[2, 1:, :, :] = masked_label[..., 1:, :,
                                             :] == masked_label[..., 0:-1, :, :]  # z channel
        affinity[1, :, 1:, :] = masked_label[..., :, 1:,
                                             :] == masked_label[..., :, 0:-1, :]  # y channel
        affinity[0, :, :, 1:] = masked_label[..., :, :,
                                             1:] == masked_label[..., :, :, 0:-1]  # x channel

        # but back in background labels
        affinity[:, masked_label == 0] = 0

        tio_affinity = tio.LabelMap(tensor=affinity)
        self.subject.add_image(tio_affinity, 'affinity')

    @property
    def shape(self):
        return self.subject.image.tensor.shape

    @property
    def image(self):
        return self.subject.image.tensor.numpy()

    @property
    def label(self):
        return self.subject.label.tensor.numpy()

    @property
    def affinity(self):
        return self.subject.affinity.tensor.numpy()

    @property
    def lsd(self):
        return self.subject.lsd.tensor.numpy()

    @property
    def target(self):
        return np.append(self.affinity, self.lsd, axis=0)

    def get_lsd_channel(self, channel):
        lsd = self.subject.lsd.tensor.numpy()
        if channel == 0:
            return np.moveaxis(lsd[0:3, :, :, :], 0, 3)
        if channel == 1:
            return np.moveaxis(lsd[3:6, :, :, :], 0, 3)
        if channel == 2:
            return np.moveaxis(lsd[6:9, :, :, :], 0, 3)
        if channel == 3:
            return lsd[9, :, :, :]

    @property
    def center(self):
        return tuple(ps // 2 for ps in self.shape)


class AffinityBatch(object):
    def __init__(self, patches):
        """An array of patches used for batching

        Args:
            patches (Array): patches
        """

        self.length = len(patches)
        self.patches = patches
        images = []
        targets = []
        labels = []

        for p in patches:
            images.append(p.image)
            targets.append(p.target)
            labels.append(p.label)

        self.images = np.array(images)
        self.targets = np.array(targets)
        self.labels = np.array(labels)
