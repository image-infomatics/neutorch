import numpy as np
import torchio as tio
from typing import Optional

from .border_mask import create_border_mask


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
                 affinity_offsets,
                 lsd_label: Optional[np.ndarray] = None,
                 border_width: int = 1):
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

        self.is_lsd = False
        if lsd_label is not None:
            self.is_lsd = True
            # add background mask
            mask = np.zeros(label.shape)
            create_border_mask(label, mask, border_width, 0)
            lsd_label[:, mask == 0] = 0

            tio_lsd = tio.LabelMap(tensor=lsd_label)
            subject.add_image(tio_lsd, 'lsd')

        self.subject = subject
        self.border_width = border_width
        self.affinity_offsets = affinity_offsets

    def _compute_affinty_from_offset(self, label, affinity_offset):

        z0, y0, x0 = label.shape
        (zo, yo, xo) = affinity_offset

        # add background mask
        masked_label = np.zeros(label.shape, dtype=np.uint64)
        create_border_mask(label, masked_label, self.border_width, 0)

        # along some axis X, affinity is 1 or 0 based on if voxel x === x-1
        affinity = np.zeros((3, z0, y0, x0))
        affinity[2, zo:, :, :] = masked_label[..., zo:, :,
                                              :] == masked_label[..., 0:-zo, :, :]  # z channel
        affinity[1, :, yo:, :] = masked_label[..., :, yo:,
                                              :] == masked_label[..., :, 0:-yo, :]  # y channel
        affinity[0, :, :, xo:] = masked_label[..., :, :,
                                              xo:] == masked_label[..., :, :, 0:-xo]  # x channel

        # but back in background labels
        affinity[:, masked_label == 0] = 0

        return affinity

    # segmentation label into affinty map
    def compute_affinity(self):

        label = np.squeeze(self.subject.label.tensor.numpy())
        z0, y0, x0 = label.shape
        aff_channels = len(self.affinity_offsets) * 3
        full_affinity = np.zeros((aff_channels, z0, y0, x0))
        for i, off in enumerate(self.affinity_offsets):
            aff = self._compute_affinty_from_offset(label, off)
            full_affinity[i*3:(i+1)*3, :, :, :] = aff

        tio_affinity = tio.LabelMap(tensor=full_affinity)
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
        assert self.is_lsd
        return self.subject.lsd.tensor.numpy()

    @property
    def target(self):
        if self.is_lsd:
            return np.append(self.affinity, self.lsd, axis=0)
        else:
            return self.affinity

    def get_lsd_channel(self, channel):
        assert self.is_lsd
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
