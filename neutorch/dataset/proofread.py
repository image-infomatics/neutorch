import math

import numpy as np

import torch
from .tio_transforms import *
from .utils import from_h5
from skimage.segmentation import expand_labels
from einops import rearrange


class ProofreadDataset(torch.utils.data.Dataset):
    def __init__(self, image: np.ndarray,
                 pred_label: np.ndarray,
                 true_label: np.ndarray,
                 min_volume: int = 300,
                 patch_size: Tuple = (1, 16, 16),
                 expd_amt: Tuple = (4, 40, 40),
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
            expd_amt (Tuple): how much to expand the label for context is x,y
        """

        self.name = name
        assert image.ndim == 3
        assert image.shape == true_label.shape[-3:]
        assert image.shape == pred_label.shape[-3:]

        self.shape = image.shape
        self.image = image
        self.true_label = true_label
        self.pred_label = pred_label
        self.patch_size = patch_size
        self.expd_amt = expd_amt

        self.classes = []
        classes, counts = np.unique(pred_label, return_counts=True)
        self.min_volume = min_volume
        for i, c in enumerate(counts):
            if c > self.min_volume:
                self.classes.append(classes[i])

    # given some label, c, within the entire volume
    # then splites the neurite defined by this label into patches
    # then returns two arrays (images, labels) where each array contains the patches
    # the image_patches contrains 4 channels, c0 = image, c1,c2,c3, contrain orignal z,y,x coords
    def get_patch_array(self, c):

        indices = np.argwhere(self.pred_label == c)  # get location of labels

        # build ranges to crop neurite
        mins = np.amin(indices, axis=0)
        maxs = np.amax(indices, axis=0)

        for i in range(3):
            mins[i] = max(mins[i] - self.expd_amt[i], 0)
            maxs[i] = min(maxs[i] + self.expd_amt[i], self.shape[i]-1)

        # crop
        crop_sl = np.s_[mins[0]:maxs[0]+1,
                        mins[1]:maxs[1]+1, mins[2]:maxs[2]+1]

        true_label_sec = self.true_label[crop_sl]
        label_sec = self.pred_label[crop_sl]
        image_sec = self.image[crop_sl]
        coord_sec = np.mgrid[crop_sl]

        # add channel for coordinates
        image_sec = np.expand_dims(image_sec, 0)
        image_sec = np.concatenate((image_sec, coord_sec), axis=0)

        # zero other classes
        label_sec_b = np.zeros_like(label_sec)
        label_sec_b[label_sec == c] = 1

        c = 1

        expanded = np.zeros_like(label_sec)
        ez, ey, ex = self.expd_amt
        # we are assuming ey == ez, could be better way to expand labels here
        # expand label in z,y
        for i in range(label_sec_b.shape[2]):
            expanded[:, :, i] = expand_labels(
                label_sec_b[:, :, i], ez//2)

        # expand label in z,x
        for i in range(expanded.shape[1]):
            expanded[:, i, :] = expand_labels(expanded[:, i, :], ez//2)

        # expand label in x,y
        for i in range(expanded.shape[0]):
            expanded[i, :, :] = expand_labels(
                expanded[i, :, :], ex-(ez//2))

        expanded = np.expand_dims(expanded, 0)
        true = np.expand_dims(true_label_sec, 0)
        combined = np.concatenate((image_sec, true, expanded), 0)
        combined_arr = self._patchify(combined,  (1, *self.patch_size))
        image_arr = combined_arr[:, :-2, ...]
        label_arr = combined_arr[:, -2:-1, ...]
        exp_label_arr = combined_arr[:, -1:, ...]
        image_arr, label_arr = self._drop_patches_without_expanded_label(
            image_arr, label_arr, exp_label_arr, c)

        return (image_arr, label_arr)

    def _patchify(self, vol, patch_size):

        # pad, could do so we pad wth actual data is possible
        vol_shape = vol.shape
        pad_width = []
        for i in range(len(patch_size)):
            left = vol_shape[i] % patch_size[i]
            if left > 0:
                add = patch_size[i] - left
                pad_width.append((math.floor(add/2), math.ceil(add/2)))
            else:
                pad_width.append((0, 0))
        padded = np.pad(vol, pad_width)
        vol_shape = padded.shape

        # check
        assert vol_shape[-1] % patch_size[-1] == 0 and vol_shape[-2] % patch_size[-2] == 0 and vol_shape[-3] % patch_size[-3] == 0, 'Image dimensions must be divisible by the patch size.'

        # patch
        pz, py, px = patch_size[-3:]
        arr = rearrange(
            padded, 'c (z pz) (y py) (x px) -> (z y x) c pz py px', pz=pz, py=py, px=px)

        return arr

    # drop zero patches
    # assumes image_arr and label_arr are in corresponding order
    def _drop_patches_without_expanded_label(self, image_arr, label_arr, exp_label_arr, c):
        i_drop_arr = []
        l_drop_arr = []
        for i in range(label_arr.shape[0]):
            if c in exp_label_arr[i]:
                i_drop_arr.append(image_arr[i])
                l_drop_arr.append(label_arr[i])
        return np.array(i_drop_arr), np.array(l_drop_arr)

    def __getitem__(self, idx):
        c = self.classes[idx]
        (X, y) = self.get_patch_array(c)

        return X, y

    def __len__(self):
        return len(self.classes)
