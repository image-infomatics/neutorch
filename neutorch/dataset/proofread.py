import math
import numpy as np
import torch
from skimage.segmentation import expand_labels
from einops import rearrange
from time import time

from neutorch.dataset.utils import pad_2_divisible_by, compute_affinty_from_offset
from positional_encodings import PositionalEncoding3D


class ProofreadDataset(torch.utils.data.Dataset):
    def __init__(self, image: np.ndarray,
                 pred: np.ndarray,
                 true: np.ndarray,
                 aff: np.ndarray,
                 min_volume: int = 300,
                 max_volume: int = 26*256*256*2,
                 patch_size=(1, 16, 16),
                 expd_amt: int = 4,
                 shuffle: bool = True,
                 sort: bool = False,
                 positional_encoding: str = 'euclid',
                 name: str = '',
                 border_width: int = 1) -> None:
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
            expd_amt (int): how much to expand the label for context, each unit expands by one patch_size
        """

        self.name = name
        assert image.ndim == 3
        assert image.shape == true.shape[-3:]
        assert image.shape == pred.shape[-3:]

        # add channel dim
        self.patch_size = (1, *patch_size)

        self.og_shape = image.shape
        (sz, sy, sx) = self.og_shape

        # volumes
        image = image.astype(np.float32) / 255.
        self.image, self.padded_shape = self._patchify(
            np.expand_dims(image, 0), return_padded_shape=True)
        self.pred = self._patchify(np.expand_dims(pred, 0))

        # optional in test case
        self.true = None
        if true is not None:
            affinities = compute_affinty_from_offset(
                true, (1, 1, 1), border_width)
            self.true = self._patchify(affinities)
        # optional, may not want pred affs
        self.aff = None
        if aff is not None:
            self.aff = self._patchify(aff)

        # do positional encodings
        if positional_encoding == 'euclid':
            self.cords = self._patchify(np.mgrid[0:sz, 0:sy, 0:sx])
        elif positional_encoding == 'sine':
            # add channel and batch dims
            dummy = np.zeros((1, 1, *self.og_shape))
            p_enc_3d = PositionalEncoding3D(11)
            z = torch.zeros((1, 5, 6, 4, 11))
            # takes in form (batchsize, x, y, z, ch) so must permute
            print(p_enc_3d(z).shape)  # (1, 5, 6, 4, 11)

        self.expd_amt = expd_amt
        self.shuffle = shuffle

        self.classes = []
        classes, counts = np.unique(pred, return_counts=True)

        if shuffle:
            shuffler = np.random.permutation(len(classes))
            classes = classes[shuffler]
            counts = counts[shuffler]

        if sort:
            sort_indices = np.argsort(counts)
            sort_indices = np.flip(sort_indices)
            classes = classes[sort_indices]
            counts = counts[sort_indices]

        new_class = max(classes)
        self.min_volume = min_volume
        self.max_volume = max_volume
        for i, count in enumerate(counts):
            c = classes[i]
            if count > self.min_volume and count <= self.max_volume:
                self.classes.append(c)

            # here if the volume of a segmentation is too large
            # we split it into parts, by spliting the instances of the classes
            # into parts in preds
            if count > self.max_volume:
                split_factor = math.ceil(count/self.max_volume)
                original_shape = self.pred.shape
                flat = self.pred.flatten()
                new_flat = flat.copy()
                step = math.ceil(count / split_factor)

                for spl in range(split_factor):
                    # pick a new_class and set label in section of patches to new_class
                    new_class = new_class + 1
                    self.classes.append(new_class)
                    splt = np.s_[spl*step:(spl+1)*step]
                    new_flat[np.where(flat == c)[0][splt]] = new_class

                self.pred = np.reshape(new_flat, original_shape)

    # given some label, c, within the entire volume
    # returns indices of the patch array where c exists after the label is expanded

    def get_patch_array_indices(self, c):
        # get array indicies where there is label
        arr_indc = np.any(self.pred == c, axis=(1, 2, 3, 4))

        # reshape back into shape of padded volume
        _, dz, dy, dx = [pad//p for p,
                         pad in zip(self.patch_size, self.padded_shape)]
        vol_shaped_indices = rearrange(
            arr_indc, '(dz dy dx) -> dz dy dx', dz=dz, dy=dy, dx=dx)

        # convert to numbers to perform expand_labels
        visible_indices = np.zeros_like(vol_shaped_indices)
        visible_indices[vol_shaped_indices] = 1

        # expand labels
        visible_indices_exp = expand_labels(
            visible_indices, distance=self.expd_amt)

        # convert back to bool indexing
        vol_shaped_indices_exp = visible_indices_exp == 1

        # rearrange back to array indices
        arr_indc_exp = rearrange(
            vol_shaped_indices_exp, 'dz dy dx -> (dz dy dx)', dz=dz, dy=dy, dx=dx)

        # return indices of the array where label is or expanded into
        return arr_indc_exp

    # convert a volume into an array of patches of size patch size
    # can return shape of padded volume needed
    def _patchify(self, vol, return_padded_shape=False):

        # patch
        _, pz, py, px = self.patch_size

        vol = pad_2_divisible_by(vol, self.patch_size)
        vol_arr = rearrange(
            vol, 'c (z pz) (y py) (x px) -> (z y x) c pz py px', pz=pz, py=py, px=px)

        if return_padded_shape:
            return vol_arr, vol.shape
        return vol_arr

    def __getitem__(self, idx):
        c = self.classes[idx]
        indices = self.get_patch_array_indices(c)

        image_select = self.image[indices]
        cords_select = self.cords[indices]

        ret = [image_select, cords_select]
        if self.aff is not None:
            aff_select = self.aff[indices]
            ret.append(aff_select)
        if self.true is not None:
            true_select = self.true[indices]
            ret.append(true_select)

        return tuple(ret)

    def __len__(self):
        return len(self.classes)
