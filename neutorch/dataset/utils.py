import os
import numpy as np
import h5py
import math


def from_h5(file_name: str,
            dataset_path: str = '/main',
            get_offset: tuple = False):

    assert os.path.exists(file_name)
    assert h5py.is_hdf5(file_name)

    with h5py.File(file_name, 'r') as f:
        arr = np.asarray(f[dataset_path])

        if get_offset:
            offset = f["/annotations"].attrs["offset"]
            # resolution is hard coded
            pixel_offset = (int(offset[0] / 40),
                            int(offset[1] / 4), int(offset[2] // 4))
            return arr, pixel_offset

    return arr


def split_int(i, bias='left'):
    f = i/2
    big = math.ceil(f)
    sm = math.floor(f)
    if bias == 'left':
        return (big, sm)
    elif bias == 'right':
        return (sm, big)

# pad, could do so we pad wth actual data is possible


def pad_2_divisible_by(vol, factor):
    vol_shape = vol.shape
    pad_width = []
    for i in range(len(factor)):
        left = vol_shape[i] % factor[i]
        if left > 0:
            add = factor[i] - left
            pad_width.append(split_int(add))
        else:
            pad_width.append((0, 0))
    padded = np.pad(vol, pad_width)
    vol_shape = padded.shape
    # check
    assert vol_shape[-1] % factor[-1] == 0 and vol_shape[-2] % factor[-2] == 0 and vol_shape[-3] % factor[-3] == 0, 'Image dimensions must be divisible by the patch size.'
    return padded
