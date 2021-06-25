import os
import numpy as np
import h5py


def from_h5(file_name: str,
            dataset_path: str = '/main',
            get_offset: tuple = False):

    assert os.path.exists(file_name)
    assert h5py.is_hdf5(file_name)

    with h5py.File(file_name, 'r') as f:
        arr = np.asarray(f[dataset_path])
        if get_offset:
            offset = f["/annotations"].attrs["offset"]

    return arr, offset
