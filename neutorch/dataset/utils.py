import os
import numpy as np
import h5py


def from_h5(file_name: str,
            dataset_path: str = '/main',
            global_offset: tuple = None):

    assert os.path.exists(file_name)
    assert h5py.is_hdf5(file_name)

    global_offset_path = os.path.join(os.path.dirname(file_name),
                                      'global_offset')
    with h5py.File(file_name, 'r') as f:
        arr = np.asarray(f[dataset_path])

        if global_offset is None:
            if global_offset_path in f:
                global_offset = tuple(f[global_offset_path])

    return cls(arr, global_offset=global_offset)
