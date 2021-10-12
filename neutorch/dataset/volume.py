from typing import Union
from time import time, sleep
from multiprocessing import cpu_count

import numpy as np
import h5py

from chunkflow.chunk import Chunk
from chunkflow.lib.bounding_boxes import BoundingBoxes

import torch
from torch.utils.data import DataLoader

from neutorch.dataset.ground_truth_volume import GroundTruthVolume
from neutorch.dataset.transform import *

from cloudvolume import CloudVolume


class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
            volume: Union[str, CloudVolume],
            patch_size: Union[int, tuple]=None,
            mask: Chunk = None,
            forground_weight: int = None):
        """Neuroglancer Precomputed Volume Dataset

        Args:
            volume_path (str): cloudvolume precomputed path
            patch_size (Union[int, tuple], optional): patch size of network input. Defaults to volume block size.
            mask (Chunk, optional): forground mask. Defaults to None.
            forground_weight (int, optional): weight of bounding boxes containing forground voxels. Defaults to None.
        """
        if isinstance(volume, str):
            self.vol = CloudVolume(volume)
        elif isinstance(volume, CloudVolume):
            self.vol = volume
        else:
            raise ValueError("volume should be either an instance of CloudVolume or precomputed volume path.")

        self.voxel_size = tuple(*self.vol.resolution)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif patch_size is None:
            patch_size = tuple(*self.vol.chunk_size)
        self.patch_size = patch_size

        self.bboxes = BoundingBoxes.from_manual_setup(
            self.patch_size,
            roi_stop=self.vol.bounds.maxpt[-3:][::-1],
        )

        if mask is not None:
            # find out bboxes containing forground voxels

            if forground_weight is None:
                pass

    def __getitem__(self, idx: int):
        bbox = self.bboxes[idx]
        xyz_slices = bbox.to_slices()[::-1]
        arr = self.vol[xyz_slices]
        arr = np.asarray(arr)
        # chunk = Chunk(arr, voxel_offset=bbox.minpt, voxel_size=self.voxel_size)
        tensor = torch.Tensor(arr)
        return tensor

    def __len__(self):
        return len(self.bboxes)


if __name__ == '__main__':
    volume = ""
    dataset = Dataset(volume)

    data_loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=cpu_count() // 2,
        drop_last=True,
        pin_memory=True,
    )

    from torch.utils.tensorboard import SummaryWriter
    from neutorch.model.io import log_tensor
    writer = SummaryWriter(log_dir='/tmp/log')

    model = torch.nn.Identity()
    print('start generating random patches...')
    for image in data_loader:
        log_tensor(writer, 'train/image', image)
        sleep(1)
