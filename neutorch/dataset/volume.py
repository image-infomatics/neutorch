import os
from typing import Union
from time import sleep
from copy import deepcopy
from multiprocessing import cpu_count

import numpy as np

from chunkflow.chunk import Chunk
from chunkflow.lib.bounding_boxes import BoundingBoxes

import torch
from torch.utils.data import DataLoader

from neutorch.dataset.transform import *
from neutorch.dataset.patch import Patch, collate_batch

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
            self.vol = CloudVolume(
                volume, 
                fill_missing=False, 
                parallel=False, 
                progress=False,
                green_threads = False,
            )
        elif isinstance(volume, CloudVolume):
            self.vol = volume
        else:
            raise ValueError("volume should be either an instance of CloudVolume or precomputed volume path.")

        # self.voxel_size = tuple(self.vol.resolution)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif patch_size is None:
            patch_size = tuple(self.vol.chunk_size)
        self.patch_size = patch_size

        self.bboxes = BoundingBoxes.from_manual_setup(
            self.patch_size,
            roi_start=(0, 0, 0),
            roi_stop=self.vol.bounds.maxpt[-3:][::-1],
            bounded=True,
        )
        print(f'found {len(self.bboxes)} bounding boxes in volume: {volume}')

        if mask is not None:
            # find out bboxes containing forground voxels

            if forground_weight is None:
                pass
        
        # prepare transform
        self.transform = Compose([
            NormalizeTo01(),
            AdjustBrightness(),
            AdjustContrast(),
            Gamma(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            BlackBox(),
        ])
        

    def __getitem__(self, idx: int):
        bbox = self.bboxes[idx]
        xyz_slices = bbox.to_slices()[::-1]
        print('xyz slices: ', xyz_slices)
        image = self.vol[xyz_slices]
        image = np.asarray(image)
        image = np.transpose(image)
        # image = image.astype(np.float32)
        # image /= 255.
        # chunk = Chunk(arr, voxel_offset=bbox.minpt, voxel_size=self.voxel_size)
        # tensor = torch.Tensor(arr)
        label = deepcopy(image)
        patch = Patch(image, label)
        self.transform(patch)
        patch.to_tensor()
        patch.normalize()
        return patch

    @property
    def random_patch(self):
        idx = random.randrange(0, len(self.bboxes))
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.bboxes)


if __name__ == '__main__':
    volume = "file:///mnt/ceph/users/neuro/wasp_em/ykreinin/sample_2.1/3.contrast"
    dataset = Dataset(volume)

    data_loader = DataLoader(
        dataset,
        shuffle=True,
        # num_workers=2,
        drop_last=True,
        # pin_memory=True,
        collate_fn=collate_batch,
    )

    from torch.utils.tensorboard import SummaryWriter
    from neutorch.model.io import log_tensor
    writer = SummaryWriter(log_dir=os.path.expanduser('./log'))

    model = torch.nn.Identity()
    print('start generating random patches...')
    idx = 0
    for patch in data_loader:
        idx += 1
        print('iteration index: ', idx)
        log_tensor(writer, 'train/image', patch.image, iter_idx = idx, nrow=1, zstride=64)
        log_tensor(writer, 'train/label', patch.label, iter_idx = idx, nrow=1, zstride=64)
        sleep(1)
