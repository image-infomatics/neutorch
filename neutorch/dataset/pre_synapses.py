import os
import random
from typing import Union, List
from time import time, sleep
from glob import glob

from tqdm import tqdm
import numpy as np
import h5py

from chunkflow.chunk import Chunk
from chunkflow.lib.bounding_boxes import Cartesian
from chunkflow.lib.synapses import Synapses
from chunkflow.volume import Volume

import torch

from .base import DatasetBase
from .ground_truth_sample import GroundTruthSampleWithPointAnnotation
from .transform import *



if __name__ == '__main__':
    dataset = Dataset(
        "~/Dropbox (Simons Foundation)/40_gt/tbar.toml",
        training_split_ratio=0.99,
    )

    from torch.utils.tensorboard import SummaryWriter
    from neutorch.model.io import log_tensor
    writer = SummaryWriter(log_dir='/tmp/log')

    import h5py
    
    model = torch.nn.Identity()
    print('start generating random patches...')
    for n in range(10000):
        ping = time()
        patch = dataset.random_training_patch
        print(f'generating a patch takes {round(time()-ping, 3)} seconds.')
        image = patch.image
        target = patch.target
        with h5py.File('/tmp/image.h5', 'w') as file:
            file['main'] = image[0,0, ...]
        with h5py.File('/tmp/target.h5', 'w') as file:
            file['main'] = target[0,0, ...]

        print('number of nonzero voxels: ', np.count_nonzero(target))
        image = torch.from_numpy(image)
        target = torch.from_numpy(target)
        log_tensor(writer, 'train/image', image, n)
        log_tensor(writer, 'train/target', target, n)

        # # print(patch)
        # logits = model(image)
        # image = image[:, :, 32, :, :]
        # tbar, _ = torch.max(tbar, dim=2, keepdim=False)
        # slices = torch.cat((image, tbar))
        # image_path = os.path.expanduser('~/Downloads/patches.png')
        # print('save a batch of patches to ', image_path)
        # torchvision.utils.save_image(
        #     slices,
        #     image_path,
        #     nrow=8,
        #     normalize=True,
        #     scale_each=True,
        # )
        sleep(1)

