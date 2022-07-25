import json
import os
import math
import random
from typing import Union
from time import time, sleep
from glob import glob

from tqdm import tqdm
import numpy as np
import h5py

from chunkflow.chunk import Chunk
from chunkflow.lib.bounding_boxes import Cartesian
from chunkflow.lib.synapses import Synapses

import torch

from neutorch.dataset.ground_truth_sample import GroundTruthSampleWithPointAnnotation
from neutorch.dataset.transform import *



class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
            glob_path: str, 
            validation_names: list,
            test_names: list,
            patch_size: Union[int, tuple, Cartesian]=(64, 64, 64),
        ):
        """
        Parameters:
            glob_path (str): the glob path with regular expression.
            validation_names (list[str]): the substring to select sample names as validation set.
            test_names (list[str]): the substring to select sample names as test set.
            patch_size (int or tuple): the patch size we are going to provide.
        """
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = Cartesian(patch_size, patch_size, patch_size)
        else:
            patch_size = Cartesian.from_collection(patch_size)

        glob_path = os.path.expanduser(glob_path)
        path_list = glob(glob_path, recursive=True)
        path_list = sorted(path_list)
        assert len(path_list) > 1
        assert len(path_list) % 2 == 0, \
            "the image and synapses should be paired."

        self._prepare_transform()
        assert isinstance(self.transform, Compose)

        self.patch_size = patch_size
        patch_size_before_transform = tuple(
            p + s0 + s1 for p, s0, s1 in zip(
                patch_size, 
                self.transform.shrink_size[:3], 
                self.transform.shrink_size[-3:]
            )
        )
        patch_size_before_transform = Cartesian.from_collection(patch_size_before_transform)
        
        # load all the datasets
        training_samples = []
        validation_samples = []
        for idx in tqdm(range(0, len(path_list), 2)):
            image_path = path_list[idx]
            synapse_path = path_list[idx+1]
                
            image_path = os.path.expanduser(image_path)
            synapse_path = os.path.expanduser(synapse_path)
            print(f'image path: {image_path}')
            print(f'synapses path: {synapse_path}')

            if 'img' not in image_path:
                breakpoint()
            assert 'img_' in image_path
            assert 'syns_' in synapse_path

            image = Chunk.from_h5(image_path)
            voxel_offset = image.voxel_offset
            image = image.astype(np.float32) / 255.
            # use the voxel number as the sampling weights
            # subject_weights.append(len(img))
            synapses = Synapses.from_h5(synapse_path)
            synapses.remove_synapses_outside_bounding_box(image.bbox)

            pre = synapses.pre 
            # print(f'min offset: {np.min(pre, axis=0)}')
            # print(f'max offset: {np.max(pre, axis=0)}')
            pre -= np.asarray(voxel_offset, dtype=pre.dtype)
            # breakpoint()

            ground_truth_sample = GroundTruthSampleWithPointAnnotation(
                image.array,
                annotation_points=pre,
                patch_size=patch_size_before_transform,
            )


            if any(s in synapse_path for s in validation_names):
                validation_samples.append(ground_truth_sample)
            elif any(s in synapse_path for s in test_names):
                print(f'skipping the test sample: {synapse_path}')
                pass
            else:
                training_samples.append(ground_truth_sample)
        

        # use the number of candidate patches as volume sampling weight
        validation_sample_weights = []
        for volume in validation_samples:
            validation_sample_weights.append(volume.sampling_weight)
        
        training_sample_weights = []
        for volume in training_samples:
            training_sample_weights.append(volume.sampling_weight)

        self.training_sample_num = len(training_samples)
        self.validation_sample_num = len(validation_samples)
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.training_sample_weights = training_sample_weights
        self.validation_sample_weights = validation_sample_weights
        
    @property
    def random_training_patch(self):
        # only sample one subject, so replacement option could be ignored
        if self.training_sample_num == 1:
            sample_index = 0
        else:
            sample_index = random.choices(
                range(self.training_sample_num),
                weights=self.training_sample_weights,
                k=1,
            )[0]
        volume = self.training_samples[sample_index]
        patch = volume.random_patch
        self.transform(patch)
        patch.apply_delayed_shrink_size()
        # print('patch shape: ', patch.shape)
        assert patch.shape[-3:] == self.patch_size, f'patch shape: {patch.shape}'
        return patch

    @property
    def random_validation_patch(self):
        if self.validation_sample_num == 1:
            sample_index = 0
        else:
            sample_index = random.choices(
                range(self.validation_sample_num),
                weights=self.validation_sample_weights,
                k=1,
            )[0]
        volume = self.validation_samples[sample_index]
        patch = volume.random_patch
        self.transform(patch)
        patch.apply_delayed_shrink_size()
        return patch
           
    def _prepare_transform(self):
        self.transform = Compose([
            NormalizeTo01(probability=1.),
            AdjustBrightness(),
            AdjustContrast(),
            Gamma(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            BlackBox(),
            Perspective2D(),
            # RotateScale(probability=1.),
            #DropSection(),
            Flip(),
            Transpose(),
            MissAlignment(),
        ])


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

