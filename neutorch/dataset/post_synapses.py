import os
import math
import random
from time import time, sleep
from collections import OrderedDict

import numpy as np
from scipy.stats import describe
from tqdm import tqdm

from chunkflow.chunk import Chunk
from chunkflow.lib.bounding_boxes import Cartesian
from chunkflow.lib.synapses import Synapses

from cloudvolume import CloudVolume

import torch
import toml

from neutorch.dataset.ground_truth_sample import PostSynapseGroundTruth
from neutorch.dataset.transform import *


def worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    # the dataset copy in this worker process
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end

    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, config_file: str,
            section_name: str, 
            patch_size: Cartesian = Cartesian(256, 256, 256), 
            ):
        """postsynapse dataset

        Args:
            config_file (str): the toml file path
            section_name (str): the section name to be used in the toml file. [training, validation, testing]
            patch_size (Cartesian, optional): [description]. Defaults to Cartesian(256, 256, 256).

        Raises:
            ValueError: [description]
        """
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = Cartesian(*((patch_size,) * 3))
        else:
            patch_size = Cartesian(*patch_size)
        
        config_file = os.path.expanduser(config_file)
        assert config_file.endswith('.toml'), "we use toml file as configuration format."
        with open(config_file, 'r') as file:
            meta = toml.load(file)

        self._load_synapses(meta[section_name]['path'])

        vol1 = CloudVolume("file://" + meta['sample1']["image"])
        vol2 = CloudVolume("file://" + meta['sample2']["image"])

        self._prepare_transform()
        assert isinstance(self.transform, Compose)

        self.patch_size = patch_size
        patch_size_before_transform = self.patch_size + self.transform.shrink_size[:3]
        patch_size_before_transform += self.transform.shrink_size[-3:]

        self.sample_list = []
        for key, synapses in tqdm(self.synapses.items(), desc="reading image chunks..."):
            sample, _ = key.split(",")
            # bbox = BoundingBox.from_filename(bbox_filename)
            bbox = synapses.pre_bounding_box
            bbox.adjust(patch_size_before_transform // 2)
            if "1" in sample:
                vol = vol1
            elif "2" in sample:
                vol = vol2
            else:
                raise ValueError("we can only read data from sample 1 or 2")

            chunk = vol[bbox.to_slices()[::-1]]
            # CloudVolume uses xyz order and we use zyx order
            chunk = np.transpose(chunk)
            chunk = Chunk(np.asarray(chunk), voxel_offset=bbox.minpt)
            sample = PostSynapseGroundTruth(
                chunk, synapses, 
                patch_size=patch_size_before_transform
            )
            self.sample_list.append(sample)

                
        # shuffle the volume list and then split it to training and test
        # random.shuffle(volumes)

        # use the number of candidate patches as volume sampling weight
        sample_weights = []
        for sample in self.sample_list:
            sample_weights.append(sample.sampling_weight) 
        self.sample_weights = sample_weights

        self.sample_num = len(self.sample_list)
        self.start = 0
        self.end = self.sample_num

    def _load_synapses(self, path: str):
        self.synapses = OrderedDict()
        path = os.path.expanduser(path)
        print(f'load synapses from: {path}')
        assert os.path.isdir(path)
        for fname in os.listdir(path):
            if not fname.endswith(".h5"):
                # ignore all other files or directories 
                continue
            full_fname = os.path.join(path, fname)
            synapses = Synapses.from_h5(full_fname, c_order=False)
            # some of the synapses only have presynapse without any postsynapses
            # this could be an error of ground truth!
            synapses.remove_synapses_without_post()

            key = fname[:-3]
            self.synapses[key] = synapses

            distances = synapses.distances_from_pre_to_post
            print(f'\n{fname}: distances from pre to post synapses (voxel unit): ', 
                    describe(distances))

    def __next__(self):
        # only sample one subject, so replacement option could be ignored
        sample_index = random.choices(
            range(self.start, self.end),
            weights=self.sample_weights[self.start : self.end],
            k=1,
        )[0]
        sample = self.sample_list[sample_index]
        patch = sample.random_patch
        self.transform(patch)
        patch.apply_delayed_shrink_size()
        print('patch shape: ', patch.shape)
        assert patch.shape[-3:] == self.patch_size, f'get patch shape: {patch.shape}, expected patch size {self.patch_size}'
        
        patch.to_tensor()
        return patch.image, patch.target

    def __iter__(self):
        """generate random patches from samples

        Yields:
            tuple[tensor, tensor]: image and target tensors
        """
        while True:
            yield next(self)

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
    
    from torch.utils.data import DataLoader
    from neutorch.dataset.patch import collate_batch
    dataset = Dataset(
        "/mnt/ceph/users/neuro/wasp_em/jwu/14_post_synapse_net/post.toml",
        section_name="validation",
    )
    data_loader = DataLoader(
        dataset,
        num_workers=4,
        prefetch_factor=2,
        drop_last=False,
        multiprocessing_context='spawn',
        collate_fn=collate_batch,
    )
    data_iter = iter(data_loader)

    from torch.utils.tensorboard import SummaryWriter
    from neutorch.model.io import log_tensor
    writer = SummaryWriter(log_dir='/tmp/log')

    model = torch.nn.Identity()
    print('start generating random patches...')
    for n in range(10000):
        ping = time()
        image, target = next(data_iter)
        image = image.cpu()
        target = target.cpu()
        print(f'generating a patch takes {round(time()-ping, 3)} seconds.')
        print('number of nonzero voxels: ', np.count_nonzero(target>0.5))
        # assert np.any(target > 0.5)
        assert torch.any(target > 0.5)

        log_tensor(writer, 'train/image', image, n)
        log_tensor(writer, 'train/target', target, n)

        sleep(1)