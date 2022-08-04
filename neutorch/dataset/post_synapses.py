import os
import math
from re import I
from time import time, sleep
from collections import OrderedDict
from functools import cached_property
from typing import List

import numpy as np
from scipy.stats import describe

from chunkflow.chunk import Chunk
from chunkflow.lib.bounding_boxes import Cartesian, BoundingBox
from chunkflow.lib.synapses import Synapses
from chunkflow.volume import Volume



import torch

from neutorch.dataset.ground_truth_sample import PostSynapseGroundTruth
from neutorch.dataset.transform import *
from .base import DatasetBase


def worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    
    # the dataset copy in this worker process
    dataset = worker_info.dataset
    overall_start = 0
    overall_end = dataset.sample_num

    # configure the dataset to only process the split workload
    per_worker = int(math.ceil(
        (overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


class PostSynapsesDataset(DatasetBase):
    def __init__(self, 
            syns_path_list: List[str],
            image_dirs: dict,
            patch_size: Cartesian = Cartesian(256, 256, 256), 
        ):
        """postsynapse dataset

        Args:
            syns_path_list (List[str]): the synapses file list
            image_dirs (dict): map the sample or volume name to a list of versions the same dataset.
            patch_size (Cartesian, optional): Defaults to Cartesian(256, 256, 256).

        Raises:
            ValueError: [description]
        """
        super().__init__(patch_size=patch_size)

        vols = {}
        for dataset_name, dir_list in image_dirs.items():
            vol_list = []
            for dir_path in dir_list:
                vol = Volume.from_cloudvolume_path(
                    'file://' + dir_path,
                    bounded = True,
                    fill_missing = True,
                    parallel=True,
                )
                vol_list.append(vol)
            vols[dataset_name] = vol_list

        self.samples = []

        def syns_path_to_dataset_name(syns_path: str, dataset_names: list):
            for dataset_name in dataset_names:
                if dataset_name in syns_path:
                    return dataset_name

        for syns_path in syns_path_list:
            synapses = Synapses.from_file(syns_path)
            if synapses.post is None:
                print(f'skip synapses without post: {syns_path}')
                continue
            print(f'loading {syns_path}')

            # bbox = BoundingBox.from_string(syns_path)
            bbox = synapses.pre_bounding_box
            bbox.adjust(self.patch_size_before_transform // 2)

            images = []
            dataset_name = syns_path_to_dataset_name(syns_path, image_dirs.keys())
            for vol in vols[dataset_name]:
                image = vol.cutout(bbox)
                images.append(image)
            sample = PostSynapseGroundTruth(
                synapses, images,
                patch_size=self.patch_size_before_transform
            )
            self.samples.append(sample)

        # shuffle the volume list and then split it to training and test
        # random.shuffle(volumes)
        self._compute_sample_weights()

        # the iteration range for DataLoader
        self.start = 0
        self.stop = self.sample_num

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

            print('stats of post synapse number: ',    
                    describe(synapses.post_synapse_num_list)
            )
            
            if np.all(synapses.post_synapse_num_list == 1):
                raise ValueError('it should be impossible that all the synapses only have exactly one post synapse')

            distances = synapses.distances_from_pre_to_post
            print(f'\n{fname}: distances from pre to post synapses (voxel unit): ', 
                    describe(distances)
            )

    @cached_property
    def transform(self):
        return Compose([
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
        # section_name="validation",
        section_name="training",
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