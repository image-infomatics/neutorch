import random
from typing import Union
from functools import cached_property
import math

import torch
from yacs.config import CfgNode

from chunkflow.lib.cartesian_coordinate import Cartesian



def load_cfg(cfg_file: str, freeze: bool = True):
    with open(cfg_file) as file:
        cfg = CfgNode.load_cfg(file)
    if freeze:
        cfg.freeze()
    return cfg

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

def path_to_dataset_name(path: str, dataset_names: list):
    for dataset_name in dataset_names:
        if dataset_name in path:
            return dataset_name

def to_tensor(arr):
    if isinstance(arr, np.ndarray):
        arr = torch.tensor(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr


class DatasetBase(torch.utils.data.IterableDataset):
    def __init__(self,
            samples: list, 
            patch_size: Union[int, tuple, Cartesian]=(128, 128, 128),
        ):
        """
        Parameters:
            patch_size (int or tuple): the patch size we are going to provide.
        """
        super().__init__()
        self.samples = samples

        if isinstance(patch_size, int):
            patch_size = Cartesian(patch_size, patch_size, patch_size)
        else:
            patch_size = Cartesian.from_collection(patch_size)

        self.patch_size = patch_size

    @cached_property
    def sample_num(self):
        return len(self.samples)
    
    def setup_iteration_range(self):
        # the iteration range for DataLoader
        self.start = 0
        self.stop = self.sample_num

    @cached_property
    def sample_weights(self):
        # use the number of candidate patches as volume sampling weight
        sample_weights = []
        for sample in self.samples:
            sample_weights.append(sample.sampling_weight)
        return sample_weights

    @property
    def random_patch(self):
         # only sample one subject, so replacement option could be ignored
        sample_index = random.choices(
            range(0, self.sample_num),
            weights=self.sample_weights,
            k=1,
        )[0]
        sample = self.samples[sample_index]
        patch = sample.random_patch
        # patch.to_tensor()
        return patch.image, patch.label
   
    def __next__(self):
        image, label = self.random_patch
        image = to_tensor(image)
        label = to_tensor(label)
        return image, label

    def __iter__(self):
        """generate random patches from samples

        Yields:
            tuple[tensor, tensor]: image and label tensors
        """
        while True:
            yield next(self)

