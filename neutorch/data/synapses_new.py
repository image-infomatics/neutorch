import os
from functools import cached_property
from time import sleep, time
from typing import List, Union

import numpy as np

from chunkflow.lib.cartesian_coordinate import Cartesian, BoundingBox
from chunkflow.lib.synapses import Synapses
from chunkflow.volume import Volume

import torch

from .transform import *
from .dataset import DatasetBase, SemanticDataset, path_to_dataset_name
from .sample import SampleWithPointAnnotation, PostSynapseReference

DEFAULT_PATCH_SIZE = Cartesian(128, 128, 128)

def vols_dir(sample_name_to_image_versions):
    vols = {}
    for dataset_name, dir_list in sample_name_to_image_versions.items():
        vol_list = []
        for dir_path in dir_list:
            vol = Volume.from_cloudvolume_path(
                'file://' + dir_path,
                bounded = True,
                fill_missing = False,
                parallel=True,
            )
            vol_list.append(vol)
        vols[dataset_name] = vol_list
    return vols 

class SynapsesDatasetBase(SemanticDataset):
    def __init__(self, 
            sample_name_to_image_versions: dict,
            patch_size: Union[int, tuple, Cartesian] = DEFAULT_PATCH_SIZE):
        super().__init__(patch_size)

        self.sample_name_to_image_versions = sample_name_to_image_versions
        self.vols = vols_dir(sample_name_to_image_versions)
        self.samples = []

    def syns_path_to_images(self, syns_path: str, bbox: BoundingBox):
        images = []
        dataset_name = path_to_dataset_name(
            syns_path, 
            self.sample_name_to_image_versions.keys()
        )
        for vol in self.vols[dataset_name]:
            image = vol.cutout(bbox)
            images.append(image)
        return images

class PreSynapsesDataset(SynapsesDatasetBase):
    def __init__(self, 
            syns_path_list: List[str],
            sample_name_to_image_versions: dict,
            patch_size: Union[int, tuple, Cartesian] = DEFAULT_PATCH_SIZE,
        ):
        """
        Parameters:
            syns_path_list (List[str]): the synapses file list
            sample_name_to_image_versions (dict): map the sample or volume name to a list of versions the same dataset.
            patch_size (int or tuple): the patch size we are going to provide.
        """

        if isinstance(patch_size, int):
            patch_size = Cartesian(patch_size, patch_size, patch_size)
        else:
            patch_size = Cartesian.from_collection(patch_size)
        super().__init__(sample_name_to_image_versions, patch_size=patch_size)
        
        self.transform = transform() #why can't this be called from semantic?
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

        for syns_path in syns_path_list:
            bbox = BoundingBox.from_string(syns_path)
            images = self.syns_path_to_images(syns_path, bbox)
            
            synapses = Synapses.from_h5(syns_path)
            synapses.remove_synapses_outside_bounding_box(bbox)
            
            pre = synapses.pre 
            # print(f'min offset: {np.min(pre, axis=0)}')
            # print(f'max offset: {np.max(pre, axis=0)}')
            pre -= np.asarray(bbox.start, dtype=pre.dtype)

            sample = SampleWithPointAnnotation(
                images,
                annotation_points=pre,
                patch_size=patch_size_before_transform,
            )
            self.samples.append(sample)

        self.compute_sample_weights()
        self.setup_iteration_range()

class PostSynapsesDataset(SynapsesDatasetBase):
    def __init__(self, 
            syns_path_list: List[str],
            sample_name_to_image_versions: dict,
            patch_size: Union[int, tuple, Cartesian] = Cartesian(256, 256, 256), #could we try default?
        ):
        """postsynapse dataset

        Args:
            syns_path_list (List[str]): the synapses file list
            sample_name_to_image_versions (dict): map the sample or volume name to a list of versions the same dataset.
            patch_size (Cartesian, optional): Defaults to Cartesian(256, 256, 256).

        Raises:
            ValueError: [description]
        """
        super().__init__(sample_name_to_image_versions, patch_size=patch_size)

        for syns_path in syns_path_list:
            synapses = Synapses.from_file(syns_path)
            if synapses.post is None:
                print(f'skip synapses without post: {syns_path}')
                continue
            print(f'loaded {syns_path}')
            synapses.remove_synapses_without_post()

            # bbox = BoundingBox.from_string(syns_path)
            bbox = synapses.pre_bounding_box
            bbox.adjust(self.patch_size_before_transform // 2)

            images = self.syns_path_to_images(syns_path, bbox)
            sample = PostSynapseReference(
                synapses, images,
                patch_size=self.patch_size_before_transform
            )
            self.samples.append(sample)

        self.compute_sample_weights()
        self.setup_iteration_range()

if __name__ == '__main__':
    
    from neutorch.dataset.patch import collate_batch
    from torch.utils.data import DataLoader
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

    from neutorch.model.io import log_tensor
    from torch.utils.tensorboard import SummaryWriter
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
        assert torch.any(target > 0.5)

        log_tensor(writer, 'train/image', image, n)
        log_tensor(writer, 'train/target', target, n)

        sleep(1)