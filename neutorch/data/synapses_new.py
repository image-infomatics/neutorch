import os
from functools import cached_property
from time import sleep, time
from typing import List, Union

import numpy as np
import torch
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.lib.synapses import Synapses
from chunkflow.volume import Volume
from neutorch.dataset.ground_truth_sample import PostSynapseGroundTruth
from neutorch.dataset.transform import *
from sklearn import datasets

from .base import DatasetBase
from .ground_truth_sample import GroundTruthSampleWithPointAnnotation

#create a synthesis path
def syns_path_to_dataset_name(syns_path: str, dataset_names: list):
    for dataset_name in dataset_names:
        if dataset_name in syns_path:
            return dataset_name

#own function
def syns_path_to_images(syns_path: str, vols, sample_name_image_versions, bbox:BoundingBox):
    images = [] 
    dataset_name = syns_path_to_dataset_name(
        syns_path,
        sample_name_image_versions.keys()
    )

    for vol in vols[dataset_name]:
        image = vol.cutout(bbox)
        images.append(image)
    return images 

'''
def volumes(syns_path: str, 
'''
class PreSynapsesDataset(DatasetBase): 
    def __init__(self, 
            syns_path_list: List[str],
            sample_name_to_image_versions: dict, #can we change this
            patch_size: Union[int, tuple, Cartesian]=Cartesian(128, 128, 128),
        ):
        
        super().__init__(patch_size) #to DatasetBase class
        """
        Parameters:
            syns_path_list (List[str]): the synapses file list
            sample_name_to_image_versions (dict): map the sample or volume name to a list of versions the same dataset.
            patch_size (int or tuple): the patch size we are going to provide.
        """
        self.sample_name_to_image_versions = sample_name_to_image_versions #good

        self.vols = {} #make a list
        for dataset_name, dir_list in sample_name_to_image_versions.items():
            vol_list = []
            for dir_path in dir_list:
                vol = Volume.from_cloudvolume_path(
                    'file://' + dir_path,
                    bounded = True,
                    fill_missing = True,
                    parallel=True,
                )
                vol_list.append(vol)
            self.vols[dataset_name] = vol_list

        self.samples = []

        for syns_path in syns_path_list:
            bbox = BoundingBox.from_string(syns_path)
            images = syns_path_to_images(syns_path, sample_name_to_image_versions, bbox)
            
            synapses = Synapses.from_h5(syns_path)
            synapses.remove_synapses_outside_bounding_box(bbox)
            
            pre = synapses.pre 
            pre -= np.asarray(bbox.start, dtype=pre.dtype) 

            sample = GroundTruthSampleWithPointAnnotation(
                images,
                annotation_points=pre,
                patch_size=self.patch_size_before_transform, #how to get this from base
            )
            self.samples.append(sample)

        super().compute_sample_weights()
        super().setup_iteration_range() #calling to super class
        super().transform()

class PostSynapsesDataset(DatasetBase):
    def __init__(self, 
            syns_path_list: List[str],
            sample_name_to_image_versions: dict,
            patch_size: Cartesian = Cartesian(256, 256, 256), 
        ):
        """postsynapse dataset

        Args:
            syns_path_list (List[str]): the synapses file list
            sample_name_to_image_versions (dict): map the sample or volume name to a list of versions the same dataset.
            patch_size (Cartesian, optional): Defaults to Cartesian(256, 256, 256).

        Raises:
            ValueError: [description]
        """
        super().__init__(patch_size=patch_size)

        self.sample_name_to_image_versions = sample_name_to_image_versions #good

        ''' #goal next time: figure out a way to turn this into a function -> that way we don't have to repeatedly call it
        self.vols = {} #make a list
        for dataset_name, dir_list in sample_name_to_image_versions.items():
            vol_list = []
            for dir_path in dir_list:
                vol = Volume.from_cloudvolume_path(
                    'file://' + dir_path,
                    bounded = True,
                    fill_missing = True,
                    parallel=True,
                )
                vol_list.append(vol)
            self.vols[dataset_name] = vol_list

        self.samples = []
        '''

        for syns_path in syns_path_list:
            synapses = Synapses.from_file(syns_path)
            if synapses.post is None:
                print(f'skip synapses without post: {syns_path}')
                continue
            print(f'loaded {syns_path}')
            synapses.remove_synapses_without_post()

            # bbox = BoundingBox.from_string(syns_path) #why?
            bbox = synapses.pre_bounding_box
            bbox.adjust(self.patch_size_before_transform // 2)

            images = syns_path_to_images(syns_path, sample_name_to_image_versions, bbox)
            sample = PostSynapseGroundTruth(
                synapses, images,
                patch_size=self.patch_size_before_transform
            )
            self.samples.append(sample)

        super().compute_sample_weights(self) #should we send this to the other class
        super().setup_iteration_range(self)
        super().transform(self)

if __name__ == '__main__':
    
    from neutorch.dataset.patch import collate_batch
    from torch.utils.data import DataLoader
    dataset = datasets(
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
        # assert np.any(target > 0.5)
        assert torch.any(target > 0.5)

        log_tensor(writer, 'train/image', image, n)
        log_tensor(writer, 'train/target', target, n)

        sleep(1)