import os
from collections import OrderedDict
from functools import cached_property
from time import sleep, time
from typing import List, Union

import numpy as np
import torch
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.lib.synapses import Synapses
from chunkflow.volume import Volume
from scipy.stats import describe

from neutorch.dataset.ground_truth_sample import PostSynapseGroundTruth
from neutorch.dataset.transform import *

from .base import *
from .ground_truth_sample import *

#create a synthesis path
def syns_path_to_dataset_name(syns_path: str, dataset_names: list):
    for dataset_name in dataset_names:
        if dataset_name in syns_path:
            return dataset_name

class PreSynapsesDataset(DatasetBase): 
    def __init__(self, 
            syns_path_list: List[str],
            sample_name_to_image_versions: dict, #can we change this
            patch_size: Union[int, tuple, Cartesian]=Cartesian(128, 128, 128),
        ):

        """
        Parameters:
            syns_path_list (List[str]): the synapses file list
            sample_name_to_image_versions (dict): map the sample or volume name to a list of versions the same dataset.
            patch_size (int or tuple): the patch size we are going to provide.
        """
        super().__init__(patch_size)
        
        #to extract volumes
        self.sample_name_to_image_versions = sample_name_to_image_versions

        self.vols = {}
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

        #following from DatasetBase -> can this be deleted?

        #below to work on

        patch_size_before_transform = Cartesian.from_collection(patch_size_before_transform)

        def syns_path_to_images(self, syns_path: str, bbox: BoundingBox):
            images = []
            dataset_name = syns_path_to_dataset_name(
                syns_path, 
                self.sample_name_to_image_versions.keys()
            )
            for vol in self.vols[dataset_name]:
                image = vol.cutout(bbox)
                images.append(image)
            return images

        for syns_path in syns_path_list:
            bbox = BoundingBox.from_string(syns_path)
            images = self.syns_path_to_images(syns_path, bbox)
            
            synapses = Synapses.from_h5(syns_path)
            synapses.remove_synapses_outside_bounding_box(bbox)
            
            pre = synapses.pre 
            pre -= np.asarray(bbox.start, dtype=pre.dtype)

            sample = GroundTruthSampleWithPointAnnotation(
                images,
                annotation_points=pre,
                patch_size=patch_size_before_transform,
            )
            self.samples.append(sample)

        super().compute_sample_weights(self)
        super().setup_iteration_range(self) #should it be self(). or super()
        super().transform(self)

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
        super().__init__(sample_name_to_image_versions, patch_size=patch_size)

        #will it be possible to make this a function in base
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
            sample = PostSynapseGroundTruth(
                synapses, images,
                patch_size=self.patch_size_before_transform
            )
            self.samples.append(sample)

        #random.shuffle on this
        super().compute_sample_weights(self)
        super().setup_iteration_range(self)
        super().transform(self)