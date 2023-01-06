import os
from functools import cached_property
from time import sleep, time
from typing import List, Union

import numpy as np
import torch
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.lib.synapses import Synapses
from chunkflow.volume import Volume

from .dataset import DatasetBase, SemanticDataset, path_to_dataset_name
from .sample import PostSynapseReference, SampleWithPointAnnotation
from .transform import *

DEFAULT_PATCH_SIZE = Cartesian(128, 128, 128)

"""
def vols_dir(sample, etc):
    vols = {} 
    for dataset_name, dir_list in sample_name_to_image_versions.items(): #fix
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
""" #where there is an error

class SynapsesDataSetBase(SemanticDataset): #call from SemanticDataset
    def __init__(self, samples: list, 
            patch_size: Union[int, tuple, Cartesian] = DEFAULT_PATCH_SIZE): #current parameters -> please check to add more
        super().__init__(samples)

        self.patch_size = patch_size

    def syns_path_to_images(self, syns_path: str, bbox: BoundingBox):
        images = []
        dataset_name = path_to_dataset_name(
            syns_path, 
            self.sample_name_to_image_versions.keys() #okay how should this be implemented
        ) 
        for vol in self.vols[dataset_name]:
            image = vol.cutout(bbox)
            images.append(image)
        return images   

class PreSynapsesDataset(SynapsesDataSetBase):
    def __init__(self, syns_path_list: List[str],
            samples: list,
            patch_size: Union[int, tuple, Cartesian]=DEFAULT_PATCH_SIZE, 
        ):

        if isinstance(patch_size, int):
            patch_size = Cartesian(patch_size, ) * 3
        else:
            patch_size = Cartesian.from_collection(patch_size)
        super().__init__(samples=samples, patch_size=patch_size)

        self._prepare_transform()
        assert isinstance(self.transform, Compose) #can this be edited

        self.patch_size = patch_size
        patch_size_before_transform = tuple(
            p + s0 + s1 for p, s0, s1 in zip(
                patch_size, 
                self.transform.shrink_size[:3], 
                self.transform.shrink_size[-3:]
            )
        )

        self.patch_size_before_transform = Cartesian.from_collection(patch_size_before_transform)

        for syns_path in syns_path_list:
            bbox = BoundingBox.from_string(syns_path)
            images = self.syns_path_to_images(syns_path, bbox)
            
            synapses = Synapses.from_h5(syns_path)
            synapses.remove_synapses_outside_bounding_box(bbox)
            
            pre = synapses.pre 
            pre -= np.asarray(bbox.start, dtype=pre.dtype)

            sample = SampleWithPointAnnotation(
                images,
                annotation_points=pre,
                output_patch_size=patch_size_before_transform,
            )
            self.samples.append(sample)

    #another way to place this
    def _prepare_transform(self): #need this function to attach self.transform to compose
        self.transform = Compose([
            NormalizeTo01(probability=1.),
            IntensityPerturbation(), #this still comes as error -> what should I do to solve it?
            # AdjustBrightness(), #u add tehse right?
            # AdjustContrast(),
            # Gamma(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            MaskBox(),
            Perspective2D(),
            # RotateScale(probability=1.),
            DropSection(),
            Flip(),
            Transpose(),
            MissAlignment(),
        ])

class PostSynapsesDataSet(SynapsesDataSetBase):
    def __init__(self, syns_path_list: List[str],
            samples: list,
            patch_size: Union[int, tuple, Cartesian]=DEFAULT_PATCH_SIZE,
        ):

        super().__init__(samples=samples, patch_size=patch_size)

        for syns_path in syns_path_list: #alright
            synapses = Synapses.from_file(syns_path)
            """
            if synapses.post is None:
                print(f'skip synapses without post: {syns_path}')
                continue
            print(f'loaded {syns_path}')
            """
            synapses.remove_synapses_without_post()

            # bbox = BoundingBox.from_string(syns_path)
            bbox = synapses.pre_bounding_box
            bbox.adjust(self.patch_size_before_transform // 2)

            images = self.syns_path_to_images(syns_path, bbox)
            sample = PostSynapseReference(
                synapses, 
                images,
                patch_size = self.patch_size_before_transform
            )
            self.samples.append(sample)
