import math
import random
from functools import cached_property

import numpy as np
import torch
from chunkflow.lib.cartesian_coordinate import Cartesian
from yacs.config import CfgNode

from neutorch.data.sample import *
from neutorch.data.transform import *

DEFAULT_PATCH_SIZE = Cartesian(128, 128, 128)

#training data
affs_train = [["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_01700/affs_160k.h5",], 
              ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_04900/affs_160k.h5",],              
              ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07338/affs_160k.h5",],
              ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07580/affs_160k.h5",],
              ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07800/affs_160k.h5",],
              # ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_09300/affs_160k.h5",],
              ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_12350/affs_160k.h5",],
              ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_13170_2/affs_160k.h5",],
              ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt1/affs_160k.h5",],
              ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt2/affs_160k.h5",]
              ] 

label_train = [ "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_01700/label_v3.h5", 
               "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_04900/label_v4.h5",
               "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07338/label_v9.h5",
               "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07580/label_v2.h5",
               "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07800/label_v2.h5",
               # "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_09300/label_v2.h5",
               "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_12350/label_v2.h5",
               "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_13170_2/label_v2.h5",
               "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt1/label_v2.h5",
               "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt2/label_v2.h5"
               ]

#validation data
affs_valid = [["/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/31_test_3072-3584_5120-5632_8196-8708/aff_zyx_3072-3584_5120-5632_8196-8708.h5",], 
                ["/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/32_test_5120-5632_5632-6144_10240-10752/aff_zyx_5120-5632_5632-6144_10240-10752.h5",], 
                ["/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/33_test_2560-3072_5632-6144_8704-9216/aff_zyx_2560-3072_5632-6144_8704-9216.h5",], 
                ["/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/41_test_2560-3584_5120-6144_8192-9216/aff_zyx_2560-3584_5120-6144_8192-9216.h5",],
                # ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt2/affs_160k.h5",] 
                ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_09300/affs_160k.h5",]
                ]

label_valid = ["/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/31_test_3072-3584_5120-5632_8196-8708/seg_zyx_3072-3584_5120-5632_8196-8708.h5", \
            "/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/32_test_5120-5632_5632-6144_10240-10752/seg_zyx_5120-5632_5632-6144_10240-10752.h5", \
            "/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/33_test_2560-3072_5632-6144_8704-9216/seg_zyx_2560-3072_5632-6144_8704-9216.h5", \
            "/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/41_test_2560-3584_5120-6144_8192-9216/seg_zyx_2560-3584_5120-6144_8192-9216.h5", \
            # "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt2/label_v2.h5", 
            "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_09300/affs_160k.h5" ]


def get_iter_range(sample_num: int) -> tuple[int, int]:
    # multiprocess data loading 
    worker_info = torch.utils.data.get_worker_info()
    iter_start = 0
    iter_stop = sample_num
    if worker_info is not None:
        # multiple processing for data loading, split workload
        worker_id = worker_info.id
        if sample_num > worker_info.num_workers:
            per_worker = int(math.ceil(
                (iter_end - iter_start) / float(worker_info.num_workers)))
            iter_start = iter_start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, iter_end)
        else:
            iter_start = worker_id % sample_num
            iter_stop = iter_start + 1
    
    return iter_start, iter_stop

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
        # Pytorch only supports types: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        if np.issubdtype(arr.dtype, np.uint16):
            arr = arr.astype(np.int32)
        elif np.issubdtype(arr.dtype, np.uint64):
            arr = arr.astype(np.int64)
        arr = torch.tensor(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr


class DatasetBase(torch.utils.data.Dataset):
    def __init__(self,
            samples: List[AbstractSample], 
        ):
        """
        Parameters:
            patch_size (int or tuple): the patch size we are going to provide.
        """
        super().__init__()
        self.samples = samples


    @cached_property
    def sample_num(self):
        return len(self.samples)
    
    @cached_property
    def sample_weights(self):
        """use the number of candidate patches as volume sampling weight
        if there is None, replace it with average value
        """
        sample_weights = []
        for sample in self.samples:
            sample_weights.append(sample.sampling_weight)

        # replace None weight with average weight 
        ws = []
        for x in sample_weights:
            if x is not None:
                ws.append(x)
        average_weight = np.mean(ws)
        for idx, w in enumerate(sample_weights):
            if w is None:
                sample_weights[idx] = average_weight 
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
    
    def __len__(self):
        patch_num = 0
        for sample in self.samples:
            patch_num += len(sample)
        return patch_num

    def __getitem__(self, index: int):
        """return a random patch from a random sample
        the exact index does not matter!

        Args:
            index (int): index of the patch
        """
        #breakpoint() 
        image_chunk, label_chunk = self.random_patch
        image = to_tensor(image_chunk.array)
        label = to_tensor(label_chunk.array)

        return image, label

class SemanticDataset(DatasetBase):
    def __init__(self, samples: list):
            #patch_size: Cartesian = DEFAULT_PATCH_SIZE):
        super().__init__(samples)
    
    @classmethod
    def from_config(cls, cfg: CfgNode, is_train: bool, **kwargs): 
        """Construct a semantic dataset with chunk or volume

        Args:
            cfg (CfgNode): _description_
            is_train (bool): _description_

        Returns:
            _type_: _description_
        """
        if is_train:
            name2chunks = cfg.dataset.training
        else:
            name2chunks = cfg.dataset.validation

        samples = []
        for name2path in name2chunks.values():
            sample = SemanticSample.from_explicit_dict(
                    name2path, 
                    output_patch_size=cfg.train.patch_size,
                    num_classes=cfg.model.out_channels,
                    **kwargs)
            samples.append(sample)

        return cls( samples )

    

class OrganelleDataset(SemanticDataset):
    def __init__(self, samples: List[AbstractSample], 
            num_classes: int = 1,
            skip_classes: list = None,
            selected_classes: list = None):
        """Dataset for organelle semantic segmentation

        Args:
            paths (list): list of samples
            sample_name_to_image_versions (dict, optional): map sample name to image volumes. Defaults to None.
            patch_size (Cartesian, optional): size of a output patch. Defaults to Cartesian(128, 128, 128).
            num_classes (int, optional): number of semantic classes to be classified. Defaults to 1.
            skip_classes (list, optional): skip some classes in the label. Defaults to None.
        """
        super().__init__(samples)

        self.samples = samples
        self.num_classes = num_classes
        
        if skip_classes is not None:
            skip_classes = sorted(skip_classes, reverse=True)
        self.skip_classes = skip_classes

        self.selected_classes = selected_classes

    @classmethod
    def from_path_list(cls, path_list: list,
            patch_size: Cartesian = Cartesian(128, 128, 128),
            num_classes: int = 1,
            skip_classes: list = None,
            selected_classes: list = None):
        path_list = sorted(path_list)
        samples = []
        # for img_path, sem_path in zip(self.path_list[0::2], self.path_list[1::2]):
        for label_path in path_list:
            sample = OrganelleSample.from_label_path(
                label_path, 
                num_classes, 
                patch_size=self.patch_size_before_transform,
                skip_classes=skip_classes,
                selected_classes = selected_classes,
            ) 
            
            samples.append(sample)
        
        return cls(samples, patch_size, num_classes, skip_classes, selected_classes)

    @cached_property
    def voxel_num(self):
        voxel_nums = [sample.voxel_num for sample in self.samples]
        return sum(voxel_nums)

    @cached_property
    def class_counts(self):
        counts = np.zeros((self.num_classes,), dtype=np.int)
        for sample in self.samples:
            counts += sample.class_counts

        return counts
     
    def __next__(self):
        # get numpy arrays of image and label
        image, label = self.random_patch
        # if label.ndim == 5:
        #     # the CrossEntropyLoss do not require channel axis
        #     label = np.squeeze(label, axis=1)
        
        # transform to PyTorch Tensor
        # transfer to device, e.g. GPU, automatically.
        image = to_tensor(image)
        target = to_tensor(label)

        return image, target

class AffinityMapVolumeWithMask(DatasetBase):
    def __init__(self, samples: List[AbstractSample]):
        super().__init__(samples)

    @classmethod
    def from_config(cls, cfg: CfgNode, mode: str = 'training', **kwargs):
        """construct affinity map volume with mask dataset

        Args:
            cfg (CfgNode): the configuration node
            mode (str, optional): ['training', 'validation', 'test']. Defaults to 'train'.
        """
        output_patch_size = Cartesian.from_collection(
            cfg.train.patch_size)
        
        sample_names = cfg.dataset[mode]

        iter_start, iter_stop = get_iter_range(sample_names)

        samples = []
        for sample_name in sample_names[iter_start : iter_stop]:
            sample_cfg = cfg.samples[sample_name]
            sample_class = eval(sample_cfg.type)
            sample = sample_class.from_config(
                sample_cfg, output_patch_size)
            samples.append(sample)
        return cls(samples)

    def __len__(self):
        # num = 0
        # for sample in self.samples:
        #     num += sample.
        # return num
        # return self.sample_num * cfg.system.gpus * cfg.train.batch_size * 16
        # our patches are randomly sampled from chunk or volume and is close to unlimited.
        # return a big enough number to make distributed sampler work
        return 1024

class AffinityMapDataset(DatasetBase):
    def __init__(self, samples: list):
        super().__init__(samples)
    
    @classmethod
    def from_config(cls, cfg: CfgNode, mode: str, **kwargs):
        """Construct a semantic dataset with chunk or volume

        Args:
            cfg (CfgNode): configuration in YAML file 
            mode (str): training mode or validation mode.
        """
        output_patch_size = Cartesian.from_collection(cfg.train.patch_size)
        sample_names = cfg.dataset[mode]

        sample_configs = CfgNode()
        #breakpoint() 
        for sample_config_path in cfg.samples:
            sample_cfg_node = load_cfg(sample_config_path, freeze=False)
            sample_config_dir = os.path.dirname(sample_config_path)
            for sample_node in sample_cfg_node.values():
                sample_node.dir = os.path.join(sample_config_dir, sample_node.dir)

            sample_configs.update(sample_cfg_node)

        sample_num = len(sample_names)
        iter_start, iter_stop = get_iter_range(sample_num)

        samples = []
        for sample_name in sample_names[iter_start : iter_stop]:
            sample_node = sample_configs[sample_name]
            sample = AffinityMapSample.from_config_node(
                sample_node, output_patch_size,
                num_classes=cfg.model.out_channels,
            )
            samples.append(sample)

        return cls( samples )
    
class BoundaryAugmentationDataset(DatasetBase):
    def __init__(self, samples: list):
        super().__init__(samples)
    
    @classmethod
    def from_config(cls, cfg: CfgNode, mode: str, **kwargs):
        """Construct a semantic dataset with chunk or volume

        Args:
            cfg (CfgNode): configuration in YAML file 
            mode (str): training mode or validation mode.
        """

        output_patch_size = Cartesian(128, 128, 128) 

        if mode == "training": 
            image_sample_paths = affs_train
            label_sample_paths = label_train
        elif mode == "validation":
            image_sample_paths = affs_valid
            label_sample_paths = label_valid

        assert len(image_sample_paths) == len(label_sample_paths)
        # iter_start, iter_stop = get_iter_range(sample_num)
        
        """
        for image_fname, label_path in zip(image_sample_paths, label_sample_paths):
            label = load_chunk_or_volume(label_path, **kwargs) 
            for image_path in image_fname:
                image = load_chunk_or_volume(image_path, **kwargs)
                print(f'image shape: {image.shape}, label shape: {label.shape}, file name: {image_path}')
                #print(f'image voxel offset: {image.voxel_offset}, label voxel offset: {label.voxel_offset}, file name: {image_path}')
        """

        samples = []
        for image_fname, label_path in zip(image_sample_paths, label_sample_paths):          
            # breakpoint() 
            label = load_chunk_or_volume(label_path, **kwargs) 
            images = []
            #breakpoint() 
            for image_path in image_fname:
                image = load_chunk_or_volume(image_path, **kwargs)
                assert image.shape[-3:] == label.shape[-3:], \
                    f'image shape: {image.shape}, label shape: {label.shape}, file name: {image_path}'
                assert image.voxel_offset == label.voxel_offset, \
                    f'image voxel offset: {image.voxel_offset}, label voxel offset: {label.voxel_offset}, file name: {image_path}'
                images.append(image)
    
            sample = AffinityMapSample( 
                images=images,
                label=label,
                output_patch_size = output_patch_size,
                num_classes=3,
            )

            samples.append(sample)

        return cls( samples )
    
"""
class BoundaryAugmentationDataset(DatasetBase): 
    def __initi__(self, samples: list):
        super.__init__(samples)

    @classmethod
    def from_config(cls, cfg: CfgNode, is_train: bool, **kwargs):
        """ #Construct a semantic dataset with chunk or volume.
"""if is_train:
            name2chunks = cfg.dataset.training
        else:
            name2chunks = cfg.dataset.validation

        samples = []
        for type_name2paths in name2chunks.values():
            paths = [x for x in type_name2paths.values()][0]
            sample = AffinityMapSampleWithMask.from_explicit_paths(
                    paths,
                    output_patch_size=cfg.train.patch_size,
                    num_classes=cfg.model.out_channels,
                    **kwargs)
            samples.append(sample)

        return cls( samples )
"""
    
if __name__ == '__main__':

    from yacs.config import CfgNode

    cfg_file = '/mnt/home/jwu/wasp/jwu/15_rna_granule_net/11/config.yaml'
    with open(cfg_file) as file:
        cfg = CfgNode.load_cfg(file)
    cfg.freeze()

    sd = AffinityMapDataset(
        path_list=['/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_01700/rna_v1.h5'],
        sample_name_to_image_versions=cfg.dataset.sample_name_to_image_versions,
        patch_size=Cartesian(128, 128, 128),
    )
    
