import os
from functools import cached_property

# from tqdm import tqdm

from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import Cartesian

from neutorch.dataset.base import DatasetBase, to_tensor
from neutorch.dataset.ground_truth_sample import SemanticSample
from neutorch.dataset.transform import *


class OrganelleDataset(DatasetBase):
    def __init__(self, path_list: list, 
            sample_name_to_image_versions: dict = None,
            patch_size: Cartesian = Cartesian(128, 128, 128),
            num_classes: int = 1,
            skip_classes: list = None):
        """Dataset for organelle semantic segmentation

        Args:
            path_list (list): list of label files
            sample_name_to_image_versions (dict, optional): map sample name to image volumes. Defaults to None.
            patch_size (Cartesian, optional): size of a patch. Defaults to Cartesian(128, 128, 128).
            num_classes (int, optional): number of semantic classes to be classified. Defaults to 1.
            skip_classes (list, optional): skip some classes in the label. Defaults to None.
        """
        super().__init__(patch_size=patch_size)

        self.path_list = sorted(path_list)
        # print(f'path list: {self.path_list}')
        self.sample_name_to_image_versions = sample_name_to_image_versions
        self.num_classes = num_classes
        
        if skip_classes is None:
            self.skip_classes = None
        else:
            self.skip_classes = sorted(skip_classes, reverse=True)

        # these two functions should always be put in the end
        # since they call some functions and properties
        self.compute_sample_weights()
        self.setup_iteration_range()


    def _compress_label_classes(self, label: Chunk):
        for class_idx in self.skip_classes:
            label.array[label.array>class_idx] -= 1
        return label

    @cached_property
    def samples(self):
        samples = []
        # for img_path, sem_path in zip(self.path_list[0::2], self.path_list[1::2]):
        for label_path in self.path_list:
            image_path = label_path.replace('label', 'image')
            print(f'image path: {image_path}')
            print(f'sem path: {label_path}')

            assert 'image' in image_path, f'image path: {image_path}'
            assert 'label' in label_path, f'sem path: {label_path}'
            
            assert os.path.exists(label_path)
            label = Chunk.from_h5(label_path)
            image = Chunk.from_h5(image_path)

            # label start from 1
            # it should be converted to start from 0
            if label.min() <= 0:
                breakpoint()
            if np.any(label == 31):
                breakpoint()
            if np.any(label == 34):
                breakpoint()
           
            # some classes could be unlabeled!
            # we also would like to reduce the number of classes for easier training
            if self.skip_classes is not None:
                self._compress_label_classes(label)

            if label.max() > self.num_classes:
                breakpoint()
            assert label.max() <= self.num_classes, f'maximum number of label: {label.max()}'
 
            label -= 1
            # CrossEntropyLoss only works with int64 data type!
            # uint8 will not work
            label = label.astype(np.int64)
            images = [image,]

            sample = SemanticSample(
                images,
                label, 
                self.num_classes,
                patch_size=self.patch_size_before_transform
            )
            samples.append(sample)
        
        return samples

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
        if label.ndim == 5:
            # the CrossEntropyLoss do not require channel axis
            label = np.squeeze(label, axis=1)
        
        if np.mean(image)> 0.9:
            breakpoint()
        # transform to PyTorch Tensor
        # transfer to device, e.g. GPU, automatically.
        image = to_tensor(image)
        target = to_tensor(label)

        return image, target

    def _prepare_transform(self):
        self.transform = Compose([
            NormalizeTo01(probability=1.),
            IntensityPerturbation(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            MaskBox(),
            Perspective2D(),
            # RotateScale(probability=1.),
            # DropSection(),
            Flip(),
            Transpose(),
            MissAlignment(),
        ])


if __name__ == '__main__':

    from yacs.config import CfgNode

    cfg_file = '/mnt/home/jwu/wasp/jwu/15_rna_granule_net/11/config.yaml'
    with open(cfg_file) as file:
        cfg = CfgNode.load_cfg(file)
    cfg.freeze()

    sd = OrganelleDataset(
        path_list=['/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_01700/rna_v1.h5'],
        sample_name_to_image_versions=cfg.dataset.sample_name_to_image_versions,
        patch_size=Cartesian(128, 128, 128),
    )
    
    # print(sd.samples)

