import os
from warnings import warn
from functools import cached_property

from skimage.exposure import rescale_intensity

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
            skip_classes: list = None,
            selected_classes: list = None,
            image_intensity_rescale_range: tuple = None):
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
        self.image_intensity_rescale_range = image_intensity_rescale_range
        
        if skip_classes is not None:
            skip_classes = sorted(skip_classes, reverse=True)
        self.skip_classes = skip_classes

        self.selected_classes = selected_classes

        # these two functions should always be put in the end
        # since they call some functions and properties
        self.compute_sample_weights()
        self.setup_iteration_range()

    def _skip_label_classes(self, label: Chunk):
        for class_idx in self.skip_classes:
            label.array[label.array>class_idx] -= 1
        return label

    def _select_class(self, label: Chunk):
        """select only one class

        Args:
            label (Chunk): the label chunk with semantic annotation

        Returns:
            label (Chunk): inplace computation  
        """
        if self.selected_classes is not None:
            label.array = np.isin(label.array, self.selected_classes)
        return label

    @cached_property
    def samples(self):
        samples = []
        # for img_path, sem_path in zip(self.path_list[0::2], self.path_list[1::2]):
        for label_path in self.path_list:
            image_path = label_path.replace('label', 'image')
            assert 'image' in image_path, f'image path: {image_path}'
            assert 'label' in label_path, f'sem path: {label_path}'
            
            assert os.path.exists(label_path)
            label = Chunk.from_h5(label_path)
            image = Chunk.from_h5(image_path)
            
            print(f'image path: {image_path} with size {image.shape}')
            print(f'sem path: {label_path}')

            if Cartesian.from_collection(image.shape) < self.patch_size:
                warn(f'volume size is smaller than patch size: {image_path}: {image.shape}')
                continue

            if np.all(image==255):
                warn('skipping image path with all 255: ', image_path)
                continue

            if self.image_intensity_rescale_range is not None and \
                    len(self.image_intensity_rescale_range) > 0:
                low, high = self.image_intensity_rescale_range
                image.array = rescale_intensity(
                    image.array, 
                    in_range=(low, high), 
                    out_range=(0, 255)
                )
                image = image.astype(np.uint8)
            
            if np.all(image==255):
                breakpoint()
           
            # if label.ndim == 3:
            #     label.array = np.expand_dims(label.array, axis=0)

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
                self._skip_label_classes(label)

            if self.selected_classes is None:
                # make the starting label from 0
                label -= 1
            else:
                self._select_class(label)
            
            # assert label.max() <= self.num_classes, f'maximum number of label: {label.max()}'

            # CrossEntropyLoss only works with int64 data type!
            # uint8 will not work
            label = label.astype(np.float32)
            images = [image,]

            # breakpoint()
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
        # if label.ndim == 5:
        #     # the CrossEntropyLoss do not require channel axis
        #     label = np.squeeze(label, axis=1)
        
        # transform to PyTorch Tensor
        # transfer to device, e.g. GPU, automatically.
        image = to_tensor(image)
        target = to_tensor(label)

        return image, target
    
    @cached_property
    def transform(self):
        return Compose([
            NormalizeTo01(probability=1.),
            AdjustContrast(factor_range = (0.95, 1.5)),
            AdjustBrightness(min_factor = 0.05, max_factor = 0.2),
            Gamma(),
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

    VOLUME_NUM = 1
    PATCH_NUM = 100

    from yacs.config import CfgNode

    #cfg_file = '/mnt/home/jwu/ceph/31_organelle/28_net/config.yaml'
    cfg_file = './config.yaml'
    with open(cfg_file) as file:
        cfg = CfgNode.load_cfg(file)
    cfg.freeze()
    
    from glob import glob    
    glob_path = os.path.expanduser(cfg.dataset.glob_path)
    path_list = glob(glob_path, recursive=True)
    # path_list = sorted(path_list)
    from random import shuffle
    shuffle(path_list)
    if VOLUME_NUM > 0:
        path_list = path_list[:VOLUME_NUM]

    dataset = OrganelleDataset(
            path_list,
            patch_size=cfg.train.patch_size,
            num_classes=cfg.model.out_channels,
            skip_classes=cfg.dataset.skip_classes,
            selected_classes=cfg.dataset.selected_classes,
            image_intensity_rescale_range=cfg.dataset.rescale_intensity,
        )
    
    from PIL import Image
    OUT_DIR = os.path.expanduser('~/Downloads/patches/')
    from tqdm import tqdm

    for idx in tqdm(range(PATCH_NUM)):
        image, label = dataset.random_patch
        
        section_idx = image.shape[-3]//2
        image = image[0,0, section_idx, :,:]
        label = label[0,0, section_idx, :,:]

        image *= 255.
        im = Image.fromarray(image).convert('L')
        im.save(os.path.join(OUT_DIR, f'{idx}_image.jpg'))

        label *= 255.
        lbl = Image.fromarray(label).convert('L')
        lbl.save(os.path.join(OUT_DIR, f'{idx}_label.jpg'))

