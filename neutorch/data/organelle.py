import os
from functools import cached_property

from chunkflow.chunk.image import Image
from chunkflow.lib.cartesian_coordinate import Cartesian

from neutorch.data.dataset import DatasetBase, to_tensor
from neutorch.data.sample import OrganelleSample
from neutorch.data.transform import *


class OrganelleDataset(DatasetBase):
    def __init__(self, samples: list, 
            patch_size: Cartesian = Cartesian(128, 128, 128),
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
        super().__init__(patch_size=patch_size)

        self.samples = samples
        self.num_classes = num_classes
        
        if skip_classes is not None:
            skip_classes = sorted(skip_classes, reverse=True)
        self.skip_classes = skip_classes

        self.selected_classes = selected_classes

        # these two functions should always be put in the end
        # since they call some functions and properties
        self.compute_sample_weights()
        self.setup_iteration_range()
    
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
    
    @cached_property
    def transform(self):
        return Compose([
            NormalizeTo01(probability=1.),
            # AdjustContrast(factor_range = (0.95, 1.8)),
            # AdjustBrightness(min_factor = 0.05, max_factor = 0.2),
            AdjustContrast(),
            AdjustBrightness(),
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
            # MissAlignment(),
        ])


if __name__ == '__main__':

    VOLUME_NUM = 10
    PATCH_NUM = 200


    #cfg_file = '/mnt/home/jwu/ceph/31_organelle/28_net/config.yaml'
    cfg_file = './config_mito.yaml'
    
    
    from glob import glob    
    glob_path = os.path.expanduser(cfg.dataset.glob_path)
    path_list = glob(glob_path, recursive=True)
    # path_list = sorted(path_list)
    from random import shuffle
    shuffle(path_list)
    if VOLUME_NUM > 0:
        path_list = path_list[:VOLUME_NUM]

    dataset = OrganelleDataset.from(
            path_list,
            patch_size=cfg.train.patch_size,
            num_classes=cfg.model.out_channels,
            skip_classes=cfg.dataset.skip_classes,
            selected_classes=cfg.dataset.selected_classes,
            image_intensity_rescale_range=cfg.dataset.rescale_intensity,
        )
    
    from PIL import Image
    OUT_DIR = os.path.expanduser('~/dropbox/patches/')
    from tqdm import tqdm

    for idx in tqdm(range(PATCH_NUM)):
        image, label = dataset.random_patch
        
        # section_idx = image.shape[-3]//2
        section_idx = 0
        image = image[0,0, section_idx, :,:]
        label = label[0,0, section_idx, :,:]

        image *= 255.
        im = Image.fromarray(image).convert('L')
        im.save(os.path.join(OUT_DIR, f'{idx}_image.jpg'))

        label *= 255.
        lbl = Image.fromarray(label).convert('L')
        lbl.save(os.path.join(OUT_DIR, f'{idx}_label.jpg'))

