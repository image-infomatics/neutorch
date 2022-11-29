import os
from functools import cached_property

from tqdm import tqdm

from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import Cartesian
from chunkflow.volume import Volume

from neutorch.dataset.base import DatasetBase, path_to_dataset_name
from neutorch.dataset.ground_truth_sample import GroundTruthSample
from neutorch.dataset.transform import *


class SemanticDataset(DatasetBase):
    def __init__(self, path_list: list, 
            sample_name_to_image_versions: dict,
            patch_size: Cartesian = Cartesian(128, 128, 128)):
        super().__init__(patch_size=patch_size)

        self.path_list = path_list
        self.sample_name_to_image_versions = sample_name_to_image_versions

        self.vols = {}
        for dataset_name, dir_list in sample_name_to_image_versions.items():
            vol_list = []
            for dir_path in dir_list:
                vol = Volume.from_cloudvolume_path(
                    'file://' + dir_path,
                    bounded = True,
                    fill_missing = False,
                    parallel = True,
                    green_threads = False,
                )
                vol_list.append(vol)
            self.vols[dataset_name] = vol_list

        self.compute_sample_weights()
        self.setup_iteration_range()

    @cached_property
    def samples(self):
        samples = []
        for sem_path in tqdm(self.path_list):
            assert os.path.exists(sem_path)
            sem = Chunk.from_h5(sem_path)

            images = []
            dataset_name = path_to_dataset_name(
                sem_path,
                self.sample_name_to_image_versions.keys()
            )
            for vol in self.vols[dataset_name]:
                image = vol.cutout(sem.bbox)
                images.append(image)

            target = (sem.array>0)
            target = target.astype(np.float32)
            sample = GroundTruthSample(
                images,
                target=target, 
                patch_size=self.patch_size_before_transform
            )
            samples.append(sample)
        
        return samples

    def _prepare_transform(self):
        self.transform = Compose([
            NormalizeTo01(probability=1.),
            IntensityPerturbation(),
            OneOf([
                Noise(),
                GaussianBlur2D(),
            ]),
            BlackBox(),
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

    sd = SemanticDataset(
        path_list=['/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_01700/rna_v1.h5'],
        sample_name_to_image_versions=cfg.dataset.sample_name_to_image_versions,
        patch_size=Cartesian(128, 128, 128),
    )
    
    # print(sd.samples)

