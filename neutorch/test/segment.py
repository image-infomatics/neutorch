import waterz as wz
import numpy as np

from chunkflow.lib.cartesian_coordinate import Cartesian 
from chunkflow.chunk import Chunk
from chunkflow.volume import load_chunk_or_volume
from chunkflow.volume import PrecomputedVolume, AbstractVolume

class segment_methodology():
    def __init__(self, 
                 affinity_paths: list,
                 ground_truth_paths: list):
    
        super().__init__()
        self.affinity_paths = affinity_paths
        self.ground_truth_paths = ground_truth_paths

    @classmethod
    def agglomerate(self, affs_paths, gt_paths, **kwargs):
        segmentations = []
        
        seg_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for aff_path, gt_path in zip(affs_paths, gt_paths): 
            affs = load_chunk_or_volume(aff_path, **kwargs) 
            gt = load_chunk_or_volume(gt_path, **kwargs) 

            assert affs.shape[-3:] == gt.shape[-3:]
            
            breakpoint()

            gt_array = gt.array.astype('float32')
            assert affs.shape[-3:] == gt.shape[-3:]
            assert affs.array.dtype == gt_array.dtype

            segmentation = wz.agglomerate(affs.array, seg_thresholds, gt=gt.array, 
                                          fragments=None, aff_threshold_low=0.0001, 
                                          aff_threshold_high=0.9999, return_merge_history=True, 
                                          return_region_graph=False)
            
            segmentations.append(segmentation) 

        return segmentations
    
    @classmethod
    def evaluate(self, affs_paths, gt_paths, **kwargs):
        segmentations = []
        
        for aff_path, gt_path in zip(affs_paths, gt_paths): 
            affs = load_chunk_or_volume(aff_path, **kwargs) 
            gt = load_chunk_or_volume(gt_paths, **kwargs) 

            gt_array = gt.array.astype('float32')
            assert affs.shape[-3:] == gt.shape[-3:]
            assert affs.array.dtype == gt_array.dtype

            segmentation = wz.evaluate(affs.array, gt_array)
            segmentations.append(segmentation) 

        return segmentations

if __name__ == '__main__':

    affs_paths = ["/mnt/home/mpaez/ceph/affsmaptrain/sample2/affstrain2_28000_vol3.h5"]
    gt_paths = ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07338/seg_v4_filled.h5"]

    segmentation = segment_methodology.agglomerate(affs_paths, gt_paths) 

    for seg in segmentation:
        [x for x in seg] 



