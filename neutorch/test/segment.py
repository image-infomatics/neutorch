import waterz as wz
import numpy as np

from chunkflow.lib.cartesian_coordinate import Cartesian 
from chunkflow.chunk import Chunk
from chunkflow.volume import load_chunk_or_volume
from chunkflow.volume import PrecomputedVolume, AbstractVolume

"""                  
affinity_paths = ["/mnt/home/mpaez/ceph/affsmaptrain/train1/affstrain1_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train2/affstrain2_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train3/affstrain3_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train4/affstrain4_vol1.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/train5/affstrain5_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train6/affstrain6_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train7/affstrain7_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train8/affstrain8_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train9/affstrain9_vol1.h5" ]
"""

class segment_methodology():
    def __init__(self, 
                 affinity_paths: list,
                 ground_truth_paths: list):
    
        super().__init__()
        self.affinity_paths = affinity_paths
        self.ground_truth_paths = ground_truth_paths

    @classmethod
    def agglomerate(self, affinity_paths, ground_truth_paths, **kwargs):
        segmentations = []
        
        threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
        for aff, gt in zip(affinity_paths, ground_truth_paths): 
            groundtruth = load_chunk_or_volume(gt, **kwargs) 
            affinities = load_chunk_or_volume(aff, **kwargs) 
            segmentation = wz.agglomerate(affinities, threshold, groundtruth, fragments=None, aff_threshold_low=0.0001, aff_threshold_high=0.9999, return_merge_history=True, return_region_graph=False)
            segmentations.append(segmentation) 

        return segmentations
    
    @classmethod
    def evaluate(self, affinity_paths, ground_truth_paths, **kwargs):
        segmentations = []
        
        threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
        for aff, gt in zip(affinity_paths, ground_truth_paths): 
            groundtruth = load_chunk_or_volume(gt, **kwargs) 
            affinities = load_chunk_or_volume(aff, **kwargs) 
            segmentation = wz.evaluate(groundtruth, affinities)
            segmentations.append(segmentation) 

        return segmentations

if __name__ == '__main__':

    affinity_paths = ["/mnt/home/mpaez/ceph/affsmaptrain/experim/affstrain1_vol1.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/train1/affstrain1_vol1.h5"]

    ground_truth_paths = ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07338/affs_160k.h5", 
                      "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07338/affs_160k.h5"]

    segmentation = segment_methodology.agglomerate(affinity_paths, ground_truth_paths) 
    
    for seg in segmentation:
        seg 



