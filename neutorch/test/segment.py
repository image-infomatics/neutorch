import waterz as wz
import numpy as np
import matplotlib.pyplot as plt

from chunkflow.lib.cartesian_coordinate import Cartesian 
from chunkflow.chunk import Chunk
from chunkflow.volume import load_chunk_or_volume
from chunkflow.volume import PrecomputedVolume, AbstractVolume
from yacs.config import CfgNode

class segment_methodology():
    def __init__(self, 
                 affinity_paths: list,
                 ground_truth_paths: list,
                 thresholds: list):
    
        super().__init__()
        self.affinity_paths = affinity_paths
        self.ground_truth_paths = ground_truth_paths
        self.thresholds = thresholds

    @classmethod
    def agglomerate(self, affs_paths, gt_paths, thresholds, **kwargs):
        segmentations = []
        
        for aff_path, gt_path in zip(affs_paths, gt_paths): 
            affs = load_chunk_or_volume(aff_path, **kwargs) 
            gt = load_chunk_or_volume(gt_path, **kwargs) 
 
            affs_array = affs.array.astype(np.float32)
            gt_array = gt.array.astype(np.uint32)
            assert affs.shape[-3:] == gt.shape[-3:]

            segmentation = wz.agglomerate(affs_array, thresholds, gt=gt_array, 
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
            gt = load_chunk_or_volume(gt_path, **kwargs) 

            affs_array = affs.array.astype(np.float32)
            gt_array = gt.array.astype(np.uint32)
            assert affs.shape[-3:] == gt.shape[-3:]

            segmentation = wz.agglomerate(affs_array, gt_array)
            segmentations.append(segmentation) 

        return segmentations

if __name__ == '__main__':

    import os
    from tqdm import tqdm
    from PIL import Image
    import numpy as np
    import pandas as pd
    from neutorch.data.dataset import load_cfg
    
    affs_paths = ["/mnt/home/mpaez/ceph/affsmaptrain/sample3/train2_28000_affs01700.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample3/train2_28000_affs04900.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_28000_affs07338.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_28000_affs07580.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_28000_affs07800.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_28000_affs09300.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_28000_affs12350.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_28000_affs13170_2.h5",
                  # "/mnt/home/mpaez/ceph/affsmaptrain/58_bmembrane/train2_28000_affs31.h5",
                  # "/mnt/home/mpaez/ceph/affsmaptrain/58_bmembrane/train2_28000_affs32.h5",
                  # "/mnt/home/mpaez/ceph/affsmaptrain/58_bmembrane/train2_28000_affs33.h5",
                  # "/mnt/home/mpaez/ceph/affsmaptrain/58_bmembrane/train2_28000_affs41.h5"
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample1/train2_28000_affs1.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample1/train2_28000_affs2.h5",
                  ]
    
    gt_paths = ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_01700/seg_v2.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_04900/mito_seg_zyx_4900-5156_3000-3256_6240-6496.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07338/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07580/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07800/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_09300/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_12350/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_13170_2/seg_v1.h5",
                # "/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/31_test_3072-3584_5120-5632_8196-8708/seg_zyx_3072-3584_5120-5632_8196-8708.h5",
                # "/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/32_test_5120-5632_5632-6144_10240-10752/seg_zyx_5120-5632_5632-6144_10240-10752",
                # "/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/33_test_2560-3072_5632-6144_8704-9216/seg_zyx_2560-3072_5632-6144_8704-9216.h5",
                # "/mnt/ceph/users/neuro/wasp_em/jwu/58_broken_membrane/41_test_2560-3584_5120-6144_8192-9216/seg_zyx_2560-3584_5120-6144_8192-9216.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt1/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt2/seg_v2.h5",
                ]
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    assert len(affs_paths) == len(gt_paths)
    segmentation = segment_methodology.agglomerate(affs_paths, gt_paths, thresholds) 
    
    results = []
    for seg in segmentation:
        datas = [x for x in seg] 
        
        result = []
        for data in datas: 
            result.append(data[1])
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv('/mnt/home/mpaez/ceph/affsmaptrain/evaluate/model_data_ver1.csv')
    

