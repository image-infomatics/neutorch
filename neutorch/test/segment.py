import numpy as np
from waterz import agglomerate
import matplotlib.pyplot as plt

from chunkflow.volume import load_chunk_or_volume

class segment_methodology():
    def __init__(self, 
                 affinity_paths: list,
                 thresholds: list,
                 ground_truth_paths: list):
    
        super().__init__()
        self.affinity_paths = affinity_paths
        self.thresholds = thresholds
        self.ground_truth_paths = ground_truth_paths

    @classmethod
    def use_agglomerate(self, affinity_paths, thresholds, ground_truth_paths, **kwargs):
        segmentations = []
        
        for aff_path, gt_path in zip(affs_paths, gt_paths): 
            affs = load_chunk_or_volume(aff_path, **kwargs) 
            gt = load_chunk_or_volume(gt_path, **kwargs) 
 
            affs_array = affs.array.astype(np.float32)
            gt_array = gt.array.astype(np.uint32)
            assert affs.shape[-3:] == gt.shape[-3:]

            breakpoint()
            segmentation = agglomerate(affs=affs_array, thresholds=thresholds, gt=gt_array, fragments=None, aff_threshold_low=0.0001, aff_threshold_high=0.9999, return_merge_history=True, return_region_graph=False)
            segmentations.append(segmentation) 

        return segmentations

if __name__ == '__main__':

    affs_paths = ["/mnt/home/mpaez/ceph/affsmaptrain/sample3/train2_chkpt28000_affs_vol01700.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample3/train2_chkpt28000_affs_vol04900.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_chkpt28000_affs_vol07338.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_chkpt28000_affs_vol07580.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_chkpt28000_affs_vol07800.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_chkpt28000_affs_vol09300.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_chkpt28000_affs_vol12350.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample2/train2_chkpt28000_affs_vol13170_2.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample1/train2_chkpt28000_affs_s1gt1.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/sample1/train2_chkpt28000_affs_s1gt2.h5",
                  ]
    
    gt_paths = ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_01700/seg_v2.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_04900/mito_seg_zyx_4900-5156_3000-3256_6240-6496.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07338/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07580/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07800/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_09300/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_12350/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_13170_2/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt1/seg_v1.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt2/seg_v2.h5",
                ]
    
    """ Note: needs to be old affinity maps """
    label_paths = ["/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_01700/label_v3.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/13_wasp_sample3/vol_04900/label_v4.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07338/label_v9.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07580/label_v2.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_07800/label_v2.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_09300/label_v2.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_12350/label_v2.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/12_wasp_sample2/vol_13170_2/label_v2.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt1/label_v2.h5",
                "/mnt/ceph/users/neuro/wasp_em/jwu/40_gt/11_wasp_sample1/s1gt2/label_v2.h5",
    ]
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    assert len(affs_paths) == len(gt_paths) 

    segmentation = segment_methodology.use_agglomerate(affinity_paths=affs_paths, thresholds=thresholds, ground_truth_paths=gt_paths) 
    
    results = []
    for seg in segmentation:
        datas = [x for x in seg] 
        
        result = []
        for data in datas: 
            result.append(data[1])
        results.append(result)

    print(results)
    

