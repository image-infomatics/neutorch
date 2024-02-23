import waterz
import numpy as np

from chunkflow.lib.cartesian_coordinate import Cartesian 
from chunkflow.chunk import Chunk
from chunkflow.volume import load_chunk_or_volume
from chunkflow.volume import PrecomputedVolume, AbstractVolume

affinity_paths = ["/mnt/home/mpaez/ceph/affsmaptrain/train1/affstrain1_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train2/affstrain2_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train3/affstrain3_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train4/affstrain4_vol1.h5",
                  "/mnt/home/mpaez/ceph/affsmaptrain/train5/affstrain5_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train6/affstrain6_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train7/affstrain7_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train8/affstrain8_vol1.h5", 
                  "/mnt/home/mpaez/ceph/affsmaptrain/train9/affstrain9_vol1.h5" ]

class segment_methodology():
    def __init__(self, affinity_paths: list):

    @classmethod
    def affinity_methodology():
        segmentations = []
        
        for affinity_path in affinity_paths: 
            affinities = load_chunk_or_volume(affinity_path, **kwargs) 
            """ affinity path -> h5 file is this correct? """
            segmentation = waterz.agglormerate(affinities, threshold)
            segmentations.append(segmentation) 

if __name__ == '__main__':

