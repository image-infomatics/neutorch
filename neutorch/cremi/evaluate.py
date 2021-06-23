from waterz import agglomerate
import numpy as np
from neutorch.cremi.neuron_ids import NeuronIds
from neutorch.dataset.patch import AFF_BORDER_WIDTH
import math


def do_agglomeration(affs, threshold=0.7, aff_threshold_low=0.001,  aff_threshold_high=0.999, flip=True):

    # flip so affinity channel is z,y,x NOT x,y,z
    if flip:
        affs = np.flip(affs, axis=0)
    affs = np.ascontiguousarray(affs, dtype=np.float32)

    seg_generator = agglomerate(affs, [threshold],
                                aff_threshold_low=aff_threshold_low,
                                aff_threshold_high=aff_threshold_high,
                                return_merge_history=False,
                                return_region_graph=False)

    res = None
    for seg in seg_generator:
        res = seg.copy()

    return res


def cremi_metrics(test, truth):
    neuron_ids_evaluation = NeuronIds(truth, border_threshold=AFF_BORDER_WIDTH)
    (voi_split, voi_merge) = neuron_ids_evaluation.voi(test)
    adapted_rand = neuron_ids_evaluation.adapted_rand(test)
    # the geometric mean of (VOI split + VOI merge) and ARAND.
    cremi_score = math.sqrt((voi_split + voi_merge) * adapted_rand)

    return {'voi_split': voi_split, 'voi_merge': voi_merge, 'adapted_rand': adapted_rand, 'cremi_score': cremi_score}
