from waterz import agglomerate
import numpy as np
from neutorch.cremi.neuron_ids import NeuronIds
import math
import os
from neutorch.cremi.volume import Volume
from neutorch.cremi.io import CremiFile


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


def cremi_metrics(test, truth, border_width=1):
    neuron_ids_evaluation = NeuronIds(truth, border_threshold=border_width)
    (voi_split, voi_merge) = neuron_ids_evaluation.voi(test)
    adapted_rand = neuron_ids_evaluation.adapted_rand(test)
    # the geometric mean of (VOI split + VOI merge) and ARAND.
    cremi_score = math.sqrt((voi_split + voi_merge) * adapted_rand)

    return {'voi_split': voi_split, 'voi_merge': voi_merge, 'adapted_rand': adapted_rand, 'cremi_score': cremi_score}


def write_output_data(aff, seg, metrics,  config_name='', example_number='', file='',
                      output_dir=f'/mnt/home/jberman/ceph'):

    output_dir = f'{output_dir}/{config_name}_{example_number}'
    # out dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write metrics
    if metrics is not None:
        f = open(f"{output_dir}/metrics.txt", "a")
        f.write(
            f'config: {config_name} example_number: {example_number} file: {file} \n')
        f.write(f'===================================\n')
        for k, v in metrics.items():
            f.write(f'{k}:{round(v,5)}\n')
        f.write(f'-----------------------------------\n')
        f.close()

    # write seg
    if seg is not None:
        cremiFile = CremiFile(f'{output_dir}/seg_{file}.hdf', "w")
        neuron_ids = Volume(seg, resolution=(
            40.0, 4.0, 4.0), comment=f'seg_{file}')
        cremiFile.write_neuron_ids(neuron_ids)

    # write aff
    if aff is not None:
        cremiFile = CremiFile(f'{output_dir}/aff_{file}.hdf', "w")
        neuron_ids = Volume(aff, resolution=(
            40.0, 4.0, 4.0), comment=f'aff_{file}')
        cremiFile.write_neuron_ids(neuron_ids)
