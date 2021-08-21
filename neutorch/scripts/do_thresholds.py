from click.core import V
import numpy as np
import torch
import click
import math
import os

from neutorch.dataset.utils import from_h5

from neutorch.model.config import *
from neutorch.model.io import load_chkpt
from neutorch.cremi.evaluate import do_agglomeration, cremi_metrics, write_output_data
from multiprocessing.pool import Pool


print('reading aff...')
affinityAp = from_h5(
    '/mnt/home/jberman/ceph/RSUnetBIG_manfix_200000/aff_sample_A+_pad.h5', dataset_path='affinity')

(sz, sy, sx) = (125, 1250, 1250)
(oz, oy, ox) = (37, 911, 911)
affinityAp = affinityAp[:, oz:oz+sz, oy:oy+sy, ox:ox+sx]
print(f'cropped aff to {affinityAp.shape}')


print('doing agglomerations...')
tholds = np.arange(0.4, 1.0, 0.05)

global do_aggo_metric_for_threshold  # needed for multiprocessing


def do_aggo_metric_for_threshold(threshold):
    print(f'doing agglomeration with threshold {threshold}')

    segmentation = do_agglomeration(affinityAp, threshold=threshold)

    write_output_data(None, segmentation, None, config_name='RSUnetBIG_manfix', example_number=200000, file=f'sample_A+_t={threshold}',
                      output_dir=f'/mnt/home/jberman/ceph')


# do aggo threaded
with Pool(processes=len(tholds)) as pool:
    all_thresholds_to_metrics = pool.map(
        do_aggo_metric_for_threshold, tholds)


print('done!')
