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


@click.command()
@click.option('--name',
              type=str, help='name for outputs'
              )
@click.option('--label-path',
              type=str, help='path to the test data file with true label'
              )
@click.option('--aff-path',
              type=str, help='path to the affinity data file'
              )
@click.option('--full-affinity',
              type=bool, help='whether the affinity is of the full padded volume or not'
              )
@click.option('--save-seg',
              type=bool, default=False, help='whether to save the segmentation'
              )
@click.option('--threshold-min',
              type=float, default=0.4, help='threshold to use for agglomeration step.'
              )
@click.option('--threshold-max',
              type=float, default=0.8, help='threshold to use for agglomeration step.'
              )
@click.option('--threshold-step',
              type=float, default=0.01, help='threshold to use for agglomeration step.'
              )
@click.option('--border-width',
              type=int, default=1, help='width applied in cremi metrics eval.'
              )
@click.option('--num-workers',
              type=int, default=-1, help='number of threads. -1 means auto.'
              )
@click.option('--output-dir', '-o',
              type=str, default='./',
              help='for output'
              )
def find_threshold(name: str, label_path: str, aff_path: str, full_affinity: bool, save_seg: bool,
                   threshold_min: float, threshold_max: float, threshold_step: float,
                   border_width: int, num_workers: int, output_dir: str):

    print(f'\nstarting...')

    # build output
    thres_name = name
    f = open(f"{output_dir}/thresholds_{thres_name}.txt", "a")
    f.write(
        f'threshold_min: {threshold_min} threshold_max: {threshold_max} threshold_step: {threshold_step} \n')
    f.write(f'===================================\n')

    # get label
    label, label_offset = from_h5(
        label_path, dataset_path='volumes/labels/neuron_ids', get_offset=True)
    (lz, ly, lx) = label.shape
    (oz, oy, ox) = label_offset

    # get affinity
    affinity = from_h5(aff_path, dataset_path='affinity')

    # if affinity is bigger than label crop
    if full_affinity:
        affinity = affinity[:, oz:oz+lz, oy:oy+ly, ox:ox+lx]

    print(f'affinity shape {affinity.shape}')
    print('doing agglomerations...')
    tholds = np.arange(threshold_min, threshold_max, threshold_step)

    # get workers
    if num_workers == -1:
        cpus = len(os.sched_getaffinity(0))  # gets machine cpus
        num_workers = min(len(tholds), cpus)
        print(f'workers: {num_workers}')

    global do_aggo_metric_for_threshold  # needed for multiprocessing

    # helper thread function
    def do_aggo_metric_for_threshold(threshold):

        print(f'doing agglomeration with threshold {threshold}')

        # get predicted segmentation from affinity map
        segmentation = do_agglomeration(affinity, threshold=threshold)

        print('computing metrics...')
        # get the CREMI metrics from true segmentation vs predicted segmentation
        metrics = cremi_metrics(
            segmentation, label, border_width=border_width)

        print(f'metrics for threshold {threshold}')
        print(metrics)
        metrics['threshold'] = threshold
        if save_seg:
            write_output_data(None, segmentation, None, config_name=thres_name, example_number=threshold, file='',
                              output_dir=f'/mnt/home/jberman/ceph')
        return metrics

    # do aggo threaded
    with Pool(processes=num_workers) as pool:
        all_thresholds_to_metrics = pool.map(
            do_aggo_metric_for_threshold, tholds)

    # write all metrics
    cremis = []
    splits = []
    merges = []
    for metrics in all_thresholds_to_metrics:
        f.write(f'-----------------------------------\n')
        t = metrics['threshold']
        f.write(f'THRESHOLD: {t} \n')
        del metrics['threshold']
        for k, v in metrics.items():
            f.write(f'{k}:{round(v,5)}\n')
        f.write(f'-----------------------------------\n')
        cremis.append(metrics['cremi_score'])
        splits.append(metrics['voi_split'])
        merges.append(metrics['voi_merge'])

    # write best
    best_i = np.argmin(cremis)
    best_t = tholds[best_i]
    f.write(f'================================\n')
    f.write(f'tholds: {tholds}\n')
    f.write(f'splits: {splits}\n')
    f.write(f'merges: {merges}\n')
    f.write(f'cremis: {cremis}\n')
    f.write(f'mean: {round(np.mean(cremis),5)}\n')
    f.write(f'var: {round(np.var(cremis),5)}\n')
    f.write(f'BEST T: {best_t} SCORED {cremis[best_i]}\n')

    f.close()


if __name__ == '__main__':
    find_threshold()
