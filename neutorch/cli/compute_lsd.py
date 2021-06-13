import click
import numpy as np

import numpy as np
from neutorch.dataset.utils import from_h5
from neutorch.dataset.local_shape_descriptor import get_local_shape_descriptors


@click.command()
@click.option('--input', '-i',
              type=str,
              required=True,
              help='label segmentation file'
              )
@click.option('--output', '-o',
              type=str,
              default='',
              help='the directory to save the output'
              )
@click.option('--sigma',
              type=tuple, default=(3, 30, 30),
              help='sigma tuple for radius of lsd computations (z,y,x)'
              )
@click.option('--downsample',
              type=int, default=2,
              help='downsample factor'
              )
@click.option('--mode',
              type=str, default='gaussian',
              help='number for threads to used to compute lsd'
              )
@click.option('--num_threads',
              type=int, default=1,
              help='number for threads to used to compute lsd'
              )
@click.option('--test',
              type=bool, default=False,
              help='used to test, will only perform on small patch of input label'
              )
def compute_lsd(input: str, output: str, sigma: tuple, downsample: int,  mode: str, num_threads: int, test: bool):

    if output == '':
        output = f'{input}_lsd'

    print(f'Computing LSD for {input}...')
    label = from_h5(
        f'{input}.hdf', dataset_path='volumes/labels/neuron_ids')

    # Segmentation shape (125, 1250, 1250) must be a multiple of downsampling factor 2
    # so we append with a copy of last axial slice
    label = np.append(label, [label[-1]], axis=0)

    if test:
        label = label[0:4, 0:128, 0:128]

    print(
        f'PARAMS: label shape: {label.shape}, sigma: {sigma}, downsample: {downsample}, mode: {mode}, num_threads: {num_threads}')

    lsd = get_local_shape_descriptors(
        label,
        sigma,
        voxel_size=None,
        roi=None,
        labels=None,
        mode=mode,
        downsample=downsample,
        num_threads=num_threads)

    print('lsd shape: ', lsd.shape)
    np.save(f'{output}.npy', lsd)
    print('Done!')


if __name__ == '__main__':
    compute_lsd()
