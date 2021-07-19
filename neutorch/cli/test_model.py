import numpy as np
import torch
import click
import os
import cc3d
from tqdm import tqdm

from neutorch.dataset.affinity import TestDataset
from neutorch.model.config import *
from neutorch.model.io import load_chkpt
from neutorch.dataset.affinity import TestDataset
from neutorch.cremi.evaluate import do_agglomeration, cremi_metrics, write_output_data


@click.command()
@click.option('--config',
              type=str,
              help='name of the configuration defined in the config list.'
              )
@click.option('--path',
              type=str, help='path to the test data file'
              )
@click.option('--patch-size', '-p',
              type=str, default='(26, 256, 256)',
              help='patch size from volume.'
              )
@click.option('--load', type=str, default='', help='load from checkpoint, path to ckpt file or chkpt number.'
              )
@click.option('--parallel',  type=str, default='dp', help='used to wrap model in necessary parallism module.'
              )
@click.option('--agglomerate',
              type=bool, default=True, help='whether to run agglomerations as well'
              )
@click.option('--test-vol',
              type=bool, default=False, help='whether this is test volume.'
              )
@click.option('--save-aff',
              type=bool, default=True, help='whether to save the affinity file.'
              )
@click.option('--save-seg',
              type=bool, default=True, help='whether to save the segmentation file'
              )
@click.option('--full-agglomerate',
              type=bool, default=False, help='whether to agglomerate over entire volume. takes longer but maybe more accurate on edges.'
              )
@click.option('--threshold',
              type=float, default=0.7, help='threshold to use for agglomeration step.'
              )
def test(path: str, config: str, patch_size: str, load: str, parallel: str,
         agglomerate: bool, test_vol: bool, save_aff: bool, save_seg: bool, full_agglomerate: bool, threshold: float):

    # convert
    patch_size = eval(patch_size)

    # get config
    config = get_config(config)
    model = build_model_from_config(config.model)

    if parallel == 'dp':
        model = torch.nn.DataParallel(model)

    output_dir = f'./{config.name}_run'

    example_number = 'unknown'
    # load chkpt
    if load != '':
        # if load is a number infer path
        if load.isnumeric():
            example_number = load
            load = f'./{config.name}_run/chkpts/model_{load}.chkpt'
        model = load_chkpt(model, load)

    # gpu settings
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    res = test_model(model, patch_size, agglomerate=agglomerate,
                     full_agglomerate=full_agglomerate, test_vol=test_vol, path=path, threshold=threshold, border_width=config.dataset.border_width)

    # save data
    affinity, segmentation, metrics = res['affinity'], res['segmentation'], res['metrics']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = path.replace('.', '').replace(
        '/', '').replace('hdf', '').replace('data', '')

    print(metrics)
    write_output_data(affinity, segmentation, metrics, config_name=config.name, example_number=example_number, file=file,
                      output_dir=f'/mnt/home/jberman/ceph')


def test_model(model, patch_size, stride=(10, 100, 100),
               agglomerate: bool = True, threshold: float = 0.7, border_width: int = 1,
               full_agglomerate=False, test_vol=False, path: str = './data/sample_C_pad.hdf'):

    res = {}  # results

    # set up
    begin = []
    for i in range(len(patch_size)):
        c = patch_size[i] // 2
        o = stride[i] // 2
        begin.append(c-o)
    begin = tuple(begin)

    dataset = TestDataset(path, patch_size, stride, with_label=not test_vol)

    # over allocate then we will crop
    affinity = np.zeros((3, *dataset.full_shape))

    print('building affinity...')
    (sz, sy, sx) = stride
    (bz, by, bx) = begin
    (pz, py, px) = patch_size

    for (index, image) in enumerate(dataset):
        (iz, iy, ix) = dataset.get_indices(index)
     
        # add dimension for batch
        image = torch.unsqueeze(image, 0)

        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            image = image.cuda()

        with torch.no_grad():
            # compute loss
            logits = model(image)
            predict = torch.sigmoid(logits)
            predict = torch.squeeze(predict)
            pred_affs = predict[0:3, ...]
            pred_affs = pred_affs.cpu()

            # write on edges, only need for full agg as otherwise we crop in
            if full_agglomerate and (iz == 0 or iy == 0 or ix == 0):
                affinity[:, iz:iz+pz, iy:iy+py,
                         ix:ix+px] = pred_affs[:, :, :, :]

            # write in center of patches
            affinity[:, iz+bz:iz+sz+bz, iy+by:iy+sy+by, ix+bx:ix +
                     sx+bx] = pred_affs[:, bz:bz+sz, by:by+sy, bx:bx+sx]

    # crop to true shape
    (tz, ty, tx) = dataset.true_shape
    affinity = affinity[:, :tz, :ty, :tx]

    if not test_vol:
        label = dataset.label
        (sz, sy, sx) = label.shape
        (oz, oy, ox) = dataset.label_offset
    else:
        (sz, sy, sx) = (125, 1250, 1250)
        (oz, oy, ox) = (37, 911, 911)

    # crop before agglomerate
    if not full_agglomerate:
        affinity = affinity[:, oz:oz+sz, oy:oy+sy, ox:ox+sx]

    res['affinity'] = affinity

    if agglomerate:

        print('doing agglomeration...')
        # get predicted segmentation from affinity map
        segmentation = do_agglomeration(
            affinity, threshold=threshold)

        # crop after agglomerate
        if full_agglomerate:
            segmentation = segmentation[oz:oz+sz, oy:oy+sy, ox:ox+sx]
            # connected component analysic after full agglomerate
            segmentation = cc3d.connected_components(segmentation)

        res['segmentation'] = segmentation

        print('computing metrics...')
        # get the CREMI metrics from true segmentation vs predicted segmentation
        metrics = cremi_metrics(segmentation, label, border_width=border_width)

        res['metrics'] = metrics

    return res


if __name__ == '__main__':
    test()
