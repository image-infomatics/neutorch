import numpy as np
import os
import torch
import click

from neutorch.dataset.affinity import TestDataset
from neutorch.cremi.evaluate import do_agglomeration, cremi_metrics
from neutorch.model.config import *
from neutorch.model.io import load_chkpt
from neutorch.dataset.affinity import TestDataset
from neutorch.cremi.evaluate import do_agglomeration, cremi_metrics


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
@click.option('--parallel',  type=str, default='ddp', help='used to wrap model in necessary parallism module.'
              )
@click.option('--agglomerate',
              type=bool, default=True, help='whether to run agglomerations as well'
              )
@click.option('--with-label',
              type=bool, default=True, help='whether to read label as well and produce CREMI metrics'
              )
@click.option('--save-aff',
              type=bool, default=False, help='whether to save the affinity file'
              )
@click.option('--save-seg',
              type=bool, default=False, help='whether to save the segmentation file'
              )
@click.option('--threshold',
              type=int, default=0.7, help='threshold to use for agglomeration step.'
              )
def test(path: str, config: str, patch_size: str, load: str, parallel: str,
         agglomerate: bool, with_label: bool, save_aff: bool, save_seg: bool, threshold: int):

    # convert
    patch_size = eval(patch_size)

    # get config
    config = get_config(config)
    model = build_model_from_config(config.model)

    if parallel == 'dp':
        model = torch.nn.DataParallel(model)
    if parallel == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[0], find_unused_parameters=True)

    output_dir = f'./run_{config.name}/tests'

    # load chkpt
    if load != '':
        # if load is a number infer path
        if load.isnumeric():
            load = f'./run_{config.name}/chkpts/model_{load}.chkpt'
        model = load_chkpt(model, load)

    # gpu settings
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    test_model(model, patch_size, output_dir, agglomerate=agglomerate,
               with_label=with_label, save_aff=save_aff, save_seg=save_seg, path=path, threshold=threshold)


def test_model(model, patch_size, output_dir: str,
               agglomerate: bool = True, threshold: float = 0.7, with_label: bool = True, save_aff: bool = False, save_seg: bool = False, path: str = './data/sample_A_pad.hdf',):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = TestDataset(path, patch_size, with_label=with_label)

    # over allocate then we will crop
    range = dataset.get_range()
    affinity = np.zeros((3, *range))
    (pz, py, px) = patch_size

    print('evaluating test set...')
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
            affinity[:, iz:iz+pz, iy:iy+py, ix:ix+px] = pred_affs.cpu()

    if save_aff:
        np.save(f'{output_dir}/affinity.npy', affinity)

    if agglomerate:
        # get predicted segmentation from affinity map
        segmentation_pred = do_agglomeration(
            affinity, threshold=threshold)
        if save_seg:
            np.save(f'{output_dir}/segmentation.npy', segmentation_pred)

        if with_label:
            label = dataset.label
            (sz, sy, sx) = label.shape
            (oz, oy, ox) = dataset.label_offset

            seg_section = segmentation_pred[oz:oz+sz, oy:oy+sy, ox:ox+sx]
            # get the CREMI metrics from true segmentation vs predicted segmentation
            metrics = cremi_metrics(seg_section, label)

            # log metrics
            for k, v in metrics.items():
                print(f'{k}:{round(v,5)}')

            f = open(f"{output_dir}/metrics.txt", "w")
            f.write(f'threshold: {threshold} data: {path}\n')
            f.write(f'===================================\n')
            for k, v in metrics.items():
                f.write(f'{k}:{round(v,5)}\n')
            f.close()


if __name__ == '__main__':
    test()
