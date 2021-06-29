from time import time
from tqdm import tqdm
import click
import numpy as np

import torch
import torch.nn as nn

from neutorch.model.RSUNet import UNetModel
from neutorch.model.io import load_chkpt
from neutorch.dataset.affinity import TestDataset
from neutorch.cremi.evaluate import do_agglomeration, cremi_metrics


@click.command()
@click.option('--path',
              type=str, help='path to the test data file'
              )
@click.option('--patch-size', '-p',
              type=str, default='(26, 256, 256)',
              help='patch size from volume.'
              )
@click.option('--output-dir', '-o',
              type=click.Path(file_okay=False, dir_okay=True,
                              writable=True, resolve_path=True),
              required=True,
              default='./preds',
              help='the directory to save all the outputs, such as checkpoints.'
              )
@click.option('--in-channels', '-c',
              type=int, default=1, help='channel number of input tensor.'
              )
@click.option('--out-channels', '-n',
              type=int, default=13, help='channel number of output tensor.')
@click.option('--load',
              type=str, default='', help='load from checkpoint, pass path to ckpt file'
              )
@click.option('--agglomerate',
              type=bool, default=False, help='whether to run agglomerations as well'
              )
@click.option('--with-label',
              type=bool, default=False, help='whether to read label as well and produce CREMI metrics'
              )
@click.option('--save-aff',
              type=bool, default=False, help='whether to save the affinity file'
              )
@click.option('--save-seg',
              type=bool, default=False, help='whether to save the segmentation file'
              )
def test(path: str, patch_size: str, output_dir: str, in_channels: int, out_channels: int, load: str,
         agglomerate: bool, with_label: bool, save_aff: bool, save_seg: bool):

    # convert
    patch_size = eval(patch_size)

    # clear in case was stopped before
    tqdm._instances.clear()

    # init model
    model = UNetModel(in_channels, out_channels)
    model = nn.DataParallel(model)

    # load chkpt
    if load != '':
        model = load_chkpt(model, load)

    # gpu settings
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    dataset = TestDataset(path, patch_size, with_label=with_label)
    pbar = tqdm(total=dataset.length)

    # over allocate then we will crop
    range = dataset.get_range()
    affinity = np.zeros((3, *range))
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
            affinity[:, iz:iz+pz, iy:iy+py, ix:ix+px] = pred_affs.cpu()

        pbar.update(1)

    if save_aff:
        np.save(f'{output_dir}/affinity.npy', affinity)

    if agglomerate:
        # get predicted segmentation from affinity map
        segmentation_pred = do_agglomeration(affinity)
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
            for k, v in metrics.items():
                f.write(f'{k}:{round(v,5)}\n')
            f.close()

    pbar.close()


if __name__ == '__main__':
    test()
