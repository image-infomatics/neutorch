import numpy as np
import torch
import click
import math
import cc3d
from tqdm import tqdm
import cv2

from neutorch.dataset.utils import from_h5, resize_along_z

from neutorch.model.config import *
from neutorch.model.io import load_chkpt
from neutorch.cremi.evaluate import do_agglomeration, cremi_metrics, write_output_data

from chunkflow.chunk import Chunk
from chunkflow.chunk.image.convnet.inferencer import Inferencer
from chunkflow.chunk.image.convnet.patch.base import PatchInferencerBase


@click.command()
@click.option('--config',
              type=str,
              help='name of the configuration   defined in the config list.'
              )
@click.option('--path',
              type=str, help='path to the test data file'
              )
@click.option('--pre-crop',
              type=str, default='None',
              help='the amount the crop the volume before inference. Each value is applied even on either end.'
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
def test(path: str, config: str, pre_crop: str, load: str, parallel: str,
         agglomerate: bool, test_vol: bool, save_aff: bool, save_seg: bool, full_agglomerate: bool, threshold: float):

    # convert
    pre_crop = eval(pre_crop)

    # get config
    config = get_config(config)
    model = build_model_from_config(config.model)
    patch_size = config.dataset.patch_size

    if parallel == 'dp':
        model = torch.nn.DataParallel(model)

    example_number = 'unknown'
    # load chkpt
    if load != '':
        # if load is a number infer path
        if load.isnumeric():
            example_number = load
            load = f'/mnt/home/jberman/ceph/runs/{config.name}_run/chkpts/model_{load}.chkpt'
        model = load_chkpt(model, load)

    # # gpu settings
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    res = test_model(model, patch_size, path, pre_crop=pre_crop, agglomerate=agglomerate,
                     full_agglomerate=full_agglomerate, test_vol=test_vol, threshold=threshold, border_width=config.dataset.border_width,
                     downsample=config.dataset.downsample)

    # save data
    affinity, segmentation, metrics = res['affinity'], res['segmentation'], res['metrics']

    file = path.replace('.', '').replace(
        '/', '').replace('hdf', '').replace('data', '')

    if metrics is not None:
        print(metrics)

    if not save_aff:
        affinity = None
    if not save_seg:
        segmentation = None

    write_output_data(affinity, segmentation, metrics, config_name=f'{config.name}_manfix', example_number=example_number, file=file,
                      output_dir=f'/mnt/home/jberman/ceph')


def fix_A(vol, offset=37):
    # 0 33 51 79 80 108 109 111
    o = offset
    vol[..., o+0, :, :] = vol[..., o+1, :, :]
    vol[..., o+33, :, :] = vol[..., o+34, :, :]
    vol[..., o+51, :, :] = vol[..., o+52, :, :]
    vol[..., o+79, :, :] = vol[..., o+78, :, :]
    vol[..., o+80, :, :] = vol[..., o+81, :, :]
    vol[..., o+108, :, :] = vol[..., o+107, :, :]
    vol[..., o+109, :, :] = vol[..., o+110, :, :]
    vol[..., o+111, :, :] = vol[..., o+112, :, :]
    return vol


def test_model(model, patch_size, path, pre_crop=None,
               agglomerate: bool = True, threshold: float = 0.7, border_width: int = 1,
               full_agglomerate=False, test_vol=False, downsample: float = 1.0):

    res = {}  # results

    print('loading data...')
    volume = from_h5(path, dataset_path='volumes/raw')
    volume = volume.astype(np.float32) / 255.
    volume = fix_A(volume)
    if not test_vol:
        label, label_offset = from_h5(
            path, dataset_path='volumes/labels/neuron_ids', get_offset=True)
        (sz, sy, sx) = label.shape
        (oz, oy, ox) = label_offset
    else:
        (sz, sy, sx) = (125, 1250, 1250)
        (oz, oy, ox) = (37, 911, 911)

    if pre_crop is not None:
        (cpz, cpy, cpx) = pre_crop
        volume = volume[cpz:-cpz, cpy:-cpy, cpx:-cpx]
        (oz, oy, ox) = (oz-cpz, oy-cpy, ox-cpx)
        (vz, vy, vx) = volume.shape
        assert vz > oz+sz and vy > oy+sy and vx > ox+sx

    if downsample != 1.0:
        pre_downsample_shape = volume.shape
        (_, prey, prex) = pre_downsample_shape
        newy, newx = math.ceil(
            prey*downsample), math.ceil(prex*downsample)
        volume = resize_along_z(
            volume, newy, newx, interpolation=cv2.INTER_NEAREST)

    volume_chunk = Chunk(volume)

    print('building affinity...')
    # params
    output_patch_overlap = (12, 124, 124)
    num_output_channels = 3

    # set up chunkflow objects
    pi = MyPatchInferencer(model, patch_size, patch_size, output_patch_overlap,
                           num_output_channels=num_output_channels)
    inferencer = Inferencer(pi, None, patch_size,
                            output_patch_size=patch_size, num_output_channels=num_output_channels,
                            output_patch_overlap=output_patch_overlap, output_crop_margin=None,
                            framework='prebuilt', bump='wu',
                            input_size=volume.shape, mask_output_chunk=True
                            )

    affinity = inferencer(volume_chunk)
    affinity = affinity.array

    if downsample != 1.0:
        affinity = resize_along_z(
            affinity, prey, prex, interpolation=cv2.INTER_NEAREST)

    res['affinity'] = affinity

    if not full_agglomerate:
        affinity = affinity[:, oz:oz+sz, oy:oy+sy, ox:ox+sx]

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

        if not test_vol:
            print('computing metrics...')
            # get the CREMI metrics from true segmentation vs predicted segmentation
            metrics = cremi_metrics(
                segmentation, label, border_width=border_width)
        else:
            metrics = None

        res['metrics'] = metrics

    return res

# used for chunkfow_api


class MyPatchInferencer(PatchInferencerBase):

    def __init__(self, model,
                 input_patch_size: tuple,
                 output_patch_size: tuple,
                 output_patch_overlap: tuple,
                 num_output_channels: int = 3,
                 dtype: str = 'float32'):

        super().__init__(input_patch_size, output_patch_size,
                         output_patch_overlap, num_output_channels,
                         dtype=dtype)

        self.num_output_channels = num_output_channels
        self.model = model
        if torch.cuda.is_available():
            self.output_patch_mask = torch.from_numpy(
                self.output_patch_mask).cuda()

    @property
    def compute_device(self):
        return torch.cuda.get_device_name()

    def __call__(self, input_patch):
        with torch.no_grad():
            input_patch = torch.from_numpy(input_patch).cuda()
            logits = self.model(input_patch)
            predict = torch.sigmoid(
                logits) * self.output_patch_mask.to(logits.device)
            pred_affs = predict[0:3, ...]
            pred_affs = pred_affs.cpu().numpy()
        return pred_affs


if __name__ == '__main__':
    test()
