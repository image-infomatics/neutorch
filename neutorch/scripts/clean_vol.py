import numpy as np
from neutorch.dataset.utils import from_h5
from neutorch.cremi.evaluate import write_output_data, do_agglomeration


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


print('reading aff...')
affinityAp = from_h5(
    '/mnt/home/jberman/ceph/RSUnetBIG_200000/aff_sample_A+_pad.h5', dataset_path='affinity')

print('fixing aff...')
fixedAp = fix_A(affinityAp)

(sz, sy, sx) = (125, 1250, 1250)
(oz, oy, ox) = (37, 911, 911)
fixedAp = fixedAp[:, oz:oz+sz, oy:oy+sy, ox:ox+sx]
print(f'cropped aff to {fixedAp.shape}')

print('doing agglomeration...')
segmentation = do_agglomeration(fixedAp, threshold=0.65)

write_output_data(fixedAp, segmentation, None, config_name='RSUnetBIG_manfix', example_number=200000, file='sample_A+_pad',
                  output_dir=f'/mnt/home/jberman/ceph')
print('done!')
