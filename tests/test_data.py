# def test_sample():

#%% 
from typing import List
import os
from tqdm import tqdm
from chunkflow.lib.cartesian_coordinate import Cartesian
from chunkflow.volume import load_chunk_or_volume

from neutorch.data.dataset import load_cfg
from neutorch.data.sample import SemanticSample

PATCH_NUM = 100
DEFAULT_PATCH_SIZE=Cartesian(1, 64, 64)
OUT_DIR = os.path.expanduser('/tmp/patches/')
cfg = load_cfg('/home/jwu/projects/63_hcs-image-analysis/42-incucyte-images/sample.yaml')

# sample = SemanticSample.from_config_v6(cfg)

def load_chunks_or_volumes(paths: List[str]):
    inputs = []
    for image_path in paths:
        image_vol = load_chunk_or_volume(image_path)
        inputs.append(image_vol)
    return inputs 

# %%

# for chunk_name, chunk_meta in cfg.items():
#     print(chunk_name)
#     images = []
#     for image_fname in chunk_meta['images']:
#         image_path = os.path.join(chunk_meta['dir'], image_fname)
#         image_vol = load_chunk_or_volume(image_path)
#         images.append(image_vol)

#     label_path = os.path.join(chunk_meta['dir'], chunk_meta['label'])
#     label = load_chunk_or_volume(label_path)


# sample = SemanticSample(images, label, output_patch_size=DEFAULT_PATCH_SIZE) 

#%%
samples = []
for node in cfg.values():
    sample = SemanticSample.from_config_v6(
        node, 
        output_patch_size=DEFAULT_PATCH_SIZE)
    samples.append(sample)

sample = samples[0]

patch = sample.random_patch
print(patch)
# breakpoint()
# %%
