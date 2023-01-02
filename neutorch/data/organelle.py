import os
from functools import cached_property

from yacs.config import CfgNode

from chunkflow.chunk.image import Image
from chunkflow.lib.cartesian_coordinate import Cartesian

from neutorch.data.dataset import SemanticDataset, to_tensor
from neutorch.data.sample import OrganelleSample
from neutorch.data.transform import *

if __name__ == '__main__':

    VOLUME_NUM = 10
    PATCH_NUM = 200


    #cfg_file = '/mnt/home/jwu/ceph/31_organelle/28_net/config.yaml'
    cfg_file = './config_mito.yaml'
    
    
    # from glob import glob    
    # glob_path = os.path.expanduser(cfg.dataset.glob_path)
    # path_list = glob(glob_path, recursive=True)
    # # path_list = sorted(path_list)
    # from random import shuffle
    # shuffle(path_list)
    # if VOLUME_NUM > 0:
    #     path_list = path_list[:VOLUME_NUM]
    from neutorch.data.dataset import load_cfg

    cfg = load_cfg(cfg_file) 
    dataset = OrganelleDataset.from_config( cfg, is_train=True )
    
    from PIL import Image
    OUT_DIR = os.path.expanduser('~/dropbox/patches/')
    from tqdm import tqdm

    for idx in tqdm(range(PATCH_NUM)):
        image, label = dataset.random_patch
        
        # section_idx = image.shape[-3]//2
        section_idx = 0
        image = image[0,0, section_idx, :,:]
        label = label[0,0, section_idx, :,:]

        image *= 255.
        im = Image.fromarray(image).convert('L')
        im.save(os.path.join(OUT_DIR, f'{idx}_image.jpg'))

        label *= 255
        lbl = Image.fromarray(label).convert('L')
        lbl.save(os.path.join(OUT_DIR, f'{idx}_label.jpg'))

