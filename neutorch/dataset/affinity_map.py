
from chunkflow.lib.bounding_boxes import Cartesian

from .sample import GroundTruthSample
from .transform import *
from .dataset_base import DatasetBase


class DataSet(DatasetBase):
    def __init__(self, config_file: str, patch_size: Cartesian):
        super().__init__(config_file, patch_size)

    
