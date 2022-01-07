from abc import abstractproperty
from functools import cached_property

import torch
import toml

from chunkflow.lib.bounding_boxes import Cartesian


class DatasetBase(torch._utils.data.Dataset):
    def __init__(self, 
            config_file: str,
            patch_size: Cartesian):
        super().__init__()

        config_file = os.path.expanduser(config_file)
        assert config_file.endswith('.toml'), "we use toml file as configuration format."
        with open(config_file, 'r') as file:
            self.meta = toml.load(file)

        self.patch_size = patch_size

    # @abstractproperty
    # @cached_property
    # def training_samples(self):
    #     pass

    # @abstractproperty
    # @cached_property
    # def validation_samples(self):
    #     pass

    @abstractproperty
    @cached_property
    def transform(self):
        pass

    @abstractproperty
    def random_training_patch(self):
        pass
    
    @abstractproperty
    def random_validation_patch(self):
        pass

    

