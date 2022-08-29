# neutorch
Neuron segmentation and synapse detection using PyTorch

> :warning: **This package is still under development, it was customized to process a Electron Microscopy volume although it could be potentially generalizable to other biological image volumes. Use at your own risk.**

# Functions
- [x] Synapse detection
- [ ] Neuron boundary detection
- [x] Image denoising by image inpainting

# Features
- [x] Training using whole terabyte or even petabyte of image volume.
- [x] Training using multiple version of image datasets as data augmentation.
- [x] Data augmentation without zero-filling.

# Install
    python setup.py install

# Usage
## Train
We provide command line tool after installation. Check out the options provided:

    neutrain --help

# Acknowledgements

This package is built upon the following packages:
- [DeepEM](https://github.com/seung-lab/DeepEM)
- [DataProvider3](https://github.com/seung-lab/DataProvider3)
- [PyTorchUtils](https://github.com/nicholasturner1/PyTorchUtils)
- [pytorch_connectomics](https://github.com/zudi-lin/pytorch_connectomics)
- [torchio](https://github.com/fepegar/torchio)
- [detectron2](https://github.com/facebookresearch/detectron2)

The development is supported by Flatiron Institute, a division of the Simons Foundation.