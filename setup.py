#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='neutorch',
    version='0.0.1',
    description='Deep Learning for brain connectomics using PyTorch',
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    url='https://github.com/brain-connectome/neutorch',
    packages=find_packages(exclude=['bin']),
    entry_points='''
        [console_scripts]
        neutrain-pre=neutorch.cli.train_pre_synapses:main
        neutrain-denoise=neutorch.cli.train_denoise:main
        neutrain-post=neutorch.cli.train_post_synapses:main
    ''',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: Apache Software License",
        'Topic :: Communications :: Email',
        'Topic :: Software Development :: Bug Tracking',
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
