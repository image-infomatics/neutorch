#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='neutorch',
    version='0.0.2',
    description='Deep Learning for brain connectomics using PyTorch',
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    url='https://github.com/brain-connectome/neutorch',
    packages=find_packages(exclude=['bin']),
    entry_points='''
        [console_scripts]
        neutrain-tbar=neutorch.cli.train_tbar:train
        neutrain-superresolution=neutorch.cli.train_superresolution:train
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
