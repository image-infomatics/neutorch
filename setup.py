#!/usr/bin/env python
import os
import re

from setuptools import setup, find_packages

from pybind11.setup_helpers import Pybind11Extension, build_ext


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PACKAGE_DIR, 'requirements.txt')) as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSIONFILE = os.path.join(PACKAGE_DIR, "neutorch/__version__.py")
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." %
                       (VERSIONFILE, ))


ext_modules = [
    Pybind11Extension("libneutorch",
        ["cpp/main.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', version)],
        ),
]

setup(
    name='neutorch',
    version=version,
    description='Deep Learning for brain connectomics using PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    url='https://github.com/brain-connectome/neutorch',
    packages=find_packages(exclude=['bin']),
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    install_requires=requirements,
    tests_require=[
        'pytest',
    ],
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
    python_requires='>=3',
    zip_safe=False,
)
