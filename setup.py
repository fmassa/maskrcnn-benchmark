# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrcnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    #ext_modules=get_extensions(),
    #cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
