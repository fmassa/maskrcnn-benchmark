# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

import torchvision

from maskrcnn_benchmark.layers import FrozenBatchNorm2d


class ResNetModule(torchvision.models.ResNet):
    def __init__(self, block, layers, norm_layer=None, return_layers=None):
        super(ResNetModule, self).__init__(
                block,
                layers,
                norm_layer=norm_layer
        )
        del self.avgpool
        del self.fc

        self.return_layers = return_layers

    def forward(self, x):
        out = []
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out.append(x)
        return out

def ResNet(cfg):
    block = torchvision.models.resnet.Bottleneck
    layers = [3, 4, 6, 3]
    return_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    model = ResNetModule(block, layers, FrozenBatchNorm2d, return_layers)
    for name, parameter in model.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)
    return model

