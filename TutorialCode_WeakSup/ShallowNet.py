#!/usr/bin/env python3

import torch.nn as nn


def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        # nn.LeakyReLU(0.2)
        nn.PReLU()
    )


class shallowCNN(nn.Module):
    def __init__(self, nin, nG, nout):
        super(shallowCNN, self).__init__()
        self.conv0 = convBatch(nin, nG * 4)
        self.conv1 = convBatch(nG * 4, nG * 4)
        self.conv2 = convBatch(nG * 4, nout)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        return x2
