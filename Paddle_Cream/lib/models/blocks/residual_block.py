'''
本文件为原lib/models/blocks/residual_block.py的转写
'''

import paddle.nn as nn


def conv3x3(in_planes, out_planes, stride = 1):
    "3x3 convolution with padding"
    return nn.Conv2D(
        in_planes, out_planes, 
        kernel_size = 3, stride = stride, padding = 1, 
        bias_attr = None)

class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):

    def __init__(self, 
        inplanes, planes, 
        stride = 1, expansion = 4):
        super(Bottleneck, self).__init__()
        planes = int(planes / expansion)
        self.conv1 = nn.Conv2D(
            inplanes, planes, 
            kernel_size = 1, bias_attr = None)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(
            planes, planes, 
            kernel_size = 3, stride = stride, padding = 1, 
            bias_attr = None)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(
            planes,
            planes * expansion,
            kernel_size = 1,
            bias_attr = None)
        self.bn3 = nn.BatchNorm2D(planes * expansion)
        self.relu = nn.ReLU()
        self.stride = stride
        self.expansion = expansion
        if inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2D(
                    inplanes, planes * self.expansion, 
                    kernel_size = 1, stride = stride, 
                    bias_attr = None),
                nn.BatchNorm2D(planes * self.expansion),)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_Bottleneck(in_c, out_c, stride):
    return Bottleneck(in_c, out_c, stride =stride)


def get_BasicBlock(in_c, out_c, stride):
    return BasicBlock(in_c, out_c, stride =stride)
