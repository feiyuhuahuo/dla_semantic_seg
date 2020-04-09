#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import glob


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=1, bias=False, padding=(kernel_size - 1) // 2)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0, root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)

            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels, root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        if in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                   stride=1, bias=False),
                                         nn.BatchNorm2d(out_channels))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)  # seems useless here to transfer 'residual' ?

        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)

        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                                        nn.BatchNorm2d(channels[0]),
                                        nn.ReLU(inplace=True))

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)

        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)

        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)

        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)

        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def _make_conv_level(inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(inplanes, planes, kernel_size=3,
                                      stride=stride if i == 0 else 1,
                                      padding=dilation, bias=False, dilation=dilation),
                            nn.BatchNorm2d(planes),
                            nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, f'level{i}')(x)
            y.append(x)

        return y

    def load_pretrained_model(self, name):
        weights = glob.glob(f'weights/{name}-*')[0]
        state_dict = torch.load(weights)
        # state_dict.pop('level2.root.conv.weight')
        # state_dict.pop('level2.project.0.weight')
        # state_dict.pop('level3.tree1.root.conv.weight')
        # state_dict.pop('level3.tree1.project.0.weight')
        # state_dict.pop('level3.tree2.root.conv.weight')
        # state_dict.pop('level3.project.0.weight')
        # state_dict.pop('level4.tree1.root.conv.weight')
        # state_dict.pop('level4.tree1.project.0.weight')
        # state_dict.pop('level4.tree2.root.conv.weight')
        # state_dict.pop('level4.project.0.weight')
        # state_dict.pop('level5.root.conv.weight')
        # state_dict.pop('level5.project.0.weight')
        self.load_state_dict(state_dict, strict=False)
        print(f'{weights} loaded.\n')


def dla34(**kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    model.load_pretrained_model('dla34')
    return model


def dla46_c(**kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256], block=Bottleneck, **kwargs)
    model.load_pretrained_model('dla46_c')
    return model


def dla46x_c(**kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256], block=BottleneckX, **kwargs)
    model.load_pretrained_model('dla46x_c')
    return model


def dla60x_c(**kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 64, 64, 128, 256], block=BottleneckX, **kwargs)
    model.load_pretrained_model('dla60x_c')
    return model


def dla60(**kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], block=Bottleneck, **kwargs)
    model.load_pretrained_model('dla60')
    return model


def dla60x(**kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], block=BottleneckX, **kwargs)
    model.load_pretrained_model('dla60x')
    return model


def dla102(**kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024], block=Bottleneck, residual_root=True, **kwargs)
    model.load_pretrained_model('dla102')
    return model


def dla102x(**kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024], block=BottleneckX, residual_root=True, **kwargs)
    model.load_pretrained_model('dla102x')
    return model


def dla102x2(**kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024], block=BottleneckX, residual_root=True, **kwargs)
    model.load_pretrained_model('dla102x2')
    return model


def dla169(**kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024], block=Bottleneck, residual_root=True, **kwargs)
    model.load_pretrained_model('dla169')
    return model
