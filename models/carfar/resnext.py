from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ['resnext']

class ResNeXtBottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, stride, cardinality, widen_factor):
        """ Constructor
        resNeXt 有三中类型 type a：inception+残差 使用sum加和
        type b：inception concat 后最好使用1x1的卷积核在卷下 残差
        type c:群组卷积  后最好使用1x1的卷积核在卷下
        这边使用了 type c
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        # cardinality 分组数 16 输出 256 widen_factor =4
        D = cardinality * out_channels // widen_factor
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality,
                                   bias=False)

        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))
