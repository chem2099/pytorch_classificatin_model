from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ['resnext']


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality, widen_factor):
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
        # in_channels  64
        # cardinality  16 out_channels=256  widen_factor =4
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
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

        def forward(self, x):
            bottleneck = self.conv_reduce.forward(x)
            bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
            bottleneck = self.conv_conv.forward(bottleneck)
            bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
            bottleneck = self.conv_expand.forward(bottleneck)
            bottleneck = self.bn_expand.forward(bottleneck)
            residual = self.shortcut.forward(x)
            return F.relu(residual + bottleneck, inplace=True)


# 具体的分类用到的resnext
class CifarResNeXt(nn.Module):
    """
     ResNext optimized for the Cifar dataset, as specified in
     https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, num_classes, widen_factor=4, dropRate=0):
        """
         Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            num_classes: number of classes
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stage[0], self.stage[1], 1)
        self.stage_2 = self.block('stage_2', self.stage[1], self.stage[2], 2)
        self.stage_3 = self.block('stage_3', self.stage[2], self.stage[3], 3)
        self.classifier = nn.Linear(1024, num_classes)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """
       Stack n bottleneck modules where n is inferred from the depth of the network.
       Args:
       name: string name of the current block.
       in_channels: number of input channels
       out_channels: number of output channels
       pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
       Returns: a Module consisting of n sequential bottlenecks.
       """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                bottleneck.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                               self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, self.cardinality, self.widen_factor))

        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, 1024)
        return self.classifier(x)


def resnext(**kwargs):
    """Constructs a ResNeXt.
    """
    model = CifarResNeXt(**kwargs)
    return model
