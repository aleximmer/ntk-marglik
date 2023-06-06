import torch
import numpy as np
from torch import nn
from torch import nn

from asdfghjkl.operations import Bias, Scale
from asdfghjkl.operations.conv_aug import Conv2dAug


def get_activation(act_str):
    if act_str == 'relu':
        return nn.ReLU
    elif act_str == 'tanh':
        return nn.Tanh
    elif act_str == 'selu':
        return nn.SELU
    elif act_str == 'silu':
        return nn.SiLU
    else:
        raise ValueError('invalid activation')


class MaxPool2dAug(nn.MaxPool2d):

    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])


class AvgPool2dAug(nn.AvgPool2d):

    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])


class AdaptiveAvgPool2dAug(nn.AdaptiveAvgPool2d):

    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])


class MLP(nn.Sequential):
    def __init__(self, input_size, width, depth, output_size, activation='relu',
                 bias=True, fixup=False, augmented=False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        flatten_start_dim = 2 if augmented else 1
        act = get_activation(activation)
        output_size = output_size

        self.add_module('flatten', nn.Flatten(start_dim=flatten_start_dim))

        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, output_size, bias=bias))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size, bias=bias))
                if fixup:
                    self.add_module(f'bias{i+1}b', Bias())
                    self.add_module(f'scale{i+1}b', Scale())
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], output_size, bias=bias))


class LeNet(nn.Sequential):

    def __init__(self, in_channels=1, n_out=10, activation='relu', n_pixels=28,
                 augmented=False):
        super().__init__()
        mid_kernel_size = 3 if n_pixels == 28 else 5
        act = get_activation(activation)
        conv = Conv2dAug if augmented else nn.Conv2d
        pool = MaxPool2dAug if augmented else nn.MaxPool2d
        flatten = nn.Flatten(start_dim=2) if augmented else nn.Flatten(start_dim=1)
        self.add_module('conv1', conv(in_channels, 6, 5, 1))
        self.add_module('act1', act())
        self.add_module('pool1', pool(2))
        self.add_module('conv2', conv(6, 16, mid_kernel_size, 1))
        self.add_module('act2', act())
        self.add_module('pool2', pool(2))
        self.add_module('conv3', conv(16, 120, 5, 1))
        self.add_module('flatten', flatten)
        self.add_module('act3', act())
        self.add_module('lin1', torch.nn.Linear(120*1*1, 84))
        self.add_module('act4', act())
        self.add_module('linout', torch.nn.Linear(84, n_out))


class MiniNet(nn.Sequential):

    def __init__(self, in_channels=1, n_out=10, augmented=False):
        super().__init__()
        conv = Conv2dAug if augmented else nn.Conv2d
        pool = MaxPool2dAug if augmented else nn.MaxPool2d
        flatten = nn.Flatten(start_dim=2) if augmented else nn.Flatten(start_dim=1)
        self.add_module('conv1', conv(in_channels, 8, 5, 1))
        self.add_module('act1', nn.ReLU())
        self.add_module('pool1', pool(2))
        self.add_module('conv2', conv(8, 16, 3, 1))
        self.add_module('act2', nn.ReLU())
        self.add_module('pool2', pool(2))
        self.add_module('conv3', conv(16, 32, 5, 1))
        self.add_module('flatten', flatten)
        self.add_module('act3', nn.ReLU())
        self.add_module('linout', nn.Linear(32, n_out))


def conv3x3(in_planes, out_planes, stride=1, augmented=False):
    """3x3 convolution with padding"""
    Conv2d = Conv2dAug if augmented else nn.Conv2d
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, augmented=False):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.augmented = augmented
        self.bias1a = Bias()
        self.conv1 = conv3x3(inplanes, planes, stride, augmented=augmented)
        self.bias1b = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = Bias()
        self.conv2 = conv3x3(planes, planes, augmented=augmented)
        self.scale = Scale()
        self.bias2b = Bias()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        biased_x = self.bias1a(x)
        out = self.conv1(biased_x)
        out = self.relu(self.bias1b(out))

        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            identity = self.downsample(biased_x)
            cat_dim = 2 if self.augmented else 1
            identity = torch.cat((identity, torch.zeros_like(identity)), cat_dim)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    FixupResnet-depth where depth is a `4 * 2 * n + 2` with `n` blocks per residual layer.
    The two added layers are the input convolution and fully connected output.
    """

    def __init__(self, depth, num_classes=10, in_planes=16, in_channels=3, augmented=False):
        super(ResNet, self).__init__()
        n_out = num_classes
        assert (depth - 2) % 8 == 0, 'Invalid ResNet depth, has to conform to 8 * n + 2'
        layer_size = (depth - 2) // 8
        layers = 4 * [layer_size]
        self.num_layers = 4 * layer_size
        self.inplanes = in_planes
        self.augmented = augmented
        AdaptiveAvgPool2d = AdaptiveAvgPool2dAug if augmented else nn.AdaptiveAvgPool2d
        self.conv1 = conv3x3(in_channels, in_planes, augmented=augmented)
        self.bias1 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FixupBasicBlock, in_planes, layers[0])
        self.layer2 = self._make_layer(FixupBasicBlock, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(FixupBasicBlock, in_planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(FixupBasicBlock, in_planes * 8, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=2 if augmented else 1)
        self.bias2 = Bias()
        self.fc = nn.Linear(in_planes * 8, n_out)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight,
                                mean=0,
                                std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        AvgPool2d = AvgPool2dAug if self.augmented else nn.AvgPool2d
        if stride != 1:
            downsample = AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, augmented=self.augmented))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, augmented=self.augmented))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bias1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(self.bias2(x))

        return x


class WRNFixupBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, augmented=False, fixup=True):
        super(WRNFixupBasicBlock, self).__init__()
        self.bias1 = Bias() if fixup else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        basemodul = Conv2dAug if augmented else nn.Conv2d
        self.augmented = augmented
        self.conv1 = basemodul(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bias2 = Bias() if fixup else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.bias3 = Bias() if fixup else nn.Identity()
        self.conv2 = basemodul(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias4 = Bias() if fixup else nn.Identity()
        self.scale1 = Scale() if fixup else nn.Identity()
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and basemodul(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bias1(x))
        else:
            out = self.relu1(self.bias1(x))
        if self.equalInOut:
            out = self.bias3(self.relu2(self.bias2(self.conv1(out))))
        else:
            out = self.bias3(self.relu2(self.bias2(self.conv1(x))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.bias4(self.scale1(self.conv2(out)))
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class WRNFixupNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, augmented=False, fixup=True):
        super(WRNFixupNetworkBlock, self).__init__()
        self.augmented = augmented
        self.fixup = fixup
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, self.augmented, self.fixup))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=10, dropRate=0.0, augmented=False, fixup=True):
        super(WideResNet, self).__init__()
        n_out = num_classes
        self.fixup = fixup
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = WRNFixupBasicBlock
        # 1st conv before any network block
        self.num_layers = n * 3
        basemodul = Conv2dAug if augmented else nn.Conv2d
        self.augmented = augmented
        self.conv1 = basemodul(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias1 = Bias() if fixup else nn.Identity()
        # 1st block
        self.block1 = WRNFixupNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, augmented=augmented, fixup=fixup)
        # 2nd block
        self.block2 = WRNFixupNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, augmented=augmented, fixup=fixup)
        # 3rd block
        self.block3 = WRNFixupNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, augmented=augmented, fixup=fixup)
        # global average pooling and classifier
        self.bias2 = Bias() if fixup else nn.Identity()
        self.relu = nn.ReLU()
        self.pool = AvgPool2dAug(8) if augmented else nn.AvgPool2d(8)
        self.fc = nn.Linear(nChannels[3], n_out)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, WRNFixupBasicBlock):
                conv = m.conv1
                k = conv.weight.shape[0] * np.prod(conv.weight.shape[2:])
                nn.init.normal_(conv.weight,
                                mean=0,
                                std=np.sqrt(2. / k) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.convShortcut is not None:
                    cs = m.convShortcut
                    k = cs.weight.shape[0] * np.prod(cs.weight.shape[2:])
                    nn.init.normal_(cs.weight,
                                    mean=0,
                                    std=np.sqrt(2. / k))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.bias1(self.conv1(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = self.pool(out)
        if self.augmented:
            out = out.flatten(start_dim=2)
        else:
            out = out.flatten(start_dim=1)
        out = self.fc(self.bias2(out))
        return out
