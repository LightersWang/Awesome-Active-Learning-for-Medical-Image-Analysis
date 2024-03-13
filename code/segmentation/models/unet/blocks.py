import torch
import torch.nn as nn
import torch.nn.functional as F


norm_dict = {
    'instance': {
        2: nn.InstanceNorm2d,
        3: nn.InstanceNorm3d,
    },
    'batch': {
        2: nn.BatchNorm2d,
        3: nn.BatchNorm3d,
    },
}

conv_dict = {
    2: nn.Conv2d,
    3: nn.Conv3d,
}

dropout_dict = {
    2: nn.Dropout2d,
    3: nn.Dropout3d,
}


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)


class PlainBlock(nn.Module):
    def __init__(self, dimension, input_channels, output_channels, stride=1, kernel_size=3, 
                 normalization='instance', dropout_prob=None):
        super(PlainBlock, self).__init__()

        conv_type = conv_dict[dimension]
        norm_type = norm_dict[normalization][dimension]
        dropout_type = dropout_dict[dimension]

        conv = conv_type(input_channels, output_channels, kernel_size, stride=stride, 
                         padding=(kernel_size - 1) // 2, bias=True)

        do = Identity() if dropout_prob is None else dropout_type(p=dropout_prob, inplace=True)

        norm = norm_type(output_channels, eps=1e-5, affine=True)

        nonlin = nn.LeakyReLU(inplace=True)

        self.all = nn.Sequential(conv, do, norm, nonlin)

    def forward(self, x):
        return self.all(x)


class ResidualBlock(nn.Module):
    def __init__(self, dimension, input_channels, output_channels, stride=1, kernel_size=3, 
                 norm_key='instance', dropout_prob=None):
        super(ResidualBlock, self).__init__()

        conv_type = conv_dict[dimension]
        norm_type = norm_dict[norm_key][dimension]
        dropout_type = dropout_dict[dimension]

        conv = conv_type(input_channels, output_channels, kernel_size, stride=stride, 
                         padding=(kernel_size - 1) // 2, bias=True)

        norm = norm_type(output_channels, eps=1e-5, affine=True)

        do = Identity() if dropout_prob is None else dropout_type(p=dropout_prob, inplace=True)

        nonlin = nn.LeakyReLU(inplace=True)

        self.all = nn.Sequential(conv, norm, do, nonlin)

        # downsample residual
        if (input_channels != output_channels) or (stride != 1):
            self.downsample_skip = nn.Sequential(
                conv_type(input_channels, output_channels, 1, stride, bias=True),
                norm_type(output_channels, eps=1e-5, affine=True), 
            )
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x

        out = self.all(x)

        residual = self.downsample_skip(x)

        return residual + out
