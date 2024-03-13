import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from .blocks import PlainBlock, ResidualBlock, Upsample

conv_dict = {
    2: nn.Conv2d,
    3: nn.Conv3d,
}

transpose_conv_dict = {
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

upsample_dict = {
    2: 'bilinear', 
    3: 'trilinear', 
}

block_dict = {
    'plain': PlainBlock,
    'res': ResidualBlock
}

class UNetEncoder(nn.Module):
    """
        U-Net Encoder (include bottleneck)

        dimension:      2D or 3D input
        input_channels: #channels of input images, e.g. 4 for BraTS multimodal input
        channels_list:  #channels of every levels, e.g. [8, 16, 32, 64, 80, 80]
        block:          Type of conv blocks, choice from PlainBlock and ResidualBlock
    """
    def __init__(self, dimension, input_channels, channels_list, 
                 block:Union[PlainBlock, ResidualBlock]=PlainBlock, **block_kwargs):
        super(UNetEncoder, self).__init__()

        self.dimension = dimension
        self.input_channels = input_channels
        self.channels_list = channels_list    # last is bottleneck
        self.block_type = block

        self.levels = nn.ModuleList()
        for l, num_channels in enumerate(self.channels_list):
            in_channels  = self.input_channels if l == 0 else self.channels_list[l-1]
            out_channels = num_channels
            first_stride = 1 if l == 0 else 2   # level 0 don't downsample

            # 2 blocks per level
            blocks = nn.Sequential(
                block(dimension, in_channels, out_channels, stride=first_stride, **block_kwargs),
                block(dimension, out_channels, out_channels, stride=1, **block_kwargs),
            )
            self.levels.append(blocks)

    def forward(self, x, return_skips=False):
        skips = []

        for s in self.levels:
            x = s(x)
            skips.append(x)

        return skips if return_skips else x
    
    def get_feature(self, x):
        B = x.shape[0]
        bottleneck_feat = self.forward(x, return_skips=False)               # [B, C, H, W]
        feat = F.adaptive_avg_pool2d(bottleneck_feat, (1, 1)).view(B, -1)   # [B, C]
        return feat


class UNetDecoder(nn.Module):

    """
        U-Net Decoder (include bottleneck)

        dimension:        2D or 3D input
        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels in a bottom-up order, e.g. [320, 320, 256, 128, 64, 32]
        deep_supervision: Whether to use deep supervision
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
        block:            Type of conv blocks, better be consistent with encoder

        NOTE: Add sigmoid in the end WILL cause numerical unstability.
    """
    def __init__(self, dimension, output_classes, channels_list, upconv=True, deep_supervision=False, ds_layer=0,
                 block:Union[PlainBlock, ResidualBlock]=PlainBlock, **block_kwargs):
        super(UNetDecoder, self).__init__()

        self.dimension = dimension
        self.output_classes = output_classes
        self.channels_list = channels_list                    # first is bottleneck
        self.deep_supervision = deep_supervision
        self.block_type = block
        num_upsample = len(self.channels_list) - 1
        assert ds_layer <= num_upsample

        conv_type = conv_dict[dimension]
        transpose_conv_type = transpose_conv_dict[dimension]
        upsample_mode = upsample_dict[dimension]

        # decoder
        self.levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for l in range(num_upsample):         # exclude bottleneck
            in_channels  = self.channels_list[l]
            out_channels = self.channels_list[l+1]

            # transpose conv
            if upconv:
                upsample = transpose_conv_type(
                    in_channels, out_channels, kernel_size=2, stride=2)
            else:
                upsample = nn.Sequential([
                    conv_type(in_channels, out_channels, kernel_size=1, stride=1), 
                    Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
                ])
            self.upsamples.append(upsample)

            # 2 blocks per level
            blocks = nn.Sequential(
                block(dimension, out_channels * 2, out_channels, stride=1, **block_kwargs),
                block(dimension, out_channels, out_channels, stride=1, **block_kwargs),
            )
            self.levels.append(blocks)
        
        # seg output 
        self.seg_output = conv_type(
            self.channels_list[-1], self.output_classes, kernel_size=1, stride=1)

        # mid-layer deep supervision
        if (self.deep_supervision) and (ds_layer > 1):
            self.ds_layer_list = list(range(num_upsample - ds_layer, num_upsample - 1))
            self.ds = nn.ModuleList()
            for l in range(num_upsample - 1):
                if l in self.ds_layer_list:
                    in_channels = self.channels_list[l+1]
                    up_factor = in_channels // self.channels_list[-1]
                    assert up_factor > 1        # otherwise downsample

                    ds = nn.Sequential(
                        conv_type(in_channels, self.output_classes, kernel_size=1, stride=1),
                        Upsample(scale_factor=up_factor, mode=upsample_mode, align_corners=False),
                    )
                else:
                    ds = None     # for easier indexing

                self.ds.append(ds)

    def forward(self, skips, return_ds=False):
        skips = skips[::-1]     # reverse so that bottleneck is the first
        x = skips.pop(0)        # bottleneck

        ds_outputs = []
        for l, feat in enumerate(skips):
            x = self.upsamples[l](x)            # upsample last-level feat
            x = torch.cat([feat, x], dim=1)     # concat upsampled feat and same-level skip feat
            x = self.levels[l](x)               # concated feat to conv

            if return_ds and (self.deep_supervision) and (l in self.ds_layer_list):
                ds_outputs.append(self.ds[l](x))

        if return_ds:
            return [self.seg_output(x)] + ds_outputs[::-1]  # reverse back
        else:
            return self.seg_output(x)
    
    def get_feature(self, skips, return_ds=False):
        skips = skips[::-1]     # reverse so that bottleneck is the first
        x = skips.pop(0)        # bottleneck

        ds_feats = []
        ds_outputs = []
        for l, feat in enumerate(skips):
            x = self.upsamples[l](x)            # upsample last-level feat
            x = torch.cat([feat, x], dim=1)     # concat upsampled feat and same-level skip feat
            x = self.levels[l](x)               # concated feat to conv

            if return_ds and (self.deep_supervision) and (l in self.ds_layer_list):
                ds_feats.append(x)
                ds_outputs.append(self.ds[l](x))

        if return_ds:
            outputs = [self.seg_output(x)] + ds_outputs[::-1]  # reverse back
            feats = [x] + ds_feats[::-1]
            return outputs, feats
        else:
            return self.seg_output(x), x
        

class UNet(nn.Module):
    """
        U-Net

        dimension:        2D or 3D input
        input_channels:   #channels of input images, e.g. 4 for BraTS multimodal input
        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels in a top-down order, e.g. [32, 64, 128, 256, 320, 320]
        block_type:       Type of conv blocks, choice from 'plain' (PlainBlock) and 'res' (ResidualBlock)
        deep_supervision: Whether to use deep supervision in decoder
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
    """
    def __init__(self, dimension, input_channels, output_classes, channels_list, deep_supervision=False, 
                 ds_layer=0, block_type='plain', **block_kwargs):
        super(UNet, self).__init__()

        block = block_dict[block_type]
        self.encoder = UNetEncoder(dimension, input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder(dimension, output_classes, channels_list[::-1], block=block, 
            deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

    def forward(self, x, return_ds=False):
        return self.decoder(self.encoder(x, return_skips=True), return_ds=return_ds)

    def get_enc_feature(self, x):
        return self.encoder.get_feature(x)

    def get_pixel_feature(self, x, return_ds=False):
        return self.decoder.get_feature(self.encoder(x, return_skips=True), return_ds=return_ds)
