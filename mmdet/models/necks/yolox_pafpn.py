# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import CSPLayer
from ..utils import sa_layer
from ..utils import eca_block

@NECKS.register_module()
class YOLOXPAFPN(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),
                 attention=0,):     #if attention == 0:none, elif attention == 1:as-net, elif attention == 2:eca-net
                                    #elif attention == 3:both of as and eca
        super(YOLOXPAFPN, self).__init__(init_cfg)

        # define the three attention ways
        self.attenion = attention
        if attention == 1:
            self.featattention128 = sa_layer(128)
            self.featattention256 = sa_layer(256)
        elif attention == 2:
            self.featattention128 = eca_block(128)
            self.featattention256 = eca_block(256)
        elif attention == 3:
            self.featattention128 = sa_layer(128)
            self.featattention256 = eca_block(256)


        # define the three attention ways
        self.attenion = attention
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh
            # print('here is up res in')
            # print(idx)
            # print(feat_heigh.shape)
            # print(feat_low.shape)
            # add attention
            if self.attenion == 1 or self.attenion == 2:
                if idx == 2:
                    feat_heigh = self.featattention256(feat_heigh)
                    feat_low = self.featattention256(feat_low)
                elif idx == 1:
                    feat_heigh = self.featattention128(feat_heigh)
                    feat_low = self.featattention128(feat_low)
            # print('here is up res out')
            # print(idx)
            # print(feat_heigh.shape)
            # print(feat_low.shape)


            upsample_feat = self.upsample(feat_heigh)
            # print('here is up samp in')
            # print(idx)
            # print(upsample_feat.shape)
            # add attention
            if self.attenion == 1 or self.attenion == 2:
                if idx == 2:
                    upsample_feat = self.featattention256(upsample_feat)
                elif idx == 1:
                    upsample_feat = self.featattention128(upsample_feat)
            # print('here is up samp out')
            # print(idx)
            # print(upsample_feat.shape)


            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            # print('here is down res in')
            # print(idx)
            # print(feat_heigh.shape)
            # print(feat_low.shape)
            # add attention
            if self.attenion == 1 or self.attenion == 2:
                if idx == 0:
                    feat_heigh = self.featattention128(feat_heigh)
                    feat_low = self.featattention128(feat_low)
                elif idx == 1:
                    feat_heigh = self.featattention128(feat_heigh)
                    feat_low = self.featattention256(feat_low)
            # print('here is down res out')
            # print(idx)
            # print(feat_heigh.shape)
            # print(feat_low.shape)


            downsample_feat = self.downsamples[idx](feat_low)
            # print('here is down samp in')
            # print(idx)
            # print(downsample_feat.shape)
            # add attention
            if self.attenion == 1 or self.attenion == 2:
                if idx == 0:
                    downsample_feat = self.featattention128(downsample_feat)
                elif idx == 1:
                    downsample_feat = self.featattention256(downsample_feat)
            # print('here is down samp out')
            # print(idx)
            # print(downsample_feat.shape)



            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)
