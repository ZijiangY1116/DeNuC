# Copyright 2026 Zijiang Yang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module.conv import ConvModule
from ..module.init_weights import xavier_init


class DeNuCHead(nn.Module):

    def __init__(
        self,
        in_channels,
        num_in,
        num_classes,
        feat_channels=256,
        stacked_convs=4,
        conv_cfg=None,
        norm_cfg=None,
        activation=None,
    ):
        super(DeNuCHead, self).__init__()
        total_in_channels = in_channels

        self.conv_layers = nn.ModuleList()
        for i in range(stacked_convs):
            in_ch = (
                total_in_channels if i == 0 else feat_channels
            )
            conv = ConvModule(
                in_ch,
                feat_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False,
            )
            self.conv_layers.append(conv)
        
        # last
        self.cls_conv = nn.Conv2d(feat_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.reg_conv = nn.Conv2d(feat_channels, 2, kernel_size=1, stride=1, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        # inputs is a list of feature maps from FPN/PAN
        
        # 1. upsample and concatenate feature maps
        x = []
        for feat_i, feat in enumerate(inputs):
            if feat_i == 0:
                x = feat
            else:
                upsampled_feat = F.interpolate(
                    feat, size=inputs[0].shape[2:], mode='nearest'
                )
                x = x + upsampled_feat
        x = x / len(inputs)

        # 2. pass through conv layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # 3. get cls and reg outputs
        cls_out = self.cls_conv(x)  # [B, C, H, W]
        reg_out = self.reg_conv(x)  # [B, 2, H, W]
        return {'cls': cls_out, 'reg': reg_out}
