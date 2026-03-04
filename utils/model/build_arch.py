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
from torch import nn
from .backbone import build_backbone
from .fpn import build_fpn
from .head import build_head
from .module.anchor import AnchorPoints


class DeNuCDet(nn.Module):

    def __init__(self, backbone_cfg, fpn_cfg, head_cfg):
        super(DeNuCDet, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.fpn = build_fpn(fpn_cfg)
        self.head = build_head(head_cfg)
        self.anchor_generator = AnchorPoints(space=8)  # by default, space=8 (downsample factor)

    def forward(self, x):
        features = self.backbone(x)
        fpn_outs = self.fpn(features)
        head_outs = self.head(fpn_outs) 
        anchors = self.anchor_generator(x)  # [B, H', W', 2]
        
        pred_coords = anchors + head_outs['reg'].permute(0, 2, 3, 1).contiguous()  # [B, H', W', 2]
        pred_logits = head_outs['cls'].permute(0, 2, 3, 1).contiguous()  # [B, H', W', num_classes]
        return {
            'pred_coords': pred_coords.flatten(1, 2).contiguous(),  # [B, N, 2]
            'pred_logits': pred_logits.flatten(1, 2).contiguous(),  # [B, N, num_classes]
        }


class  DeNuCUNI2HDet(nn.Module):

    def __init__(self, backbone_cfg, head_cfg):
        super(DeNuCUNI2HDet, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.head = build_head(head_cfg)
        self.anchor_generator = AnchorPoints(space=14)  # by default, space=14, the size of each patch
    
    def forward(self, x):
        features = [self.backbone(x)]
        head_outs = self.head(features)  # [B, head_dim, H', W']
        anchors = self.anchor_generator(x)

        pred_coords = anchors + head_outs['reg'].permute(0, 2, 3, 1).contiguous()  # [B, H', W', 2]
        pred_logits = head_outs['cls'].permute(0, 2, 3, 1).contiguous()  # [B, H', W', num_classes]
        return {
            'pred_coords': pred_coords.flatten(1, 2).contiguous(),  # [B, N, 2]
            'pred_logits': pred_logits.flatten(1, 2).contiguous(),  # [B, N, num_classes]
        }


def denuc_det_shufflenet_x0_5():
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '0.5x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': True,
    }
    fpn_cfg = {
        'name': 'PAN',
        'in_channels': [48, 96, 192],
        'out_channels': 96,
        'start_level': 0,
        'num_outs': 3,
    }
    head_cfg = {
        'name': 'DeNuCHead',
        'in_channels': 96,
        'num_in': 3,
        'num_classes': 2,
        'feat_channels': 96,
        'stacked_convs': 2,
        'norm_cfg': {
            'type': 'BN'
        }
    }

    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_shufflenet_x1_0():
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': True,
    }
    fpn_cfg = {
        'name': 'PAN',
        'in_channels': [116, 232, 464],
        'out_channels': 96,
        'start_level': 0,
        'num_outs': 3,
    }
    head_cfg = {
        'name': 'DeNuCHead',
        'in_channels': 96,
        'num_in': 3,
        'num_classes': 2,
        'feat_channels': 96,
        'stacked_convs': 2,
        'norm_cfg': {
            'type': 'BN'
        }
    }

    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_shufflenet_x1_5():
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.5x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': True,
    }
    fpn_cfg = {
        'name': 'PAN',
        'in_channels': [176, 352, 704],
        'out_channels': 192,
        'start_level': 0,
        'num_outs': 3,
    }
    head_cfg = {
        'name': 'DeNuCHead',
        'in_channels': 192,
        'num_in': 3,
        'num_classes': 2,
        'feat_channels': 192,
        'stacked_convs': 2,
        'norm_cfg': {
            'type': 'BN'
        }
    }

    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_shufflenet_x2_0():
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '2.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': True,
    }
    fpn_cfg = {
        'name': 'PAN',
        'in_channels': [244, 488, 976],
        'out_channels': 192,
        'start_level': 0,
        'num_outs': 3,
    }
    head_cfg = {
        'name': 'DeNuCHead',
        'in_channels': 192,
        'num_in': 3,
        'num_classes': 2,
        'feat_channels': 192,
        'stacked_convs': 2,
        'norm_cfg': {
            'type': 'BN'
        }
    }

    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_r50():
    backbone_cfg = {
        'name': 'ResNet50',
        'out_stages': [2, 3, 4],
        'pretrain': True,
    }
    fpn_cfg = {
        'name': 'PAN',
        'in_channels': [512, 1024, 2048],
        'out_channels': 256,
        'start_level': 0,
        'num_outs': 3,
    }
    head_cfg = {
        'name': 'DeNuCHead',
        'in_channels': 256,
        'num_in': 3,
        'num_classes': 2,
        'feat_channels': 256,
        'stacked_convs': 2,
        'norm_cfg': {
            'type': 'BN'
        }
    }

    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_convnext_t():
    backbone_cfg = {
        'name': 'ConvNeXt',
        'variant': 'tiny',
        'out_stages': [2, 3, 4],
        'pretrain': True,
        'disable_cudnn_benchmark': True,
    }
    fpn_cfg = {
        'name': 'PAN',
        'in_channels': [192, 384, 768],
        'out_channels': 256,
        'start_level': 0,
        'num_outs': 3,
    }
    head_cfg = {
        'name': 'DeNuCHead',
        'in_channels': 256,
        'num_in': 3,
        'num_classes': 2,
        'feat_channels': 256,
        'stacked_convs': 2,
        'norm_cfg': {
            'type': 'BN'
        }
    }

    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_convnext_s():
    backbone_cfg = {
        'name': 'ConvNeXt',
        'variant': 'small',
        'out_stages': [2, 3, 4],
        'pretrain': True,
        'disable_cudnn_benchmark': True,
    }
    fpn_cfg = {
        'name': 'PAN',
        'in_channels': [192, 384, 768],
        'out_channels': 256,
        'start_level': 0,
        'num_outs': 3,
    }
    head_cfg = {
        'name': 'DeNuCHead',
        'in_channels': 256,
        'num_in': 3,
        'num_classes': 2,
        'feat_channels': 256,
        'stacked_convs': 2,
        'norm_cfg': {
            'type': 'BN'
        }
    }

    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_convnext_b():
    backbone_cfg = {
        'name': 'ConvNeXt',
        'variant': 'base',
        'out_stages': [2, 3, 4],
        'pretrain': True,
        'disable_cudnn_benchmark': True,
    }
    fpn_cfg = {
        'name': 'PAN',
        'in_channels': [256, 512, 1024],
        'out_channels': 256,
        'start_level': 0,
        'num_outs': 3,
    }
    head_cfg = {
        'name': 'DeNuCHead',
        'in_channels': 256,
        'num_in': 3,
        'num_classes': 2,
        'feat_channels': 256,
        'stacked_convs': 2,
        'norm_cfg': {
            'type': 'BN'
        }
    }

    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_convnext_l():
    backbone_cfg = {
        'name': 'ConvNeXt',
        'variant': 'large',
        'out_stages': [2, 3, 4],
        'pretrain': True,
        'disable_cudnn_benchmark': True,
    }
    fpn_cfg = {
        'name': 'PAN',
        'in_channels': [384, 768, 1536],
        'out_channels': 256,
        'start_level': 0,
        'num_outs': 3,
    }
    head_cfg = {
        'name': 'DeNuCHead',
        'in_channels': 256,
        'num_in': 3,
        'num_classes': 2,
        'feat_channels': 256,
        'stacked_convs': 2,
        'norm_cfg': {
            'type': 'BN'
        }
    }

    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


__all__ = {
    'denuc_det_shufflenet_x0_5': denuc_det_shufflenet_x0_5,
    'denuc_det_shufflenet_x1_0': denuc_det_shufflenet_x1_0,
    'denuc_det_shufflenet_x1_5': denuc_det_shufflenet_x1_5,
    'denuc_det_shufflenet_x2_0': denuc_det_shufflenet_x2_0,
    'denuc_det_r50': denuc_det_r50,
    'denuc_det_convnext_t': denuc_det_convnext_t,
    'denuc_det_convnext_s': denuc_det_convnext_s,
    'denuc_det_convnext_b': denuc_det_convnext_b,
    'denuc_det_convnext_l': denuc_det_convnext_l,
}
