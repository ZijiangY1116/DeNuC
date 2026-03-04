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
import copy
from .convnext import ConvNeXt
from .resnet import ResNet50
from .shufflenetv2 import ShuffleNetV2


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == 'ShuffleNetV2':
        return ShuffleNetV2(**backbone_cfg)
    if name in {'ResNet50', 'resnet50', 'r50'}:
        return ResNet50(**backbone_cfg)
    if name in {'ConvNeXt', 'convnext'}:
        return ConvNeXt(**backbone_cfg)
    if name in {'convnext_t', 'convnext_tiny'}:
        backbone_cfg['variant'] = 'tiny'
        return ConvNeXt(**backbone_cfg)
    if name in {'convnext_s', 'convnext_small'}:
        backbone_cfg['variant'] = 'small'
        return ConvNeXt(**backbone_cfg)
    if name in {'convnext_b', 'convnext_base'}:
        backbone_cfg['variant'] = 'base'
        return ConvNeXt(**backbone_cfg)
    if name in {'convnext_l', 'convnext_large'}:
        backbone_cfg['variant'] = 'large'
        return ConvNeXt(**backbone_cfg)
    else:
        raise ValueError(f"Backbone '{name}' is not supported.")
