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
import torch.nn as nn

try:
    from torchvision.models import ResNet50_Weights, resnet50
    _HAS_TV_WEIGHTS = True
except ImportError:
    from torchvision.models import resnet50
    ResNet50_Weights = None
    _HAS_TV_WEIGHTS = False


class ResNet50(nn.Module):
    def __init__(self, out_stages=(2, 3, 4), pretrain=True, *args, **kwargs):
        super(ResNet50, self).__init__()
        assert set(out_stages).issubset((1, 2, 3, 4))

        if _HAS_TV_WEIGHTS and pretrain:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            try:
                backbone = resnet50(pretrained=pretrain)
            except TypeError:
                backbone = resnet50(weights=None)

        self.out_stages = out_stages
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        x = self.layer1(x)
        if 1 in self.out_stages:
            outputs.append(x)
        x = self.layer2(x)
        if 2 in self.out_stages:
            outputs.append(x)
        x = self.layer3(x)
        if 3 in self.out_stages:
            outputs.append(x)
        x = self.layer4(x)
        if 4 in self.out_stages:
            outputs.append(x)
        return tuple(outputs)
