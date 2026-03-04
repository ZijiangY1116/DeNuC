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


class AnchorPoints(nn.Module):
    """
    Generate anchor points on feature maps.

    Mainly copy from DPA-P2PNet: https://github.com/windygoo/PromptNucSeg/blob/main/prompter/models/dpa_p2pnet.py
    In addtion, we replace np with torch for better efficiency.
    """
    def __init__(self, space=8):
        super(AnchorPoints, self).__init__()
        self.space = space
    
        self.h = None
        self.w = None
        self.anchors = None

    def forward(self, images):
        bs, _, h, w = images.shape
        
        # try to reuse, if possible
        if self.h == h and self.w == w and self.anchors is not None:
            return self.anchors.repeat(bs, 1, 1, 1)
        else:
            # build mesh
            grid_y, grid_x = torch.meshgrid(
                torch.arange(torch.ceil(torch.tensor(h / self.space)), device=images.device),
                torch.arange(torch.ceil(torch.tensor(w / self.space)), device=images.device),
                indexing='ij'
            )
            anchors = torch.stack([grid_x, grid_y], dim=-1) * self.space

            # correct the origin offset
            w_origin = w % self.space or self.space
            h_origin = h % self.space or self.space
            origin_coord = torch.tensor([w_origin, h_origin], device=images.device) / 2
            anchors = anchors + origin_coord

            # record for future use
            self.h = h
            self.w = w
            self.anchors = anchors

        return anchors.repeat(bs, 1, 1, 1)
