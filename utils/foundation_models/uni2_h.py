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
import os
import torch
import timm
from torch import nn
from .nuclei_extractor import MultiLayerFeatureExtractorHead
from torchvision import transforms


class UNI2HFeatureExtractor(nn.Module):

    def __init__(self, model_weights='/nfs_diskpool/weights/pathology/UNI2-H/pytorch_model.bin'):
        super().__init__()

        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }

        self.basic_model = timm.create_model(**timm_kwargs)

        self.basic_model.load_state_dict(torch.load(model_weights, map_location="cpu"))

        self.dense_feature_extractor = MultiLayerFeatureExtractorHead()
        self.out_dim = 1536

        # init normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, input_img, input_coords=None):
        input_size = input_img.shape[-1]
        bs = input_img.shape[0]

        pred = self.basic_model.forward_features(input_img)
        # please note that
        # 1. cls token
        # 2. reg tokens
        feat_map = pred[:, 1 + 8:, :]  # remove cls token and reg tokens
        h, w = int(input_img.shape[2] / 14), int(input_img.shape[3] / 14)
        feat_map = feat_map.reshape(bs, h, w, -1).permute(0, 3, 1, 2)
        
        return self.dense_feature_extractor([feat_map], input_coords, input_size)
