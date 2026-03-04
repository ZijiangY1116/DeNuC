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
import numpy as np
import albumentations as A
from torch.utils.data import Dataset


class DeNuCDataset(Dataset):
    """
    Custom Dataset class for DeNuC.
    """

    def __init__(self, metadata, transform=None):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing metadata
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = metadata
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample_info = self.metadata.iloc[idx]
        sample_path = sample_info['sample_path']
        sample_data = np.load(sample_path, allow_pickle=True).item()  # Load the sample data
        img = sample_data['patch']
        coords = sample_data['ann'][:, :2]  # Get the coordinates (x, y)

        # apply transformations if any
        if self.transform:
            sample = self.transform(np.array(img), np.array(coords))
            img, coords = sample['image'], sample['keypoints']
        
        return img, coords
    
    @staticmethod
    def collate_fn(batch):
        
        imgs = list()
        coords = list()
        n_max = -1

        for item in batch:
            imgs.append(item[0])
            coords.append(torch.tensor(item[1], dtype=torch.float32))
            n_max = max(n_max, item[1].shape[0])
        
        imgs = torch.stack(imgs, dim=0)  # B, C, H, W

        # pad coords to the same length
        masks = torch.zeros((len(batch), n_max), dtype=torch.bool)  # B, n_max
        for i in range(len(batch)):
            n_pts = batch[i][1].shape[0]
            masks[i, :n_pts] = 1  # valid points
            pad_size = n_max - n_pts
            if pad_size > 0:
                pad = torch.zeros((pad_size, 2), dtype=torch.float32)
                coords[i] = torch.cat([coords[i], pad], dim=0)

        coords = torch.stack(coords, dim=0)  # B, n_max, 2
        return imgs, coords, masks


class  TrainAugmentation(object):
    """
    Augmentation for DeNuC.
    1. ColorJitter: brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2
    2. RandomRotate90: 0.5
    3. Downscale: scale_range [0.5, 0.7] 0.15
    4. GaussNoise: std_range: [0.1, 0.44]  0.25
    5. ZoomBlur: max_factor: 1.05  p 0.1
    6. HorizontalFlip: 0.5
    7. VerticalFlip: 0.5
    8. Affine:
        rotate: 0
        shear: 0
        scale: [0.8, 0.95]
        p: 0.15 
    """

    def __init__(self, target_size=256, seed=0):
        # augmentation
        self.transfo = A.Compose([
            A.Resize(target_size, target_size),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2),
            A.RandomRotate90(p=0.5),
            A.Downscale(scale_range=[0.5, 0.7], p=0.15),
            A.GaussNoise(p=0.25),
            A.ZoomBlur(max_factor=1.05, p=0.1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(rotate=0, shear=0, scale=[0.8, 0.95], p=0.15),
            A.Normalize(),
            A.pytorch.ToTensorV2()
        ], p=1.0, keypoint_params=A.KeypointParams(format='xy'), seed=seed)
    
    def __call__(self, img, coords, *args, **kwargs):
        return self.transfo(image=img, keypoints=coords)
