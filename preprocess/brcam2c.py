# Modification 2026 Zijiang Yang
# Copyright 2025 Alibaba.
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
"""
args:
- brcam2c_folder: the folder of brcam2c
- wsi_folder: the folder of WSIs
- output_folder: the folder to save the data

This file will build six subdataset for linear/knn evaluation:
- 40x, 256: 40x resolution, 256x256 image, 256x256 annotation

For each raw annotated sample, we crop it to small patches, and save them to the output_folder.
"""

import os
import cv2
import glob
import tqdm
import argparse
import openslide
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--brcam2c_folder', type=str, default='', help='the folder of brcam2c')
parser.add_argument('--wsi_folder', type=str, default='', help='the folder of WSIs')
parser.add_argument('--output_folder', type=str, default='', help='the folder to save the data')
args = parser.parse_args()


def analysis_split_file(split_file, record_dict, split):

    for line in open(split_file):
        sample_info = line.strip()
        sample_name = sample_info.split('_')[0]
        crop_info = sample_info.split('_')[1:3]
        crop_x, crop_y = [int(x) for x in crop_info]

        wsi_name = glob.glob(f'{args.wsi_folder}/{sample_name}*')[0]
        wsi_name = os.path.splitext(os.path.basename(wsi_name))[0]

        record_dict['raw_img_name'].append(os.path.splitext(sample_info)[0])
        record_dict['sample_name'].append(sample_name)
        record_dict['wsi_name'].append(wsi_name)
        record_dict['split'].append(split)
        record_dict['crop_x'].append(crop_x)
        record_dict['crop_y'].append(crop_y)
        record_dict['crop_size'].append(1000)
    
    return record_dict


if __name__ == '__main__':

    # ========================
    # build the metadata
    # ========================
    print('======== build the metadata =======')
    meta_dict = {
        'raw_img_name': list(),
        'sample_name': list(),
        'wsi_name': list(),
        'split': list(),
        'crop_x': list(),
        'crop_y': list(),
        'crop_size': list()
    }

    meta_dict = analysis_split_file(os.path.join(args.brcam2c_folder, 'brca_ds_train.txt'), meta_dict, 'train')
    meta_dict = analysis_split_file(os.path.join(args.brcam2c_folder, 'brca_ds_val.txt'), meta_dict, 'val')
    meta_dict = analysis_split_file(os.path.join(args.brcam2c_folder, 'brca_ds_test.txt'), meta_dict, 'test')

    meta_df = pd.DataFrame(meta_dict).reset_index()

    print(meta_df)

    # ========================
    # loop all samples
    # ========================
    sample_dict = {
        'sample_name': list(),
        'split': list()
    }
    save_dict = dict()
    for row in tqdm.tqdm(meta_df.iterrows()):

        # =========================
        # get the basic information
        # =========================
        raw_img_name = row[1]['raw_img_name']
        split = row[1]['split']
        wsi_name = row[1]['wsi_name']
        raw_crop_x = row[1]['crop_x']
        raw_crop_y = row[1]['crop_y']
        crop_size = 1000  # setting the crop_size == 1000 is ok.

        wsi = openslide.OpenSlide(f'{args.wsi_folder}/{wsi_name}.svs')
        wsi_mpp = eval(wsi.properties['aperio.MPP'])

        # for brcam2c, we should map the coordinates to the raw mpp
        ann_mpp = 0.5
        ann = np.loadtxt(
            f'{args.brcam2c_folder}/labels/{raw_img_name}_gt_class_coords.txt',
            dtype=int,
            delimiter=' '
        )
        ann[:, :2] = ann[:, :2] * (ann_mpp / wsi_mpp)
        ann = ann[:, [1, 0, 2]]  # the order of the columns is x, y, w

        # ===========================
        # split the annotated sample
        # ===========================
        target_mpp = 0.25  # 40x
        target_central_size = int(256 * (target_mpp / wsi_mpp))
        target_patch_sizes = [int(256 * (target_mpp / wsi_mpp))]
        save_patch_sizes = [256]

        # crop the annotated sample based on the target size (no overlap)
        x_num = int(np.ceil(1000 / target_central_size))
        for x_id in range(x_num):
            for y_id in range(x_num):
                shift_x = x_id * target_central_size  # local coord (raw ann patch)
                shift_y = y_id * target_central_size  # local coord (raw ann patch)
                crop_x = raw_crop_x + shift_x  # global coord (wsi)
                crop_y = raw_crop_y + shift_y  # global coord (wsi)

                # crop the annotation
                ann_crop = ann[
                    (ann[:, 0] >= shift_x) & (ann[:, 0] < shift_x + target_central_size) &
                    (ann[:, 1] >= shift_y) & (ann[:, 1] < shift_y + target_central_size)
                ]
                    
                # we skip the patches that have no annotations
                if len(ann_crop) == 0:
                    continue

                ann_crop[:, :2] = ann_crop[:, :2] - np.array([shift_x, shift_y])  # local coord (central patch)
                ann_crop[:, :2] = ann_crop[:, :2] + (target_patch_sizes[-1] - target_central_size) // 2

                # there should have some spatial process
                if crop_x + target_patch_sizes[-1] - raw_crop_x > 1000:
                    # the right part is out of the raw crop, reduce the crop_size
                    crop_size_x = raw_crop_x + 1000 - crop_x
                else:
                    crop_size_x = target_patch_sizes[-1]
                if crop_y + target_patch_sizes[-1] - raw_crop_y > 1000:
                    # the bottom part is out of the raw crop, reduce the crop_size
                    crop_size_y = raw_crop_y + 1000 - crop_y
                else:
                    crop_size_y = target_patch_sizes[-1]

                wsi_crop = np.array(wsi.read_region(
                    (crop_x, crop_y),
                    0,
                    (crop_size_x, crop_size_y)
                ).convert('RGB'), dtype=np.uint8)

                # pad write to right and bottom if need
                if crop_size_x < target_patch_sizes[-1] or crop_size_y < target_patch_sizes[-1]:
                    wsi_crop = cv2.copyMakeBorder(
                        wsi_crop,
                        0, target_patch_sizes[-1] - crop_size_y,
                        0, target_patch_sizes[-1] - crop_size_x,
                        cv2.BORDER_CONSTANT,
                        value=(255, 255, 255)
                    )

                # build the true sample
                for i, target_patch_size in enumerate(target_patch_sizes):
                    
                    sample_name = f'{raw_img_name}_{x_id}_{y_id}'

                    # center crop
                    true_patch = wsi_crop[
                        (target_patch_sizes[-1] - target_patch_size) // 2 : (target_patch_sizes[-1] - target_patch_size) // 2 + target_patch_size,
                        (target_patch_sizes[-1] - target_patch_size) // 2 : (target_patch_sizes[-1] - target_patch_size) // 2 + target_patch_size
                    ]
                    # shift the ann_crop
                    true_ann = ann_crop.copy()
                    true_ann[:, :2] = true_ann[:, :2] - (target_patch_sizes[-1] - target_patch_size) // 2

                    true_size = save_patch_sizes[i]
                    true_scale_ratio = true_size / target_patch_size
                    
                    true_ann[:, :2] = true_ann[:, :2] * true_scale_ratio
                    true_ann = np.array(true_ann, dtype=int)
                    true_patch = cv2.resize(true_patch, (true_size, true_size), interpolation=cv2.INTER_CUBIC)

                    # for linear / knn eval
                    # the bg is ignored
                    # therefore, the ann should - 1
                    save_ann = true_ann.copy()
                    save_ann[:, 2] -= 1
                    data_dict = {
                        'patch': np.array(true_patch, dtype=np.uint8),
                        'ann': np.array(save_ann, dtype=np.int32)
                    }
                    sample_dict['sample_name'].append(sample_name)
                    sample_dict['split'].append(split)

                    # the ann should be in the true_patch
                    # otherwise, there may be some errors
                    assert np.all(true_ann[:, :2] >= 0) and np.all(true_ann[:, :2] < true_size), f'{raw_img_name} {x_id} {y_id}: the ann is out of the patch, please check it'

                    # draw the ann
                    preview_save_dir = os.path.join(args.output_folder, f'preview')
                    os.makedirs(preview_save_dir, exist_ok=True)
                    draw_img = np.array(true_patch)
                    for x, y, t in save_ann:
                        x = int(x)
                        y = int(y)
                        if t == 0:
                            cv2.circle(draw_img, (x, y), 5, (0, 162, 232), -1)
                        elif t == 1:
                            cv2.circle(draw_img, (x, y), 5, (255, 0, 0), -1)
                        elif t == 2:
                            cv2.circle(draw_img, (x, y), 5, (255, 255, 0), -1)
                        else:
                            raise ValueError(f'Unknown annotation type: {t}')
                    
                    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(preview_save_dir, f'{sample_name}.png'), draw_img)
    
                    # svae the patch and ann
                    np.save(os.path.join(args.output_folder, f'{sample_name}.npy'), data_dict)

    # save the save_dict
    meta_save_path = os.path.join(args.output_folder, 'meta.csv')
    sample_df = pd.DataFrame(sample_dict)
    sample_df.to_csv(meta_save_path, index=False)
