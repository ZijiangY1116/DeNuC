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
- ocelot_folder: the folder of ocelot
- wsi_folder: the folder of WSIs
- output_folder: the folder to save the data

This file will build six subdataset for linear/knn evaluation:
- 40x, 256: 40x resolution, 256x256 image, 256x256 annotation

For each raw annotated sample, we crop it to small patches, and save them to the output_folder.
"""

import os
import cv2
import json
import glob
import tqdm
import argparse
import openslide
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--ocelot_folder', type=str, default='', help='the folder of ocelot')
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

    with open(args.ocelot_folder + '/metadata.json', 'r') as f:
        total_meta = json.load(f)
    sample_meta = total_meta['sample_pairs']

    # ana sample meta
    meta_dict = {
        'raw_img_name': list(),
        'sample_name': list(),
        'wsi_name': list(),
        'split': list(),
        'crop_x': list(),
        'crop_y': list(),
        'crop_size': list(),
        'cell_data_mpp': list(),
    }

    for sample_name in sample_meta.keys():
        slide_name = sample_meta[sample_name]['slide_name']
        crop_x = sample_meta[sample_name]['cell']['x_start']
        crop_y = sample_meta[sample_name]['cell']['y_start']

        crop_x_end = sample_meta[sample_name]['cell']['x_end']
        crop_y_end = sample_meta[sample_name]['cell']['y_end']

        split = sample_meta[sample_name]['subset']

        cell_patch_mpp_x = sample_meta[sample_name]['cell']['resized_mpp_x']
        cell_patch_mpp_y = sample_meta[sample_name]['cell']['resized_mpp_y']

        assert cell_patch_mpp_x == cell_patch_mpp_y
        assert crop_x_end - crop_x == crop_y_end - crop_y

        meta_dict['raw_img_name'].append(sample_name)
        meta_dict['sample_name'].append(sample_name)
        meta_dict['wsi_name'].append(slide_name)
        meta_dict['split'].append(split)
        meta_dict['crop_x'].append(crop_x)
        meta_dict['crop_y'].append(crop_y)
        meta_dict['crop_size'].append(crop_x_end - crop_x)
        meta_dict['cell_data_mpp'].append(cell_patch_mpp_x)

    meta_df = pd.DataFrame(meta_dict).reset_index()

    print(meta_df)

    ann_folder = os.path.join(args.ocelot_folder, 'annotations')
    image_folder = os.path.join(args.ocelot_folder, 'images')

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
        crop_size = row[1]['crop_size']
        cell_data_mpp = row[1]['cell_data_mpp']

        # ==========================
        # ana the cells from the annotations
        # ==========================
        image_file = os.path.join(image_folder, split, 'cell', f'{raw_img_name}.jpg')
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_file = os.path.join(ann_folder, split, 'cell', f'{raw_img_name}.csv')
        try:
            ann = pd.read_csv(ann_file, header=None).values.astype(int)
        except:
            continue
        
        # ann[:, :2] = ann[:, :2] * (cell_data_mpp / wsi_mpp)
        assert np.max(ann[:, :2]) < 1024, 'There may be some error that the ann is out of the image'

        # ===========================
        # split the annotated sample
        # ===========================
        target_patch_sizes = 256

        # crop the annotated sample based on the target size (no overlap)
        x_num = int(np.ceil(1024 / target_patch_sizes))
        for x_id in range(x_num):
            for y_id in range(x_num):
                shift_x = x_id * target_patch_sizes  # local coord (raw ann patch)
                shift_y = y_id * target_patch_sizes  # local coord (raw ann patch)
                crop_x = raw_crop_x + shift_x  # global coord (wsi)
                crop_y = raw_crop_y + shift_y  # global coord (wsi)

                # crop the annotation
                ann_crop = ann[
                    (ann[:, 0] >= shift_x) & (ann[:, 0] < shift_x + target_patch_sizes) &
                    (ann[:, 1] >= shift_y) & (ann[:, 1] < shift_y + target_patch_sizes)
                ]
                    
                # we skip the patches that have no annotations
                # if len(ann_crop) == 0:
                #     continue

                ann_crop[:, :2] = ann_crop[:, :2] - np.array([shift_x, shift_y])  # local coord (central patch)

                patch_img = image[shift_y:shift_y + target_patch_sizes, shift_x:shift_x + target_patch_sizes]
                sample_name = f'{raw_img_name}_{x_id}_{y_id}'

                # for linear / knn eval
                # the bg is ignored
                # therefore, the ann should - 1
                save_ann = ann_crop.copy()
                save_ann[:, 2] -= 1
                data_dict = {
                    'patch': np.array(patch_img, dtype=np.uint8),
                    'ann': np.array(save_ann, dtype=np.int32)
                }
                sample_dict['sample_name'].append(sample_name)
                sample_dict['split'].append(split)

                # the ann should be in the true_patch
                # otherwise, there may be some errors
                assert np.all(save_ann[:, :2] >= 0) and np.all(save_ann[:, :2] < target_patch_sizes), f'{raw_img_name} {x_id} {y_id}: the ann is out of the patch, please check it'

                # draw the ann
                preview_save_dir = os.path.join(args.output_folder, f'preview')
                os.makedirs(preview_save_dir, exist_ok=True)
                draw_img = np.array(patch_img)
                for x, y, t in save_ann:
                    x = int(x)
                    y = int(y)
                    if t == 0:
                        cv2.circle(draw_img, (x, y), 5, (0, 162, 232), -1)
                    elif t == 1:
                        cv2.circle(draw_img, (x, y), 5, (255, 0, 0), -1)
                
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(preview_save_dir, f'{sample_name}.png'), draw_img)

                # svae the patch and ann
                np.save(os.path.join(args.output_folder, f'{sample_name}.npy'), data_dict)

    # save the save_dict
    meta_save_path = os.path.join(args.output_folder, 'meta.csv')
    sample_df = pd.DataFrame(sample_dict)
    sample_df.to_csv(meta_save_path, index=False)