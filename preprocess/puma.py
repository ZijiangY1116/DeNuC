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
- puma_folder: the folder of PUMA
- output_folder: the folder to save the data

base on https://puma.grand-challenge.org/dataset/, the default scale is 40x

This file will build six subdataset for linear/knn evaluation:
- 40x, 256: 40x resolution, 256x256 image, 256x256 annotation

For each raw annotated sample, we crop it to small patches, and save them to the output_folder.
"""
import os
import cv2
import glob
import json
import tqdm
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--puma_folder', type=str, default='', help='Path to the puma folder.')
parser.add_argument('--output_folder', type=str, default='', help='Path to the output folder.')
args = parser.parse_args()


def ana_split_file(input_dict, split_file, split_name):

    with open(split_file, 'r') as f:
        split_dict = json.load(f)
        file_names = [file_name[7:-5] for file_name in list(split_dict['anno'].keys())]

        for file_name in file_names:
            assert file_name not in input_dict, 'file name already exists'
            input_dict[file_name] = split_name
    
    return input_dict


def compute_center(point_list):
    contour = np.array(point_list, dtype=np.int32).reshape((-1, 1, 2))
    M = cv2.moments(contour)

    if M['m00'] != 0:
        # The contour has a non-zero area, we can compute the centroid using moments
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        # Degenerate case (e.g., all points are collinear or a single point): fallback to geometric mean
        cx, cy = np.mean(contour[:, 0, :], axis=0)
    
    return cx, cy


def ana_geojson_ann(ann_file):

    tp_name_id_map = {
        'nuclei_tumor': 0,
        'nuclei_lymphocyte': 1,
        'nuclei_plasma_cell': 2,
        'nuclei_histiocyte': 3,
        'nuclei_melanophage': 4,
        'nuclei_neutrophil': 5,
        'nuclei_stroma': 6,
        'nuclei_endothelium': 7,
        'nuclei_epithelium': 8,
        'nuclei_apoptosis': 9
    }

    ann_data = json.load(open(ann_file))['features']

    cells = list()
    for cell_info in ann_data:
        geo = cell_info['geometry']['coordinates']
        tp = cell_info['properties']['classification']['name']
        _id = cell_info['id']

        for true_poly in geo:
            try:
                cx, cy = compute_center(true_poly)
                cells.append([cx, cy, tp_name_id_map[tp]])
            except:
                print(f'Something wrong, may because of the poly includes multiple contour ({len(true_poly)}), try to analyze it')
                for _poly in true_poly:
                    cx, cy = compute_center(_poly)
                    cells.append([cx, cy, tp_name_id_map[tp]])
    
    return np.array(cells)


if __name__ == '__main__':

    # ========================
    # build the metadata
    # ========================
    print('======== build the metadata =======')
    nuclei_ann_folder = os.path.join(args.puma_folder, '01_training_dataset_geojson_nuclei')
    central_ann_folder = os.path.join(args.puma_folder, '01_training_dataset_tif_ROIs')
    context_ann_folder = os.path.join(args.puma_folder, '01_training_dataset_tif_context_ROIs')

    samples = [os.path.splitext(os.path.basename(file))[0] for file in glob.glob(central_ann_folder + '/*.tif')]
    print(f'Fine {len(samples)} samples')

    print('======== Ana all nuclei info =======')
    cell_ann_dict = dict()
    for sample_name in samples:
        ann_file = os.path.join(nuclei_ann_folder, sample_name + '_nuclei' + '.geojson')
        cell_ann_dict[sample_name] = ana_geojson_ann(ann_file)
    
    print('======== Load the split info =======')
    split_meta = dict()
    split_meta = ana_split_file(split_meta, './metadata/puma/train.json', 'train')
    split_meta = ana_split_file(split_meta, './metadata/puma/val.json', 'val')
    split_meta = ana_split_file(split_meta, './metadata/puma/test.json', 'test')
    
    # ========================
    # loop all samples
    # ========================
    meta_dict = {
        'sample_name': list(),
        'split': list()
    }
    for raw_sample_name in tqdm.tqdm(samples):

        raw_crop_x = 0
        raw_crop_y = 0
        crop_size = 1024

        context_image = cv2.imread(os.path.join(central_ann_folder, f'{raw_sample_name}.tif'))

        ann = cell_ann_dict[raw_sample_name]
        split = split_meta[raw_sample_name]

        wsi_mpp = 0.25  # we suppose the default is 40x
        target_mpp = 0.25  # 40x
        target_central_size = int(256 * (target_mpp / wsi_mpp))
        target_patch_sizes = [int(256 * (target_mpp / wsi_mpp))]
        save_patch_sizes = [256]

        x_num = int(np.ceil(1024 / target_central_size))
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

                # crop the max patch from wsi
                wsi_crop = context_image[crop_y:crop_y + target_patch_sizes[-1], crop_x:crop_x + target_patch_sizes[-1]]

                # build the true sample
                for i, target_patch_size in enumerate(target_patch_sizes):

                    sample_name = f'{raw_sample_name}_{x_id}_{y_id}'

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
                    true_patch = cv2.cvtColor(true_patch, cv2.COLOR_BGR2RGB)
                    
                    save_ann = true_ann.copy()
                    save_ann[:, 2][save_ann[:, 2] > 2] = 2

                    data_dict = {
                        'patch': np.array(true_patch, dtype=np.uint8),
                        'ann': np.array(save_ann, dtype=np.int32)
                    }
                    meta_dict['sample_name'].append(sample_name)
                    meta_dict['split'].append(split)

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
                            cv2.circle(draw_img, (x, y), 5, (0, 255, 0), -1)
                        elif t == 3:
                            cv2.circle(draw_img, (x, y), 5, (0, 0, 255), -1)
                        elif t == 4:
                            cv2.circle(draw_img, (x, y), 5, (255, 255, 0), -1)
                        elif t == 5:
                            cv2.circle(draw_img, (x, y), 5, (255, 0, 255), -1)
                        elif t == 6:
                            cv2.circle(draw_img, (x, y), 5, (0, 255, 255), -1)
                        elif t == 7:
                            cv2.circle(draw_img, (x, y), 5, (122, 0, 255), -1)
                        elif t == 8:
                            cv2.circle(draw_img, (x, y), 5, (0, 255, 122), -1)
                        elif t == 9:
                            cv2.circle(draw_img, (x, y), 5, (255, 0, 122), -1)
                    
                    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(preview_save_dir, f'{sample_name}.png'), draw_img)

                    # save the patch and ann
                    np.save(os.path.join(args.output_folder, f'{sample_name}.npy'), data_dict)

    # save the save_dict
    # meta is save as csv
    meta_save_path = os.path.join(args.output_folder, 'meta.csv')
    meta_df = pd.DataFrame(meta_dict)
    meta_df.to_csv(meta_save_path, index=False)
    print(f'All data is saved to {args.output_folder}')