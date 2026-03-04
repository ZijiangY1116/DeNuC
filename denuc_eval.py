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
"""
This file is used to evaluate saved checkpoints on training/validation datasets.

Most of arguments (e.g., dataset, model architecture) are loaded from the corresponding experiment folder.
The results will be saved in the same experiment folder.
"""

import os
import json
import glob
import yaml
import tqdm
import torch
import shutil
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A

import utils.data as data
from utils.model import build_arch
import utils.basic_utils as basic_utils


def get_args_parse():

    parser = argparse.ArgumentParser("Evaluate saved checkpoints on training/validation datasets", add_help=False)

    parser.add_argument('--exp_name', type=str, default='none', help='Name of the experiment folder where checkpoints are saved')

    # eval settings
    parser.add_argument('--eval_dataset', type=str, required=True, help='Comma separated list of datasets to use for evaluation.')
    parser.add_argument('--eval_mode', type=str, default='val', choices=['train', 'val', 'test'], help='Evaluation mode: train, val, or test')
    parser.add_argument('--nms_dist', type=float, default=12.0, help='NMS distance threshold for filtering predictions.')

    return parser


def eval_denuc(eval_args):
    """
    Main function to evaluate saved checkpoints on training/validation datasets.
    """
    # ======================
    # load experiment config
    # ======================
    # check if exp_folder is a folder
    if os.path.isdir(os.path.join(eval_args.exp_name)):
        exp_folder = eval_args.exp_name
    else:
        # try to ana from output folder
        exp_folder = os.path.join('./outputs', eval_args.exp_name)
    training_arg_file = os.path.join(exp_folder, 'args.yaml')
    training_args = yaml.load(open(training_arg_file, 'r'), Loader=yaml.FullLoader)
    print(f"Loaded experiment config from {training_arg_file}")

    basic_utils.fix_random_seeds(training_args['seed'])

    if eval_args.eval_mode == 'train' or eval_args.eval_mode == 'val':
        output_folder = os.path.join(exp_folder, f'{eval_args.eval_mode}_results')
    elif eval_args.eval_mode == 'test':
        # as the test datasets may be different from validation datasets, we save test results in a separate folder
        output_folder = os.path.join(exp_folder, 'test_results', '-'.join([ds.strip() for ds in eval_args.eval_dataset.split(',')]))
    else:
        raise ValueError(f"Invalid eval_mode: {eval_args.eval_mode}")

    if os.path.exists(output_folder):
        print(f'Warning: {output_folder} folder already exists! Results will be overwritten.')
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # ===================
    # build eval tasks
    # ===================
    if eval_args.eval_mode == 'train' or eval_args.eval_mode == 'val':
        saved_ckpts = glob.glob(os.path.join(exp_folder, '*.pth'))
        # pop checkpoint.pth if exists
        if os.path.join(exp_folder, 'checkpoint.pth') in saved_ckpts:
            saved_ckpts.remove(os.path.join(exp_folder, 'checkpoint.pth'))
        # sort by name
        saved_ckpts = sorted(saved_ckpts)
        print(f"Found {len(saved_ckpts)} saved checkpoints for evaluation.")
    elif eval_args.eval_mode == 'test':
        # check if there is a best_checkpoint.pth
        if os.path.exists(os.path.join(exp_folder, 'best_checkpoint.pth')):
            saved_ckpts = [os.path.join(exp_folder, 'best_checkpoint.pth')]
            print('Test evaluation will be performed on the best checkpoint: best_checkpoint.pth')
        else:
            print(f"No best checkpoint found for evaluation in {exp_folder}! Validation should be performed first to select the best checkpoint. Evaluation will be skipped.")
            return
    else:
        raise ValueError(f"Invalid eval_mode: {eval_args.eval_mode}")

    # ===================
    # build metadata of data
    # ===================
    # 1. build the metadata
    meta_dict = {
        'sample_path': list(),
        'split': list()
    }
    for dataset_name in eval_args.eval_dataset.split(','):
        dataset_name = dataset_name.strip()
        if dataset_name in ['puma', 'brcam2c', 'ocelot', 'pannuke', 'puma_512', 'brcam2c_512', 'ocelot_512']:
            dataset_meta = pd.read_csv(f'./dataset/{dataset_name}/meta.csv')
            for _, row in dataset_meta.iterrows():
                meta_dict['sample_path'].append(os.path.join(f'./dataset/{dataset_name}', f"{row['sample_name']}.npy"))
                meta_dict['split'].append(row['split'])
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    meta_df = pd.DataFrame(meta_dict)
    # build the meta
    meta_df = meta_df[meta_df['split'] == eval_args.eval_mode].copy().reset_index(drop=True)
    print(f"Built metadata for evaluation with {len(meta_df)} samples.")
    meta_df.to_csv(os.path.join(output_folder, 'eval_metadata.csv'), index=False)

    # ===================
    # evaluate checkpoints
    # ===================
    eval_dict = dict()
    for ckpt_path in saved_ckpts:
        print(f'Evaluating checkpoint: {ckpt_path}')

        # init model
        model_net = build_arch.__all__[training_args['arch']]()
        model_weights = torch.load(ckpt_path, weights_only=False)['model']
        model_weights = {k.replace('module.', ''): v for k, v in model_weights.items()}
        model_net.load_state_dict(model_weights)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_net.to(device)
        model_net.eval()

        # inference and evaluation
        infer_res, raw_infer_res = infer_loop(
            eval_meta_df=meta_df,
            model_net=model_net,
            model_infer_size=training_args['img_size'],
            nms_dist=eval_args.nms_dist,
            device=device,
            seed=training_args['seed'],
            large_crop_eval=True if eval_args.eval_mode == 'test' else False
        )
        # save infer res as npy
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        np.save(os.path.join(output_folder, f'{ckpt_name}_infer_res.npy'), infer_res)
        
        if raw_infer_res is not None:
            np.save(os.path.join(output_folder, f'{ckpt_name}_raw_infer_res.npy'), raw_infer_res)
            eval_res_dict = eval_res(raw_infer_res)
        else:
            eval_res_dict = eval_res(infer_res)

        print(f"Checkpoint {ckpt_name}: Precision = {eval_res_dict['precision']:.2f}%, Recall = {eval_res_dict['recall']:.2f}%, F1 = {eval_res_dict['f1']:.2f}%")
        eval_dict[ckpt_name] = eval_res_dict

    # post-process
    with open(os.path.join(output_folder, 'eval_results.json'), 'w') as f:
            json.dump(eval_dict, f, indent=4)

    if eval_args.eval_mode == 'train':
        # for train, only save the evaluation results
        pass
    elif eval_args.eval_mode == 'val':
        # for validation, find the best checkpoint based on F1 score, and save the best checkpoint for later test evaluation
        best_ckpt_name = max(eval_dict.keys(), key=lambda x: eval_dict[x]['f1'])
        print(f"Best checkpoint: {best_ckpt_name} with F1 = {eval_dict[best_ckpt_name]['f1']:.2f}%")
        # copy best checkpoint to exp folder
        shutil.copy(os.path.join(exp_folder, f'{best_ckpt_name}.pth'), os.path.join(exp_folder, 'best_checkpoint.pth'))

        # copy the output of the best checkpoint to the best_checkpoint_infer_res.npy
        shutil.copy(os.path.join(output_folder, f'{best_ckpt_name}_infer_res.npy'), os.path.join(output_folder, 'best_checkpoint_infer_res.npy'))
    elif eval_args.eval_mode == 'test':
        # for test, save the evaluation results and visualization results of the best checkpoint
        assert len(eval_dict) == 1, "There should be only one checkpoint for test evaluation."
        best_ckpt_name = list(eval_dict.keys())[0]
        print(f"Test evaluation: Precision = {eval_dict[best_ckpt_name]['precision']:.2f}%, Recall = {eval_dict[best_ckpt_name]['recall']:.2f}%, F1 = {eval_dict[best_ckpt_name]['f1']:.2f}%")
    else:
        raise ValueError(f"Invalid eval_mode: {eval_args.eval_mode}")
    
    print("Evaluation completed. All results are saved to the output folder.")


@torch.no_grad()
def infer_loop(eval_meta_df, model_net, model_infer_size, nms_dist=12.0, device=None, seed=0, large_crop_eval=False):
    """
    Inference loop for evaluating a checkpoint on the evaluation dataset.
    """
    eval_transfo = A.Compose([
        A.Resize(model_infer_size, model_infer_size),
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ], p=1.0, keypoint_params=A.KeypointParams(format='xy'), seed=seed)

    res_dict = dict()
    raw_res_dict = None

    if large_crop_eval:
        meta_df = eval_meta_df.copy()

        def parse_large_name(sample_path):
            base_name = os.path.splitext(os.path.basename(sample_path))[0]
            parts = base_name.rsplit('_', 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid sample name format: {base_name}. Expected <name>_<x>_<y>.")
            large_name, col_idx, row_idx = parts[0], int(parts[1]), int(parts[2])
            return large_name, col_idx, row_idx

        large_names = []
        crop_col_idx = []
        crop_row_idx = []
        for sample_path in meta_df['sample_path']:
            large_name, col_idx, row_idx = parse_large_name(sample_path)
            large_names.append(large_name)
            crop_col_idx.append(col_idx)
            crop_row_idx.append(row_idx)

        meta_df['large_image_name'] = large_names
        meta_df['crop_col_idx'] = crop_col_idx
        meta_df['crop_row_idx'] = crop_row_idx

        overlap_ratio = 0.25
        score_threshold = 0.5

        raw_res_dict = dict()
        for large_name, group_df in tqdm.tqdm(meta_df.groupby('large_image_name'), desc="Infer large images"):
            group_df = group_df.sort_values(['crop_row_idx', 'crop_col_idx']).reset_index(drop=True)

            # Load patches and infer patch size from the first sample.
            first_sample_path = group_df.iloc[0]['sample_path']
            first_data = np.load(first_sample_path, allow_pickle=True).item()
            first_patch = first_data['patch']
            patch_h, patch_w = first_patch.shape[:2]
            if patch_h != patch_w:
                raise ValueError(f"Patch is not square: {first_sample_path} ({patch_h}x{patch_w}).")

            max_col = int(group_df['crop_col_idx'].max())
            max_row = int(group_df['crop_row_idx'].max())
            large_h = (max_row + 1) * patch_h
            large_w = (max_col + 1) * patch_w

            large_img = np.ones((large_h, large_w, first_patch.shape[2]), dtype=first_patch.dtype) * 255
            patch_meta = {}

            large_gt_coords_list = []
            for _, row in group_df.iterrows():
                sample_path = row['sample_path']
                col_idx = int(row['crop_col_idx'])
                row_idx = int(row['crop_row_idx'])

                sample_data = np.load(sample_path, allow_pickle=True).item()
                patch_img = sample_data['patch']
                coords = sample_data['ann'][:, :2]

                if patch_img.shape[:2] != (patch_h, patch_w):
                    raise ValueError(f"Inconsistent patch size in {sample_path}: {patch_img.shape[:2]} vs {(patch_h, patch_w)}.")

                off_x = col_idx * patch_w
                off_y = row_idx * patch_h
                large_img[off_y:off_y + patch_h, off_x:off_x + patch_w] = patch_img

                patch_meta[sample_path] = {
                    'offset': (off_x, off_y),
                    'gt_coords': coords,
                    'patch_size': (patch_h, patch_w)
                }

                if len(coords) > 0:
                    large_gt_coords_list.append(coords + np.array([off_x, off_y]))

            crop_size = patch_h
            overlap_size = int(crop_size * overlap_ratio)

            cropped_patches, offsets = basic_utils.sliding_window_crop(
                image=large_img,
                crop_size=crop_size,
                overlap_ratio=overlap_ratio
            )

            batch_local_coords = []
            batch_scores = []

            for crop_img in cropped_patches:
                sample = eval_transfo(image=crop_img, keypoints=[])
                img = sample['image'].unsqueeze(0).to(device)

                scale_ratio = model_infer_size / max(crop_img.shape[:2])
                pred = model_net(img)
                pred_coords = pred['pred_coords'].cpu().numpy()[0]
                pred_logits = pred['pred_logits'].softmax(dim=-1).cpu().numpy()[0]

                keep_mask = pred_logits[:, 0] > score_threshold
                pred_coords = pred_coords[keep_mask]
                pred_scores = pred_logits[keep_mask, 0]

                pred_coords = pred_coords / scale_ratio
                batch_local_coords.append(pred_coords)
                batch_scores.append(pred_scores)

            global_coords, global_scores = basic_utils.merge_patches(
                batch_local_coords=batch_local_coords,
                batch_scores=batch_scores,
                batch_offsets=offsets,
                img_size=(large_h, large_w),
                win_size=crop_size,
                overlap_size=overlap_size
            )

            global_coords, global_scores = basic_utils.point_nms(
                points=global_coords,
                scores=global_scores,
                nms_dist=nms_dist,
                meth_type='kdtree'
            )

            if len(large_gt_coords_list) > 0:
                large_gt_coords = np.concatenate(large_gt_coords_list, axis=0)
            else:
                large_gt_coords = np.zeros((0, 2))

            raw_res_dict[large_name] = {
                'gt_coords': large_gt_coords,
                'pred_coords': global_coords,
                'pred_scores': global_scores,
                'eval_nms_dist': nms_dist
            }

            for sample_path, meta in patch_meta.items():
                off_x, off_y = meta['offset']
                patch_h, patch_w = meta['patch_size']
                gt_coords = meta['gt_coords']

                if len(global_coords) > 0:
                    in_x = (global_coords[:, 0] >= off_x) & (global_coords[:, 0] < off_x + patch_w)
                    in_y = (global_coords[:, 1] >= off_y) & (global_coords[:, 1] < off_y + patch_h)
                    patch_mask = in_x & in_y
                    pred_coords = global_coords[patch_mask] - np.array([off_x, off_y])
                    pred_scores = global_scores[patch_mask]
                else:
                    pred_coords = np.zeros((0, 2))
                    pred_scores = np.zeros((0,))

                res_dict[sample_path] = {
                    'gt_coords': gt_coords,
                    'pred_coords': pred_coords,
                    'pred_scores': pred_scores,
                    'eval_nms_dist': nms_dist
                }
    else:
        # evaluate on validation set
        for sample_idx in tqdm.tqdm(range(len(eval_meta_df)), desc="Infer samples"):
            sample_info = eval_meta_df.iloc[sample_idx]
            sample_path = sample_info['sample_path']
            sample_data = np.load(sample_path, allow_pickle=True).item()  # Load the sample data
            raw_img = sample_data['patch']
            coords = sample_data['ann'][:, :2]  # Get the coordinates (x, y)

            # apply transformations
            sample = eval_transfo(image=raw_img, keypoints=coords)
            img, coords = sample['image'], sample['keypoints']
            img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

            scale_ratio = model_infer_size / max(raw_img.shape[:2])

            pred = model_net(img)
            pred_coords = pred['pred_coords'].cpu().numpy()[0]  # [N, 2]
            pred_logits = pred['pred_logits'].softmax(dim=-1).cpu().numpy()[0]  # [N, num_classes]

            # filter low-confidence predictions
            score_threshold = 0.5
            keep_mask = pred_logits[:, 0] > score_threshold  # Assuming class 0
            pred_coords = pred_coords[keep_mask]
            pred_scores = pred_logits[keep_mask, 0]

            pred_coords = pred_coords / scale_ratio  # Rescale back to original image size
            coords = coords / scale_ratio  # Rescale GT coords as well for fair evaluation

            # nms filter
            pred_coords, pred_scores = basic_utils.point_nms(
                points=pred_coords,
                scores=pred_scores,
                nms_dist=nms_dist,
                meth_type='kdtree'
            )

            # save results
            res_dict[sample_path] = {
                'gt_coords': coords,
                'pred_coords': pred_coords,
                'pred_scores': pred_scores,
                'eval_nms_dist': nms_dist
            }

    return res_dict, raw_res_dict


def eval_res(infer_res):
    """
    Evaluate detection results.
    """
    n_det_pred, n_det_gt = 0, 0
    n_det_match = 0
    
    for sample_name in tqdm.tqdm(infer_res.keys(), desc="Evaluating samples"):
        sample_res = infer_res[sample_name]
        gt_coords = sample_res['gt_coords']
        pred_coords = sample_res['pred_coords']
        pred_scores = sample_res['pred_scores']
        eval_nms_dist = sample_res['eval_nms_dist']

        n_det_gt += len(gt_coords)
        n_det_pred += len(pred_coords)

        # match
        matched = basic_utils.get_match(
            gt_points=gt_coords,
            pred_points=pred_coords,
            pred_scores=pred_scores,
            match_dist=eval_nms_dist
        )
        n_det_match += matched
    
    det_recall = n_det_match * 100 / (n_det_gt + 1e-6)
    det_precision = n_det_match * 100 / (n_det_pred + 1e-6)
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall + 1e-6)
    return {'precision': det_precision, 'recall': det_recall, 'f1': det_f1}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("DeNuC", parents=[get_args_parse()])
    eval_args = parser.parse_args()

    eval_denuc(eval_args)
