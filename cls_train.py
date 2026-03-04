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

import datetime
import os
import tqdm
import yaml
import json
import torch
import argparse
import pandas as pd
import numpy as np
from torch import nn
import albumentations as A
from pathlib import Path

import utils.basic_utils as basic_utils
from utils.foundation_models import FMFacotry
from torch.utils.data import DataLoader, Dataset


def get_args_parse():

    parser = argparse.ArgumentParser(description='Train KNN classifier for nuclei classification with a fixed Foundation Model.', add_help=False)

    parser.add_argument('--exp_name', type=str, required=True, help='The name of the experiment.')
    parser.add_argument('--output_dir', default='./output', type=str, 
                        help='Directory to save outputs (logs, checkpoints, etc.). ' \
                        'Please note that the output_dir will be automatically modified to include ' \
                        'a timestamp and experiment name to avoid overwriting previous experiments. ' \
                        'Therefore, do not use this argument.')
    parser.add_argument('--dataset', type=str, default='brcam2c', help='The dataset to train the classifier on.')

    parser.add_argument('--use_gt_coords', action='store_true', help='Whether to use the GT coordinates as the detection results. This is used for ablation study to show the upper bound performance of the classifier.')
    parser.add_argument('--match_dist', type=float, default=12.0, help='The distance threshold to match the detected coordinates with GT coordinates.')
    
    # model settings
    parser.add_argument('--det_exp_name', type=str, default='None', help='The name of the experiment.')
    parser.add_argument('--det_nms_dist', type=float, default=12.0, help='NMS distance threshold for filtering predictions.')

    parser.add_argument('--det_infer_size', type=int, default=256, help='The input image size for the detection model during inference. The images will be resized to this size before being fed into the detection model.')
    parser.add_argument('--cls_infer_size', type=int, default=224, help='The input image size for the foundation model during inference. The images will be resized to this size before being fed into the foundation model.')

    # only for linear
    parser.add_argument('--train_lr', type=float, default=1e-2, help='learning rate for linear classifier')
    parser.add_argument('--train_bs', type=int, default=256, help='batch size for training linear classifier')
    parser.add_argument('--train_epochs', type=int, default=100, help='number of epochs for training linear classifier')

    return parser


class DeNuCClsWrapper(object):

    def __init__(self, fm_model, nms_dist=12.0):
        self.fm_model = fm_model
        self.nms_dist = nms_dist

    def __call__(self, cls_input_imgs, cls_scale_ratio=1.0, det_coords=None):
        # input_imgs: [B, C, H, W]
        with torch.no_grad():

            # 1. get the det results
            det_scores = [np.array([1.0] * len(coords)) for coords in det_coords]  # assign a score of 1.0 for all coordinates
            # 2. get the fm features of each detected point
            fm_feats = self.fm_forward(cls_input_imgs, [coords * (cls_scale_ratio) for coords in det_coords])  # scale the coordinates to match the resized image size (cls_infer_size)
            return fm_feats, det_coords, det_scores
    
    def fm_forward(self, input_imgs, det_coords):
        # input_imgs: [B, C, H, W]
        # det_coords: list of [n_valid, 2] for each image in the batch
        fm_feats = self.fm_model(input_imgs, [torch.from_numpy(coords) for coords in det_coords])  # list of [n_valid, feat_dim] for each image in the batch
        return fm_feats


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset in ['brcam2c']:
        tps = ['Lym.', 'Tum.', 'Oth.']
    elif args.dataset in ['puma']:
        tps = ['Tum.', 'Lym.', 'Oth.']
    elif args.dataset in ['ocelot']:
        tps = ['Oth.', 'Tum.']
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")

    # ======================
    # init the det model
    # ======================
    if args.use_gt_coords:
        # for this case, there is no need to load the detection model
        seed = 0
    else:
        if os.path.isdir(args.det_exp_name):
            det_exp_folder = args.det_exp_name
        else:
            det_exp_folder = os.path.join('./outputs', args.det_exp_name)
        det_args = yaml.load(open(os.path.join(det_exp_folder, 'args.yaml'), 'r'), Loader=yaml.FullLoader)
        seed = det_args['seed']
    
    basic_utils.fix_random_seeds(seed)

    # ======================
    # init the foundation model
    # ======================
    fm_model = FMFacotry.get_model('uni2_h')
    fm_model.to(device)
    fm_model.eval()
    print('Init foundation model successfully!')

    # ======================
    # init the Wrapper
    # ======================
    end2end_model = DeNuCClsWrapper(
        fm_model=fm_model,
        nms_dist=args.det_nms_dist,
    )
    print('Init wrapper model successfully!')

    # ===================
    # build metadata of data
    # ===================
    meta_dict = {
        'sample_path': list(),
        'split': list()
    }
    if args.dataset in ['puma', 'brcam2c', 'ocelot']:
        dataset_meta = pd.read_csv(f'./dataset/{args.dataset}/meta.csv')
        for _, row in dataset_meta.iterrows():
            meta_dict['sample_path'].append(os.path.join(f'./dataset/{args.dataset}', f"{row['sample_name']}.npy"))
            meta_dict['split'].append(row['split'])
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")
    meta_df = pd.DataFrame(meta_dict)

    train_meta_df = meta_df[meta_df['split'] == 'train'].copy().reset_index(drop=True)
    test_meta_df = meta_df[meta_df['split'] == 'test'].copy().reset_index(drop=True)
    print(f"Train samples: {len(train_meta_df)}, Test samples: {len(test_meta_df)}")

    train_meta_df.to_csv(os.path.join(args.output_dir, 'train_metadata.csv'), index=False)
    test_meta_df.to_csv(os.path.join(args.output_dir, 'test_metadata.csv'), index=False)

    # ===================
    # inference all the data and save the feature
    # ===================
    # load the det res
    test_det_res_path = os.path.join(det_exp_folder, 'test_results', args.dataset, 'best_checkpoint_infer_res.npy')
    test_det_res = np.load(test_det_res_path, allow_pickle=True).item()

    train_set_infer_res = infer_loop(train_meta_df, end2end_model, cls_infer_size=args.cls_infer_size, device=device, seed=seed, use_gt_coords=True)
    test_set_infer_res = infer_loop(test_meta_df, end2end_model, cls_infer_size=args.cls_infer_size, device=device, seed=seed, use_gt_coords=args.use_gt_coords, use_det_res=True, det_res=test_det_res)

    np.save(os.path.join(args.output_dir, 'train_set_infer_res.npy'), train_set_infer_res)
    np.save(os.path.join(args.output_dir, 'test_set_infer_res.npy'), test_set_infer_res)
    print('Inference completed and results saved!')

    if args.dataset in ['brcam2c']:
        n_cls = 3
    elif args.dataset in ['puma']:
        n_cls = 3
    elif args.dataset in ['ocelot']:
        n_cls = 2
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")
    
    # 1. build the train set for classifier
    train_feats = list()
    train_labels = list()
    for _, infer_res in train_set_infer_res.items():
        _, matched_gt_indices = basic_utils.get_match(
            gt_points=infer_res['gt_coords'],
            pred_points=infer_res['det_coords'],
            pred_scores=infer_res['det_scores'],
            match_dist=args.match_dist,
            return_index=False,
            return_matched_indices=True
        )
        train_feats.append(infer_res['fm_feats'].cpu().numpy())  # [n_det, feat_dim]
        if len(matched_gt_indices) == 0:
            train_labels.append(np.array([n_cls] * infer_res['fm_feats'].shape[0]))  # if any detections, all are background
        else:
            train_labels.append(np.array([infer_res['gt_tps'][idx] if idx != -1 else n_cls for idx in matched_gt_indices]))  # [n_det, ]

        assert train_feats[-1].shape[0] == train_labels[-1].shape[0], "The number of features and labels should be the same for each sample. But got {} features and {} labels.".format(train_feats[-1].shape, train_labels[-1])
    train_feats = np.concatenate(train_feats, axis=0)  # [total_n_det, feat_dim]
    train_labels = np.concatenate(train_labels, axis=0)  # [total_n_det, ]
    print(train_feats.shape, train_labels.shape)
    print(f'Build train set for classification successfully! Total samples: {len(train_labels)}.')

    # ===================
    # classifier
    # ===================
    val_det_res_path = os.path.join(det_exp_folder, 'val_results', 'best_checkpoint_infer_res.npy')
    val_det_res = np.load(val_det_res_path, allow_pickle=True).item()
    # build the val set
    val_meta_df = meta_df[meta_df['split'] == 'val'].copy().reset_index(drop=True)
    val_set_infer_res = infer_loop(val_meta_df, end2end_model, cls_infer_size=args.cls_infer_size, device=device, seed=seed, use_gt_coords=args.use_gt_coords, use_det_res=True, det_res=val_det_res)
    # svae the val set inference results
    np.save(os.path.join(args.output_dir, 'val_set_infer_res.npy'), val_set_infer_res)
    # 1. build the train set for classifier
    val_feats = list()
    val_labels = list()
    for _, infer_res in val_set_infer_res.items():
        _, matched_gt_indices = basic_utils.get_match(
            gt_points=infer_res['gt_coords'],
            pred_points=infer_res['det_coords'],
            pred_scores=infer_res['det_scores'],
            match_dist=args.match_dist,
            return_index=False,
            return_matched_indices=True
        )
        val_feats.append(infer_res['fm_feats'].cpu().numpy())  # [n_det, feat_dim]
        if len(matched_gt_indices) == 0:
            val_labels.append(np.array([n_cls] * infer_res['fm_feats'].shape[0]))  # if any detections, all are background
        else:
            val_labels.append(np.array([infer_res['gt_tps'][idx] if idx != -1 else n_cls for idx in matched_gt_indices]))  # [n_det, ]

        assert val_feats[-1].shape[0] == val_labels[-1].shape[0], "The number of features and labels should be the same for each sample. But got {} features and {} labels.".format(val_feats[-1].shape, val_labels[-1])
    val_feats = np.concatenate(val_feats, axis=0)  # [total_n_det, feat_dim]
    val_labels = np.concatenate(val_labels, axis=0)  # [total_n_det, ]

    test_pred_dict = run_eval_linear(
        train_feats=torch.from_numpy(train_feats).float(),
        train_labels=torch.from_numpy(train_labels).long(),
        val_feats=torch.from_numpy(val_feats).float(),
        val_labels=torch.from_numpy(val_labels).long(),
        test_res=test_set_infer_res,
        n_cls=n_cls,
        device=device,
        output_folder=args.output_dir
    )

    # build the evaluation results for the test set
    gt_points = list()
    gt_cls = list()
    pred_points = list()
    pred_cls = list()
    pred_scores = list()
    for test_img_path in test_pred_dict.keys():
        
        # filter out the pred_cls is background
        valid_mask = test_pred_dict[test_img_path] != n_cls

        gt_points.append(test_set_infer_res[test_img_path]['gt_coords'])
        gt_cls.append(test_set_infer_res[test_img_path]['gt_tps'])
        pred_points.append(test_set_infer_res[test_img_path]['det_coords'][valid_mask])
        pred_cls.append(test_pred_dict[test_img_path][valid_mask])
        pred_scores.append(test_set_infer_res[test_img_path]['det_scores'][valid_mask])

    def parse_large_name(sample_path):
        base_name = os.path.splitext(os.path.basename(sample_path))[0]
        parts = base_name.rsplit('_', 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid sample name format: {base_name}. Expected <name>_<x>_<y>.")
        large_name, col_idx, row_idx = parts[0], int(parts[1]), int(parts[2])
        return large_name, col_idx, row_idx

    large_groups = dict()
    for sample_path in test_pred_dict.keys():
        large_name, col_idx, row_idx = parse_large_name(sample_path)
        if large_name not in large_groups:
            large_groups[large_name] = []
        large_groups[large_name].append((sample_path, col_idx, row_idx))

    large_gt_points = []
    large_gt_cls = []
    large_pred_points = []
    large_pred_cls = []
    large_pred_scores = []

    for large_name, entries in large_groups.items():
        entries = sorted(entries, key=lambda x: (x[2], x[1]))
        first_sample_path = entries[0][0]
        first_data = np.load(first_sample_path, allow_pickle=True).item()
        first_patch = first_data['patch']
        patch_h, patch_w = first_patch.shape[:2]
        if patch_h != patch_w:
            raise ValueError(f"Patch is not square: {first_sample_path} ({patch_h}x{patch_w}).")

        tmp_gt_points = []
        tmp_gt_cls = []
        tmp_pred_points = []
        tmp_pred_cls = []
        tmp_pred_scores = []

        for sample_path, col_idx, row_idx in entries:
            off_x = col_idx * patch_w
            off_y = row_idx * patch_h

            gt_coords = test_set_infer_res[sample_path]['gt_coords']
            gt_tps = test_set_infer_res[sample_path]['gt_tps']

            valid_mask = test_pred_dict[sample_path] != n_cls
            det_coords = test_set_infer_res[sample_path]['det_coords'][valid_mask]
            det_scores = test_set_infer_res[sample_path]['det_scores'][valid_mask]
            det_cls = test_pred_dict[sample_path][valid_mask]

            if len(gt_coords) > 0:
                tmp_gt_points.append(gt_coords + np.array([off_x, off_y]))
                tmp_gt_cls.append(gt_tps)

            if len(det_coords) > 0:
                tmp_pred_points.append(det_coords + np.array([off_x, off_y]))
                tmp_pred_scores.append(det_scores)
                tmp_pred_cls.append(det_cls)

        if len(tmp_gt_points) > 0:
            large_gt_points.append(np.concatenate(tmp_gt_points, axis=0))
            large_gt_cls.append(np.concatenate(tmp_gt_cls, axis=0))
        else:
            large_gt_points.append(np.zeros((0, 2)))
            large_gt_cls.append(np.zeros((0,), dtype=int))

        if len(tmp_pred_points) > 0:
            large_pred_points.append(np.concatenate(tmp_pred_points, axis=0))
            large_pred_cls.append(np.concatenate(tmp_pred_cls, axis=0))
            large_pred_scores.append(np.concatenate(tmp_pred_scores, axis=0))
        else:
            large_pred_points.append(np.zeros((0, 2)))
            large_pred_cls.append(np.zeros((0,), dtype=int))
            large_pred_scores.append(np.zeros((0,)))

    large_res_dict, large_res_table = basic_utils.multi_cls_eval(
        gt_points=large_gt_points,
        gt_classes=large_gt_cls,
        pred_points=large_pred_points,
        pred_scores=large_pred_scores,
        pred_classes=large_pred_cls,
        n_cls=n_cls,
        match_dist=args.match_dist,
        tps=tps
    )
    print(large_res_table)
    with open(os.path.join(args.output_dir, 'test_eval_results.json'), 'w') as f:
        json.dump(large_res_dict, f, indent=4)
    
    print('All done!')
    print('output folder: {}'.format(args.output_dir))


class FeatureEvalDataset(Dataset):

    def __init__(self, all_feats, all_labels):
        super().__init__()

        self.all_feats = all_feats
        self.all_labels = all_labels

        print(self.all_feats.shape)
        print(self.all_labels.shape)
    
    def __len__(self):
        return len(self.all_feats)
    
    def __getitem__(self, idx):
        return self.all_feats[idx], self.all_labels[idx]


def build_eval_ds_dl(eval_feats, eval_labels, shuffle=False, drop_last=False):
    _ds = FeatureEvalDataset(eval_feats, eval_labels)
    _dl = DataLoader(
        _ds,
        batch_size=args.train_bs,
        shuffle=shuffle,
        num_workers=4,
        drop_last=drop_last,
        pin_memory=True
    )
    return _dl


def run_eval_linear(train_feats, train_labels, val_feats, val_labels, test_res, n_cls, device, output_folder):

    classifier = LinearClassifier(train_feats.shape[1], n_cls)
    classifier.to(device)

    # == build the train/val/test dataset & loader ==
    train_dl = build_eval_ds_dl(
        train_feats,
        train_labels,
        shuffle=True,
        drop_last=True
    )

    # == build optim / sch ==
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.train_lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.train_epochs,
        eta_min=0
    )

    criterion = nn.CrossEntropyLoss()

    # == train/val loop ==
    best_val_acc = 0.0
    print('Start training linear classifier...')
    for epoch in range(args.train_epochs):

        classifier.train()

        epoch_gt = list()
        epoch_pred = list()
        for bs_id, bs in enumerate(train_dl):
            in_feats, labels = bs
            in_feats = in_feats.to(device)
            labels = labels.to(device)

            pred = classifier(in_feats)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_gt.append(labels.detach().cpu())
            epoch_pred.append(pred.detach().cpu())

            if bs_id % 20 == 0:
                print('Epoch: {:03d} | Batch: {:03d} | Loss: {:.4f} | LR: {:.4e}'.format(epoch, bs_id, loss.item(), optimizer.param_groups[0]['lr']))

        sch.step()

        epoch_gt = torch.cat(epoch_gt)
        epoch_pred = torch.cat(epoch_pred)
        epoch_loss = criterion(epoch_pred, epoch_gt).item()

        # val loop
        with torch.no_grad():
            classifier.eval()

            val_pred = classifier(val_feats.to(device)).argmax(dim=-1).cpu()
            val_acc = (val_pred == val_labels).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(classifier.state_dict(), os.path.join(output_folder, 'best_linear_classifier.pth'))
        
        epoch_str = f'Epoch {epoch} | EpochTrainLoss : {epoch_loss:.4f} | ValAcc: {val_acc:.4f}\n'
        print(epoch_str)
    
    # == test ==
    # load the best model
    classifier.load_state_dict(torch.load(os.path.join(output_folder, 'best_linear_classifier.pth'), map_location=device, weights_only=False))
    test_pred_dict = dict()
    with torch.no_grad():
        classifier.eval()

        for img_path in tqdm.tqdm(test_res.keys(), desc='Evaluating on test set'):
            test_feats = test_res[img_path]['fm_feats'].to(device)  # [n_det, feat_dim]
            if len(test_feats) == 0:
                test_pred_dict[img_path] = np.array([-1] * 0)  # No detections, return empty array
                continue

            pred = classifier(test_feats)
            test_pred_dict[img_path] = pred.argmax(dim=-1).cpu().numpy()

    return test_pred_dict


def infer_loop(meta_df, end2end_model, cls_infer_size, device, seed, use_gt_coords=False, use_det_res=False, det_res=None):

    infer_transfo2 = A.Compose([
        A.Resize(cls_infer_size, cls_infer_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.pytorch.ToTensorV2()
    ], p=1.0, seed=seed)

    res_dict = dict()

    # infer
    for idx in tqdm.tqdm(range(len(meta_df)), desc="Infer samples"):
        sample_info = meta_df.iloc[idx]
        sample_path = sample_info['sample_path']
        sample_data = np.load(sample_path, allow_pickle=True).item()  # Load the sample data
        img = sample_data['patch']  # [H, W, C]
        gt_coords = sample_data['ann'][:, :2]  # Get the coordinates (x, y)
        gt_tps = sample_data['ann'][:, 2]  # Get the types (tp)

        # apply transformations
        cls_sample = infer_transfo2(image=img)
        cls_img = cls_sample['image']
        cls_img = cls_img.unsqueeze(0).to(device)

        if use_gt_coords:
            # use the GT coordinates as the detection results
            det_coords = [gt_coords.astype(np.float32)]  # list of [n_gt, 2]
        elif use_det_res:
            # use the detection results from the detection experiment
            det_coords = [det_res[sample_path]['pred_coords'].astype(np.float32)]  # list of [n_det, 2]
        else:
            raise ValueError("Either use_gt_coords or use_det_res should be True to get the detection coordinates for the classification inference.")
        fm_feats, det_coords, det_scores = end2end_model(cls_img, cls_scale_ratio=cls_infer_size / img.shape[0], det_coords=det_coords)

        res_dict[sample_path] = {
            'fm_feats': fm_feats[0],  # [n_det, feat_dim]
            'det_coords': det_coords[0],
            'det_scores': det_scores[0],
            'gt_coords': gt_coords,
            'gt_tps': gt_tps,
        }

    return res_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Cls", parents=[get_args_parse()])
    args = parser.parse_args()

    cur_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = f'./outputs/{cur_time}_{args.exp_name}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f, default_flow_style=False)

    main(args)
