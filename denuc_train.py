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
import cv2
import sys
import yaml
import json
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

import utils.data as data
from utils.model import build_arch
import utils.basic_utils as basic_utils


def get_args_parse():

    parser = argparse.ArgumentParser("NuTo", add_help=False)

    # basic parameters
    parser.add_argument('--exp_name', default='denuc_experiment', type=str, help='Name of the experiment.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', default='./output', type=str, 
                        help='Directory to save outputs (logs, checkpoints, etc.). ' \
                        'Please note that the output_dir will be automatically modified to include ' \
                        'a timestamp and experiment name to avoid overwriting previous experiments. ' \
                        'Therefore, do not use this argument.')
    parser.add_argument("--dist_url", default="env://", type=str, 
                        help="url used to set up " \
                        "distributed training; see https://pytorch.org/docs/stable/distributed.html")
    
    # model parameters
    parser.add_argument('--arch', default='denuc_det_shufflenet_x1_0', type=str, 
                        help='Backbone model to use. As the nuclei detection is a relatively simple task, ' \
                            'we recommend using a small backbone like ShuffleNet to save computation resources during large-scale inference.')

    # dataset parameters
    parser.add_argument('--datasets', type=str, default="puma,brcam2c,ocelot", help='Comma separated list of datasets to use for training.')

    # training / optimization parameters
    parser.add_argument('--use_fp16', type=basic_utils.bool_flag, default=False, help='Whether to use mixed precision training to save GPU memory.')
    parser.add_argument('--batch_size_per_gpu', default=32, type=int, help='Batch size per GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs.')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer.')
    parser.add_argument('--optimizer', default='adamw', type=str, help='Optimizer to use: adamw or sgd.')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate for cosine scheduler.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--save_freq', default=1, type=int, help='Saving frequency (in epochs).')
    parser.add_argument('--resume', type=basic_utils.bool_flag, default=False, help='Whether to resume from the latest checkpoint in the output directory.')
    parser.add_argument('--resume_exp_name', default='', type=str, help='Experiment name to resume from.')
    parser.add_argument('--img_size', default=256, type=int, help='Input image size for training. The input patches will be resized to (img_size, img_size) during training.')
    parser.add_argument('--reg_coef', default=5e-3, type=float, help='Coefficient for the regression loss.')
    parser.add_argument('--reg_type', default='l2', type=str, help='Type of regression loss: l1 or l2.')
    parser.add_argument('--cls_coef', default=1.0, type=float, help='Coefficient for the classification loss.')
    parser.add_argument('--cls_type', default='ce', type=str, help='Type of classification loss: ce (cross-entropy).')
    parser.add_argument('--match_topk', default=1, type=int, help='Top-k predictions will be matched to each ground truth during training.')
    parser.add_argument('--bg_point_coef', default=0.5, type=float, help='Coefficient for background points in classification loss. Use -1 for automatic balancing.')

    # eval parameters
    parser.add_argument('--eval_nms', default=12.0, type=float, help='NMS distance threshold (in pixels) for evaluation.')

    return parser


def train_denuc(args):
    """
    Main function to train DeNuC.
    """
    basic_utils.init_distributed_mode(args)
    basic_utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(basic_utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # =================
    # Build Dataset
    # =================
    # 1. build the metadata
    meta_dict = {
        'sample_path': list(),
        'split': list()
    }
    for dataset_name in args.datasets.split(','):
        dataset_name = dataset_name.strip()
        if dataset_name in ['puma', 'brcam2c', 'ocelot']:
            dataset_meta = pd.read_csv(f'./dataset/{dataset_name}/meta.csv')
            for _, row in dataset_meta.iterrows():
                meta_dict['sample_path'].append(os.path.join(f'./dataset/{dataset_name}', f"{row['sample_name']}.npy"))
                meta_dict['split'].append(row['split'])
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    
    meta_df = pd.DataFrame(meta_dict)
    # build the train meta
    train_meta_df = meta_df[meta_df['split'] == 'train'].copy().reset_index(drop=True)
    train_meta_path = os.path.join(args.output_dir, 'train_meta.csv')
    train_meta_df.to_csv(train_meta_path, index=False)

    # 2. build the dataset and dataloader
    train_transform = data.TrainAugmentation(target_size=args.img_size)
    train_dataset = data.DeNuCDataset(
        metadata=train_meta_df,
        transform=train_transform
    )
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    print(f"Dataset built: {len(train_dataset)} training samples.")

    # 3. random visualization
    if basic_utils.is_main_process():
        vis_dir = os.path.join(args.output_dir, 'train_sample_vis')
        os.makedirs(vis_dir, exist_ok=True)
        for i in np.random.choice(len(train_dataset), size=6, replace=False):
            img, coords = train_dataset[i]
            # denormalize
            img = img.numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            # draw
            for coord in coords:
                x, y = int(coord[0]), int(coord[1])
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(vis_dir, f'sample_{i}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # =================
    # Build Model
    # =================
    model_net = build_arch.__all__[args.arch]()
    
    # move to gpu
    model_net = model_net.cuda()
    model_net = nn.SyncBatchNorm.convert_sync_batchnorm(model_net)
    model_net = nn.parallel.DistributedDataParallel(model_net, device_ids=[args.gpu])
    print("Model built.")

    # =================
    # Build Optimizer and Scheduler
    # =================
    denuc_loss = DeNuCLoss(
        reg_coef=args.reg_coef,
        cls_coef=args.cls_coef,
        reg_type=args.reg_type,
        cls_type=args.cls_type,
        bg_point_coef=args.bg_point_coef,
        match_topk=args.match_topk,
    ).cuda()

    params_group = basic_utils.get_params_groups(model_net)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer '{args.optimizer}' is not supported.")
    
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.amp.GradScaler(enabled=args.use_fp16)

    lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    print("Optimizer and scheduler built.")

    # =================
    # Try to resume
    # =================
    # try to resume from the latest checkpoint
    to_restore = {"epoch": 0}
    basic_utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model_net,
        optimizer=optimizer,
        lr_scheduler=lr_sch,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    # =================
    # Training Loop
    # =================
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train_one_epoch(model_net, denuc_loss, train_dataloader, optimizer, lr_sch, epoch, fp16_scaler, args)

        # build the save dict
        save_dict = {
            'model': model_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_sch.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        
        # save
        basic_utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if epoch % args.save_freq == 0:
            basic_utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04d}.pth'))
        
        # logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if basic_utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if basic_utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f'Training time {total_time_str}\n')
    
    # cleanup
    dist.destroy_process_group()


def train_one_epoch(model, criterion, data_loader, optimizer, lr_scheduler, epoch, fp16_scaler, args):
    
    metric_logger = basic_utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    log_file = os.path.join(args.output_dir, "log-step.txt")
    for it, (images, gt_coords, gt_masks) in enumerate(metric_logger.log_every(data_loader, 10, header, log_file)):

        # get batch data
        images = images.cuda(non_blocking=True)
        gt_coords = gt_coords.cuda(non_blocking=True)
        gt_masks = gt_masks.cuda(non_blocking=True)
     
        gt = {
            'gt_coords': gt_coords,
            'gt_labels': torch.zeros(gt_coords.shape[:2], dtype=torch.long, device=gt_coords.device),  # all is nuclei
            'gt_masks': gt_masks
        }

        # forward
        with torch.amp.autocast(device_type='cuda', enabled=fp16_scaler is not None):
            # forward_begin_time = time.time()
            preds = model(images)
            # forward_time = time.time() - forward_begin_time
            losses = criterion(preds, gt)
            loss = sum(losses.values())
            # loss_time = time.time() - forward_begin_time - forward_time
            # print(f"Iter {it}: forward time {forward_time:.4f}s, loss time {loss_time:.4f}s, n_gt {gt_masks.sum().item()}, n_pred {preds['pred_coords'].shape[1] * images.shape[0]}")
        
        if not torch.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        # backward
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        
        # logging
        torch.cuda.synchronize()
        metric_logger.update(reg_loss=losses['reg'].item())
        metric_logger.update(cls_loss=losses['cls'].item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # update lr
    lr_scheduler.step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DeNuCLoss(nn.Module):

    def __init__(self, reg_coef, cls_coef, reg_type='l2', cls_type='ce', bg_point_coef=1.0, match_topk=4):
        super().__init__()
        self.matcher = Matcher(top_k=match_topk)
        self.n_cls = 1

        self.reg_coef = reg_coef
        self.cls_coef = cls_coef

        self.reg_type = reg_type
        self.cls_type = cls_type

        # the coefficient for background points in classification loss
        # -1: special value, auto define based on num positive / num negative points in a batch
        self.bg_point_coef = bg_point_coef
    
    def reg_loss_fn(self, pred_coords, gt_coords):
        if self.reg_type == 'l1':
            loss = F.l1_loss(pred_coords, gt_coords, reduction='mean')
        elif self.reg_type == 'l2':
            loss = F.mse_loss(pred_coords, gt_coords, reduction='mean')
        else:
            raise ValueError(f"Regression loss type '{self.reg_type}' is not supported.")
        return loss
    
    def cls_loss_fn(self, pred_logits, gt_labels):
        if self.cls_type == 'ce':
            if self.bg_point_coef == 1.0:
                loss = F.cross_entropy(pred_logits, gt_labels, reduction='mean')
            else:
                # weighted cross-entropy loss
                weight = torch.ones(self.n_cls + 1, device=pred_logits.device)
                if self.bg_point_coef == -1:
                    # auto define bg_point_coef
                    # balance
                    n_pos = (gt_labels < self.n_cls).sum().item()
                    n_neg = (gt_labels == self.n_cls).sum().item()
                    if n_neg == 0:
                        bg_point_coef = 1.0
                    else:
                        bg_point_coef = n_pos / n_neg
                    weight[-1] = bg_point_coef
                else:
                    weight[-1] = self.bg_point_coef
                loss = F.cross_entropy(pred_logits, gt_labels, reduction='mean', weight=weight)
        else:
            raise ValueError(f"Classification loss type '{self.cls_type}' is not supported.")
        return loss
    
    def forward(self, preds, gt):
        matched_topk_idx = self.matcher(preds, gt)  # [B, k, N_padded]

        bs, top_k, n_gt_padded = matched_topk_idx.shape
        device = preds['pred_coords'].device

        # =========================
        # for reg loss
        # the reg_loss only apply to the matched nuclei
        # =========================
        batch_idx = torch.arange(bs, device=device).unsqueeze(1).unsqueeze(2).expand(-1, top_k, n_gt_padded)  # [B, k, N_padded]
        matched_pred_coords = preds['pred_coords'][batch_idx, matched_topk_idx]  # [B, k, N_padded, 2]
        target_coords = gt['gt_coords'].unsqueeze(1).expand(bs, top_k, n_gt_padded, 2)  # [B, k, N_padded, 2]
        valid_masks = gt['gt_masks'].unsqueeze(1).expand(bs, top_k, n_gt_padded)  # [B, k, N_padded]

        valid_pred_coords = matched_pred_coords[valid_masks]  # [N_valid, 2]
        valid_target_coords = target_coords[valid_masks]  # [N_valid, 2]

        reg_loss = self.reg_loss_fn(valid_pred_coords, valid_target_coords) * self.reg_coef

        # =========================
        # for cls loss
        # the cls_loss apply to all proposals
        # =========================
        target_labels = torch.full(preds['pred_logits'].shape[:2], self.n_cls, dtype=torch.long, device=device)  # [B, N]
        valid_batch_idx = batch_idx[valid_masks]  # [N_valid]
        valid_query_idx = matched_topk_idx[valid_masks]  # [N_valid]

        target_labels[valid_batch_idx, valid_query_idx] = 0  # positive samples
        cls_loss = self.cls_loss_fn(preds['pred_logits'].flatten(0, 1), target_labels.flatten(0, 1)) * self.cls_coef

        return {'reg': reg_loss, 'cls': cls_loss}

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


class Matcher(nn.Module):
    """
    In p2p method, we should get the matching relationship between predictions and ground truths.

    Generally, the Hungarian algorithm is used to get a one(pred)-to-one(gt) matching.
    However, in nuclei detection, as the shape and size across different nuclei are similar,
    we can relax the one-to-one matching to k(pred)-to-one(gt) matching to improve the training stability and performance.
    """

    def __init__(self, cost_point=0.1, cost_class=1.0, top_k=4):
        super(Matcher, self).__init__()
        self.cost_point = cost_point
        self.cost_class = cost_class
        self.top_k = top_k

    @torch.no_grad()
    def forward(self, preds, gt):

        # get the prediction
        pred_coords = preds['pred_coords']  # [B, N, 2]
        pred_probs = preds['pred_logits'].softmax(-1)  # [B, N, num_classes]

        # get the ground truth
        gt_coords = gt['gt_coords']  # [B, N_padded, 2]
        gt_labels = gt['gt_labels']  # [B, N_padded]
        gt_masks = gt['gt_masks']  # [B, N_padded]

        # basic dimensions
        bs, n_queries = pred_coords.shape[:2]
        n_gt_padded = gt_coords.shape[1]

        # compute the cost matrices
        cost_point = torch.cdist(pred_coords, gt_coords, p=2)  # [B, N, N_padded]
        gt_labels_expanded = gt_labels.unsqueeze(1).expand(-1, n_queries, -1)  # [B, N, N_padded]
        cost_class = -torch.gather(pred_probs, 2, gt_labels_expanded)  # [B, N, N_padded]

        C = self.cost_point * cost_point + self.cost_class * cost_class  # [B, N, N_padded]

        # apply gt masks
        gt_masks_expanded = gt_masks.unsqueeze(1).expand(-1, n_queries, -1)  # [B, N, N_padded]
        C = C.masked_fill(~gt_masks_expanded, float('inf'))

        # get the top-k indices
        _, topk_idx = torch.topk(C, self.top_k, dim=-2, largest=False)  # [B, k, N_padded]

        return topk_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DeNuC", parents=[get_args_parse()])
    args = parser.parse_args()

    # build output_dir
    if args.resume and args.resume_exp_name != '':
        args.output_dir = os.path.join('./outputs', args.resume_exp_name)
    else:
        cur_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'./outputs/{cur_time}_{args.exp_name}'

    if basic_utils.is_main_process():
        # only the main process create the output dir
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f, default_flow_style=False)

    train_denuc(args)

    # evaluation on validation set
    if basic_utils.is_main_process():
        print("Start evaluation on val/test sets...")
        # eval all checkpoints on validation data
        print("---------------------------------------------------")
        print("Evaluating on validation set...")
        os.system(f'python ./denuc_eval.py --exp_name {cur_time}_{args.exp_name} --eval_dataset {args.datasets} --eval_mode val --nms_dist {args.eval_nms}')
        # eval the best checkpoint on test data
        print("---------------------------------------------------")
        print("Evaluating on test set...")
        for dataset_name in args.datasets.split(','):
            os.system(f'python ./denuc_eval.py --exp_name {cur_time}_{args.exp_name} --eval_dataset {dataset_name.strip()} --eval_mode test --nms_dist {args.eval_nms}')

    print("Training and evaluation completed.")
    print(f"All outputs are saved to {args.output_dir}.")
