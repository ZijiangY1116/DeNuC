# Modification 2026 Zijiang Yang
# Copyright (c) Facebook, Inc. and its affiliates.
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
import sys
import tqdm
import random
import time
import torch
import datetime
import argparse
import subprocess
import numpy as np
from PIL import Image
import prettytable as pt
import scipy.spatial as S
import matplotlib.pyplot as plt
import torch.distributed as dist
from collections import defaultdict, deque


def bool_flag(s):
    """
    Parse boolean arguments from the command line.

    Copy from DINO: https://github.com/facebookresearch/dino/blob/main/utils.py
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def is_main_process():
    return get_rank() == 0


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(random.randint(29500, 29599))
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def fix_random_seeds(seed=0):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, log_file=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_str = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                else:
                    print_str = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))
                print(print_str)
                if is_main_process() and log_file is not None:
                    with open(log_file, "a") as f:
                        f.write(print_str + "\n")
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def point_nms(points, scores, nms_dist=-1, meth_type='native'):

    if len(points) == 0:
        # no points
        return points, scores

    _reserved = np.ones(len(points), dtype=bool)
    if meth_type == 'native':
        dis_matrix = S.distance_matrix(points, points)
        np.fill_diagonal(dis_matrix, np.inf)

        for idx in np.argsort(-scores):
            if _reserved[idx]:
                _reserved[dis_matrix[idx] <= nms_dist] = False
    elif meth_type == 'kdtree':
        from scipy.spatial import KDTree
        tree = KDTree(points)

        for idx in np.argsort(-scores):
            if _reserved[idx]:
                neighbors_idx = tree.query_ball_point(points[idx], r=nms_dist)
                for neighbor_idx in neighbors_idx:
                    if neighbor_idx != idx:
                        _reserved[neighbor_idx] = False

    points = points[_reserved]
    scores = scores[_reserved]
    return points, scores


def get_match(gt_points, pred_points, pred_scores, match_dist=12, return_index=False, return_matched_indices=False):
    """
    This function is used to match the gt & pred
    Args:
        gt_points: (N, 2)
        pred_points: (M, 2)
        pred_scores: (M, )
        match_dist: the threshold for matching
        return_index:
        return_matched_indices:
    """

    if gt_points.shape[0] == 0 or pred_points.shape[0] == 0:
        if return_index:
            if return_matched_indices:
                return 0, [], []
            else:
                return 0, []
        else:
            if return_matched_indices:
                return 0, []
            else:
                return 0

    # 1. sort the pred_points by scores
    sorted_pred_indices = np.argsort(-pred_scores)
    sorted_pred_points = pred_points[sorted_pred_indices]

    # 2. match
    unmatched = np.ones(len(gt_points), dtype=bool)
    matched_indices = np.full(len(pred_points), -1, dtype=int)
    dis = S.distance_matrix(sorted_pred_points, gt_points)

    for i in range(len(pred_points)):
        min_index = dis[i, unmatched].argmin()
        if dis[i, unmatched][min_index] <= match_dist:
            matched_indices[sorted_pred_indices[i]] = np.where(unmatched)[0][min_index]
            unmatched[np.where(unmatched)[0][min_index]] = False
        if not np.any(unmatched):
            break

    if return_index:
        if return_matched_indices:
            return sum(~unmatched), np.where(unmatched)[0], matched_indices
        else:
            return sum(~unmatched), np.where(unmatched)[0]
    else:
        if return_matched_indices:
            return sum(~unmatched), matched_indices
        else:
            return sum(~unmatched)


def multi_cls_eval(gt_points, gt_classes, pred_points, pred_scores, pred_classes, n_cls, match_dist=12, eps=1e-6, tps=None):
    """
    This function is used to evaluate the performance of the model.
    Args:
        gt_points: ground truth points, list(gt1, gt2, ...)
        gt_classes: ground truth classes, list(gt1, gt2, ...)
        pred_points: predicted points, list(pred1, pred2, ...)
        pred_scores: predicted scores, list(pred1, pred2, ...)
        pred_classes: predicted classes, list(pred1, pred2, ...)
        n_cls: number of classes, int
        match_dist: the threshold for matching, int
        eps: a small value to avoid division by zero, float
        tps: the class names, list(str)
    Return:
        dict:
            'cls1_recall': XXX,
            'cls1_precision': XXX,
            'cls1_f1': XXX,
            ...
            'total_recall': XXX,
            'total_precision': XXX,
            'total_f1': XXX
        table
    """

    # ========== init ==========
    n_samples = len(pred_scores)
    if tps is None:
        tps = [f'cls_{i}' for i in range(n_cls)]

    # record the number of predicted and true samples for each class
    n_cls_pred_list, n_cls_gt_list = np.zeros(n_cls), np.zeros(n_cls)
    # record the number of matched samples for each class
    n_cls_match_list = np.zeros(n_cls)

    n_det_pred, n_det_gt = 0, 0
    n_det_match = 0

    # ========== loop through all samples ==========
    tmp_iterator = tqdm.tqdm(range(n_samples), desc='Eval (X / X Samples)', leave=False, total=n_samples)
    for sample_id in tmp_iterator:

        tmp_gt_points = gt_points[sample_id]
        tmp_gt_classes = gt_classes[sample_id]
        tmp_pred_points = pred_points[sample_id]
        tmp_pred_scores = pred_scores[sample_id]
        tmp_pred_classes = pred_classes[sample_id]

        # 1. analysis the detection performance
        n_det_pred += tmp_pred_points.shape[0]
        n_det_gt += tmp_gt_points.shape[0]

        n_det_match += get_match(tmp_gt_points, tmp_pred_points, tmp_pred_scores, match_dist)

        # 2. analysis the classification performance
        # method: analyze each class separately
        for cls_id in range(n_cls):

            # get the points of this class (gt & pred)
            tmp_gt_points_cls = tmp_gt_points[tmp_gt_classes == cls_id]
            tmp_pred_points_cls = tmp_pred_points[tmp_pred_classes == cls_id]
            tmp_pred_scores_cls = tmp_pred_scores[tmp_pred_classes == cls_id]

            n_cls_gt_list[cls_id] += tmp_gt_points_cls.shape[0]
            n_cls_pred_list[cls_id] += tmp_pred_points_cls.shape[0]

            n_cls_match_list[cls_id] += get_match(tmp_gt_points_cls, tmp_pred_points_cls, tmp_pred_scores_cls, match_dist)
    
    # ========== compute metrics ==========
    det_recall = n_det_match * 100 / (n_det_gt + eps)
    det_precision = n_det_match * 100 / (n_det_pred + eps)
    det_f1 = (2 * det_recall * det_precision) / (det_recall + det_precision + eps)

    cls_recall_list = n_cls_match_list * 100 / (n_cls_gt_list + eps)
    cls_precision_list = n_cls_match_list * 100 / (n_cls_pred_list + eps)
    cls_f1_list = (2 * cls_recall_list * cls_precision_list) / (cls_recall_list + cls_precision_list + eps)
    cls_recall = np.mean(cls_recall_list)
    cls_precision = np.mean(cls_precision_list)
    cls_f1 = np.mean(cls_f1_list)

    # ========== build the result dict ==========
    res_dict = {
        'det_recall': det_recall,
        'det_precision': det_precision,
        'det_f1': det_f1,
        'cls_recall': cls_recall,
        'cls_precision': cls_precision,
        'cls_f1': cls_f1,
    }
    for cls_id in range(n_cls):
        cls_name = tps[cls_id]
        res_dict[f'cls_{cls_name}_recall'] = cls_recall_list[cls_id]
        res_dict[f'cls_{cls_name}_precision'] = cls_precision_list[cls_id]
        res_dict[f'cls_{cls_name}_f1'] = cls_f1_list[cls_id]

    res_table = pt.PrettyTable()
    res_table.add_column('CLASS', tps)
    res_table.add_column('Precision', cls_precision_list.round(2))
    res_table.add_column('Recall', cls_recall_list.round(2))
    res_table.add_column('F1', cls_f1_list.round(2))
    res_table.add_row(['---'] * 4)
    res_table.add_row(['---'] * 4)
    res_table.add_row(['Det', np.round(det_precision, 2), np.round(det_recall, 2), np.round(det_f1, 2)])
    res_table.add_row(['Cls', np.round(cls_precision, 2), np.round(cls_recall, 2), np.round(cls_f1, 2)])
    
    return res_dict, res_table
    

def vis_dets(img, point_group1, point_group2=None, point_size=5, save_path=None, group1_name='Group 1', group2_name='Group 2'):
    """
    Visualize points on the image.
    Args:
        img: numpy array H x W x C
        point_group1: numpy array N x 2
        point_group2: numpy array M x 2
        point_size: int
        save_path: str
        group1_name: str
        group2_name: str
    
    if point_group2 is None, only visualize point_group1 in red color.
    else, visualize with subplots.
    """
    if point_group2 is None:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.scatter(point_group1[:, 0], point_group1[:, 1], s=point_size, c='r', marker='o')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    else:
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        if len(point_group1) > 0:
            # special case for no points
            plt.scatter(point_group1[:, 0], point_group1[:, 1], s=point_size, c='r', marker='o')
        plt.title(group1_name)

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        if len(point_group2) > 0:
            # special case for no points
            plt.scatter(point_group2[:, 0], point_group2[:, 1], s=point_size, c='b', marker='^')
        plt.title(group2_name)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


def merge_patches(batch_local_coords, batch_scores, batch_offsets, img_size, win_size, overlap_size):
    """
    Merge predictions from overlapping patches.
    
    Args:
        batch_local_coords: [B, N, 2] local coordinates in each patch
        batch_scores: [B, N] confidence scores for each prediction
        batch_offsets: [B, 2] top-left corner offsets of each patch in the original image
        img_size: (H, W) size of the original image
        win_size: window size of each patch
        overlap_size: overlap size used in sliding window cropping

    Returns:
        global_points: [M, 2] merged global coordinates in the original image
        global_scores: [M] merged confidence scores
    """

    margin = overlap_size // 2  # define margin as half of the overlap
    img_h, img_w = img_size
    win_h, win_w = (win_size, win_size) if isinstance(win_size, int) else win_size

    global_points = []
    global_scores = []

    for i in range(len(batch_local_coords)):
        local_coords = batch_local_coords[i]  # [N, 2]
        scores = batch_scores[i]  # [N]
        off_x, off_y = batch_offsets[i]

        # define ROI
        min_x, min_y = margin, margin
        max_x, max_y = win_w - margin, win_h - margin
        if off_x == 0: min_x = 0
        if off_y == 0: min_y = 0
        if off_x + win_w >= img_w: max_x = win_w
        if off_y + win_h >= img_h: max_y = win_h

        # filter
        mask_x = (local_coords[:, 0] >= min_x) & (local_coords[:, 0] < max_x)
        mask_y = (local_coords[:, 1] >= min_y) & (local_coords[:, 1] < max_y)
        valid_mask = mask_x & mask_y

        # get valid points
        valid_local_coords = local_coords[valid_mask]
        valid_scores = scores[valid_mask]

        # convert to global coords
        if len(valid_local_coords) > 0:
            valid_global_coords = valid_local_coords + np.array([off_x, off_y])
            global_points.append(valid_global_coords)
            global_scores.append(valid_scores)
    
    # merge and return
    if len(global_points) > 0:
        global_points = np.concatenate(global_points, axis=0)
        global_scores = np.concatenate(global_scores, axis=0)
    else:
        global_points = np.zeros((0, 2))
        global_scores = np.zeros((0,))
    return global_points, global_scores


def sliding_window_crop(image, crop_size, overlap_ratio=0.25):
    """
    Sliding window crop for image.

    Args:
        image:numpy array H x W x C
        crop_size: int or tuple
        overlap_ratio: float, between 0 and 1
    Returns:
        img_list: list of cropped images
        coord_list: list of (x0, y0) for each cropped image
    """
    # compute strides
    assert type(image) is np.ndarray, "Only support numpy array."
    orig_h, orig_w = image.shape[:2]
    if type(crop_size) is int:
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size
    
    stride_h = int(crop_h * (1 - overlap_ratio))
    stride_w = int(crop_w * (1 - overlap_ratio))
    stride_h = max(stride_h, 1)
    stride_w = max(stride_w, 1)

    # begin crop
    img_list = list()
    coord_list = list()
    
    # loop over y
    y = 0
    while y < orig_h:
        # make sure the last patch reaches the image border
        if y + crop_h > orig_h:
            y = orig_h - crop_h
            if y < 0: y = 0
        
        # loop over x
        x = 0
        while x < orig_w:
            # make sure the last patch reaches the image border
            if x + crop_w > orig_w:
                x = orig_w - crop_w
                if x < 0: x = 0
            
            # crop
            if y+crop_h <= orig_h and x+crop_w <= orig_w:
                img_crop = image[y:y+crop_h, x:x+crop_w]
            else:
                # the crop size is larger than the image size
                # pad the image (fill value is 255, white, for histopathology images)
                img_crop = np.ones((crop_h, crop_w, image.shape[2]), dtype=image.dtype) * 255
                img_crop[:min(crop_h, orig_h - y), :min(crop_w, orig_w - x)] = image[y:y+crop_h, x:x+crop_w]
            img_list.append(img_crop)
            coord_list.append((x, y))

            if x + crop_w >= orig_w:
                break
            x += stride_w
        if y + crop_h >= orig_h:
            break
        y += stride_h
    return img_list, coord_list
