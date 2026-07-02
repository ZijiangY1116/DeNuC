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

import argparse
import glob
import os
import openslide
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
import yaml
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from utils.model import build_arch


TARGET_MPP = 0.25
WINDOW_SIZE = 256
OVERLAP_RATIO = 0.25
SCORE_THRESHOLD = 0.5
NMS_DISTANCE = 12.0
PREDET_TARGET_SIZES = (None, 2048, 512, 4096, 256)
WSI_EXTENSIONS = {".svs", ".ndpi", ".mrxs", ".scn"}

Image.MAX_IMAGE_PIXELS = None


class RasterImageSource:
    """
    Read and scale a raster image file (JPEG, PNG, TIFF, etc.) using Pillow.
    This is a fallback for non-WSI files that OpenSlide cannot read.
    """
    def __init__(self, path):
        self.path = path
        self.image = Image.open(path)
        self.width, self.height = self.image.size
        self.mpp = None

    def read_scaled_region(self, x, y, width, height, scale):
        box = (x / scale, y / scale, (x + width) / scale, (y + height) / scale)
        region = self.image.crop(box).convert("RGB")
        if region.size != (width, height):
            region = region.resize((width, height), Image.Resampling.BICUBIC)
        output = Image.new("RGB", (width, height), (255, 255, 255))
        valid_width = min(width, max(0, int(np.ceil((self.width * scale) - x))))
        valid_height = min(height, max(0, int(np.ceil((self.height * scale) - y))))
        if valid_width > 0 and valid_height > 0:
            output.paste(region.crop((0, 0, valid_width, valid_height)), (0, 0))
        return np.asarray(output, dtype=np.uint8)

    def close(self):
        self.image.close()


class OpenSlideImageSource:
    """
    Read and scale a WSI file using OpenSlide.
    """
    def __init__(self, path):
        self.path = path
        self.slide = openslide.OpenSlide(path)
        self.width, self.height = self.slide.dimensions
        self.mpp = self._read_mpp()

    def _read_mpp(self):
        properties = self.slide.properties
        candidates = (
            properties.get("openslide.mpp-x"),
            properties.get("aperio.MPP"),
        )
        for value in candidates:
            try:
                mpp = float(value)
                if mpp > 0:
                    return mpp
            except (TypeError, ValueError):
                pass
        return None

    def read_scaled_region(self, x, y, width, height, scale):
        downsample = 1.0 / scale
        level = self.slide.get_best_level_for_downsample(downsample)
        level_downsample = float(self.slide.level_downsamples[level])
        level_width = max(1, int(np.ceil(width / (scale * level_downsample))))
        level_height = max(1, int(np.ceil(height / (scale * level_downsample))))
        location = (int(np.floor(x / scale)), int(np.floor(y / scale)))
        region = self.slide.read_region(location, level, (level_width, level_height))
        background = Image.new("RGBA", region.size, (255, 255, 255, 255))
        background.alpha_composite(region)
        region = background.convert("RGB")
        if region.size != (width, height):
            region = region.resize((width, height), Image.Resampling.BICUBIC)
        return np.asarray(region, dtype=np.uint8)

    def close(self):
        self.slide.close()


class ArrayImageSource:
    """Small in-memory source used for pre-detection."""

    def __init__(self, image):
        self.image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
        self.width, self.height = self.image.size
        self.mpp = None

    def read_scaled_region(self, x, y, width, height, scale):
        box = (x / scale, y / scale, (x + width) / scale, (y + height) / scale)
        region = self.image.crop(box)
        if region.size != (width, height):
            region = region.resize((width, height), Image.Resampling.BICUBIC)
        output = Image.new("RGB", (width, height), (255, 255, 255))
        valid_width = min(width, max(0, int(np.ceil((self.width * scale) - x))))
        valid_height = min(height, max(0, int(np.ceil((self.height * scale) - y))))
        if valid_width > 0 and valid_height > 0:
            output.paste(region.crop((0, 0, valid_width, valid_height)), (0, 0))
        return np.asarray(output, dtype=np.uint8)

    def close(self):
        self.image.close()


def open_image_source(path):
    """
    Open a file-backed image source, using OpenSlide for WSI files and Pillow for other raster images.
    """
    extension = Path(path).suffix.lower()
    if extension in WSI_EXTENSIONS:
        return OpenSlideImageSource(path)
    try:
        return RasterImageSource(path)
    except Exception:
        # Some TIFF-family WSI files need OpenSlide even without a dedicated suffix.
        if extension in {".tif", ".tiff"}:
            return OpenSlideImageSource(path)
        raise


def resolve_checkpoint(model_path):
    """
    Resolve a model checkpoint path from a file or directory.
    """
    model_path = os.path.abspath(model_path)
    if os.path.isfile(model_path):
        return model_path
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    best_path = os.path.join(model_path, "best_checkpoint.pth")
    if os.path.isfile(best_path):
        return best_path
    checkpoints = sorted(glob.glob(os.path.join(model_path, "checkpoint*.pth")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint*.pth found in: {model_path}")
    return checkpoints[-1]


def load_model(model_path, device):
    """
    Try to load a model checkpoint and return the model, input size, checkpoint path, and architecture name.
    """
    checkpoint_path = resolve_checkpoint(model_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_args = checkpoint.get("args")
    if checkpoint_args is not None:
        if isinstance(checkpoint_args, dict):
            arch = checkpoint_args.get("arch")
            model_size = int(checkpoint_args.get("img_size", WINDOW_SIZE))
        else:
            arch = getattr(checkpoint_args, "arch", None)
            model_size = int(getattr(checkpoint_args, "img_size", WINDOW_SIZE))
    else:
        args_path = os.path.join(os.path.dirname(checkpoint_path), "args.yaml")
        if not os.path.isfile(args_path):
            raise RuntimeError("Checkpoint has no training args and args.yaml was not found.")
        training_args = yaml.safe_load(open(args_path, "r"))
        arch = training_args["arch"]
        model_size = int(training_args.get("img_size", WINDOW_SIZE))

    if arch not in build_arch.__all__:
        raise ValueError(f"Unsupported architecture in checkpoint: {arch}")
    model = build_arch.__all__[arch]()
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, model_size, checkpoint_path, arch


def find_largest_batch_size(probe, start=32, maximum=256, on_oom=None):
    """Find the largest power-of-two batch accepted by a probe callback."""
    candidate = min(start, maximum)
    while candidate >= 1:
        try:
            probe(candidate)
            break
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if on_oom is not None:
                on_oom()
            candidate //= 2
    if candidate < 1:
        raise RuntimeError("Unable to find a usable inference batch size.")

    best = candidate
    candidate *= 2
    while candidate <= maximum:
        try:
            probe(candidate)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if on_oom is not None:
                on_oom()
            break
        best = candidate
        candidate *= 2
    return best


def autotune_batch_size(model, model_size, device):
    """
    Automatically select the largest inference batch size that fits in GPU memory, if using CUDA.
    """
    if device.type != "cuda":
        return 1

    def probe(batch_size):
        inputs = torch.empty(
            (batch_size, 3, model_size, model_size),
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode():
            outputs = model(inputs)
            # Materialize all outputs before releasing the probe tensors.
            _ = outputs["pred_coords"].shape, outputs["pred_logits"].shape
        torch.cuda.synchronize(device)
        del inputs, outputs

    try:
        batch_size = find_largest_batch_size(
            probe, on_oom=lambda: torch.cuda.empty_cache()
        )
    finally:
        torch.cuda.empty_cache()
    print(f"Automatically selected inference batch size: {batch_size}")
    return batch_size


def window_positions(length, window=WINDOW_SIZE, overlap=OVERLAP_RATIO):
    """
    Compute the starting positions of sliding windows along a single dimension, given the total length,
    window size, and overlap ratio.
    """
    stride = max(1, int(window * (1.0 - overlap)))
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


def point_nms(points, scores, nms_distance=NMS_DISTANCE):
    """
    Perform non-maximum suppression on a set of points based on their scores.
    """
    if len(points) == 0:
        return points, scores
    reserved = np.ones(len(points), dtype=bool)
    tree = cKDTree(points)
    for index in np.argsort(-scores):
        if reserved[index]:
            neighbors = tree.query_ball_point(points[index], r=nms_distance)
            reserved[np.asarray(neighbors, dtype=int)] = False
            reserved[index] = True
    return points[reserved], scores[reserved]


def normalize_patch(image, model_size):
    """
    Normalize a single image patch to the model's expected input format, including resizing, converting to tensor,
    and applying mean/std normalization.
    """
    pil_image = Image.fromarray(image)
    if pil_image.size != (model_size, model_size):
        pil_image = pil_image.resize((model_size, model_size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(pil_image, dtype=np.float32).copy())
    tensor = tensor.permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def normalize_batch(images, model_size):
    return torch.stack([normalize_patch(image, model_size) for image in images])


class SlidingWindowDataset(Dataset):
    """Read and normalize windows in independent DataLoader workers."""

    def __init__(self, source_path, offsets, scale, model_size):
        self.source_path = source_path
        self.offsets = offsets
        self.scale = scale
        self.model_size = model_size
        self._source = None

    def __len__(self):
        return len(self.offsets)

    def _get_source(self):
        if self._source is None:
            self._source = open_image_source(self.source_path)
        return self._source

    def __getitem__(self, index):
        off_x, off_y = self.offsets[index]
        patch = self._get_source().read_scaled_region(
            off_x, off_y, WINDOW_SIZE, WINDOW_SIZE, self.scale
        )
        return normalize_patch(patch, self.model_size), np.asarray([off_x, off_y], dtype=np.int64)

    def __del__(self):
        if self._source is not None:
            self._source.close()


def initialize_data_worker(_worker_id):
    # Avoid CPU thread oversubscription across multiple reader workers.
    torch.set_num_threads(1)


def patch_batches(source, offsets, scale, model_size, batch_size):
    """Yield prefetched batches for file-backed sources, with an in-memory fallback."""
    source_path = getattr(source, "path", None)
    if source_path and os.path.isfile(source_path):
        if len(offsets) < 128:
            num_workers = 0
        else:
            num_workers = min(
                8,
                os.cpu_count() or 1,
                max(2, len(offsets) // 256),
            )
        dataset = SlidingWindowDataset(source_path, offsets, scale, model_size)
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
                worker_init_fn=initialize_data_worker,
            )
        loader = DataLoader(**loader_kwargs)
        yield from loader
        return

    for start in range(0, len(offsets), batch_size):
        batch_offsets = offsets[start:start + batch_size]
        patches = [
            source.read_scaled_region(x, y, WINDOW_SIZE, WINDOW_SIZE, scale)
            for x, y in batch_offsets
        ]
        yield normalize_batch(patches, model_size), torch.as_tensor(batch_offsets, dtype=torch.int64)


def filter_batch_predictions(
    predictions,
    batch_offsets,
    scaled_width,
    scaled_height,
    model_size,
):
    """Filter scores/overlap margins and apply offsets before copying to CPU."""
    pred_coords = predictions["pred_coords"]
    pred_scores = predictions["pred_logits"].softmax(dim=-1)[..., 0]
    coordinate_ratio = WINDOW_SIZE / model_size
    pred_coords = pred_coords * coordinate_ratio

    offsets = batch_offsets.to(device=pred_coords.device, dtype=pred_coords.dtype, non_blocking=True)
    off_x = offsets[:, 0]
    off_y = offsets[:, 1]
    margin = int(WINDOW_SIZE * OVERLAP_RATIO) // 2

    min_x = torch.where(off_x == 0, 0.0, float(margin))
    min_y = torch.where(off_y == 0, 0.0, float(margin))
    max_x = torch.where(
        off_x + WINDOW_SIZE >= scaled_width,
        float(WINDOW_SIZE),
        float(WINDOW_SIZE - margin),
    )
    max_y = torch.where(
        off_y + WINDOW_SIZE >= scaled_height,
        float(WINDOW_SIZE),
        float(WINDOW_SIZE - margin),
    )
    valid = (
        (pred_scores > SCORE_THRESHOLD)
        & (pred_coords[..., 0] >= min_x[:, None])
        & (pred_coords[..., 0] < max_x[:, None])
        & (pred_coords[..., 1] >= min_y[:, None])
        & (pred_coords[..., 1] < max_y[:, None])
    )
    global_coords = pred_coords + offsets[:, None, :]
    return global_coords[valid].detach().cpu().numpy(), pred_scores[valid].detach().cpu().numpy()


@torch.inference_mode()
def infer_scaled_image(source, scale, model, model_size, device, description, batch_size):
    """
    The main sliding-window inference loop. Returns coordinates and scores in the scaled image space.
    """
    scaled_width = max(1, int(round(source.width * scale)))
    scaled_height = max(1, int(round(source.height * scale)))
    xs = window_positions(scaled_width)
    ys = window_positions(scaled_height)
    offsets = [(x, y) for y in ys for x in xs]
    all_coords = []
    all_scores = []

    total_batches = int(np.ceil(len(offsets) / batch_size))
    batches = patch_batches(source, offsets, scale, model_size, batch_size)
    progress = tqdm.tqdm(batches, total=total_batches, desc=description)
    for inputs, batch_offsets_tensor in progress:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        predictions = model(inputs)
        batch_coords, batch_scores = filter_batch_predictions(
            predictions,
            batch_offsets_tensor,
            scaled_width,
            scaled_height,
            model_size,
        )
        if len(batch_coords):
            all_coords.append(batch_coords)
            all_scores.append(batch_scores)

    if all_coords:
        coords = np.concatenate(all_coords).astype(np.float32)
        scores = np.concatenate(all_scores).astype(np.float32)
    else:
        coords = np.empty((0, 2), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
    coords, scores = point_nms(coords, scores)
    return coords.astype(np.float32), scores.astype(np.float32)


def read_center_crop(source):
    """
    Read a center crop of the source image, up to 1024x1024 pixels.
    """
    crop_width = min(1024, source.width)
    crop_height = min(1024, source.height)
    x = max(0, (source.width - crop_width) // 2)
    y = max(0, (source.height - crop_height) // 2)
    image = source.read_scaled_region(x, y, crop_width, crop_height, scale=1.0)
    return image, (x, y)


def select_unknown_scale(source, model, model_size, device, batch_size):
    """
    Try multiple target scales for pre-detection and select the one with the most detections (the best).
    """
    center_image, _ = read_center_crop(source)
    pre_source = ArrayImageSource(center_image)
    longest_side = max(pre_source.width, pre_source.height)
    candidates = []
    try:
        for target_size in PREDET_TARGET_SIZES:
            scale = 1.0 if target_size is None else target_size / longest_side
            coords, _ = infer_scaled_image(
                pre_source, scale, model, model_size, device,
                description=f"Pre-det scale {scale:.4g}x",
                batch_size=batch_size,
            )
            candidates.append((len(coords), scale, target_size))
            label = "original" if target_size is None else f"max-side={target_size}"
            print(f"[pre-det] {label}: {len(coords)} detections")
    finally:
        pre_source.close()

    # Stable tie-breaking follows the requested candidate order.
    best_index = max(range(len(candidates)), key=lambda index: candidates[index][0])
    count, scale, target_size = candidates[best_index]
    label = "original" if target_size is None else f"max-side={target_size}"
    print(f"[pre-det] selected {label} ({scale:.6g}x, {count} detections)")
    return scale, label, candidates


def output_h5_path(output_path, input_path):
    """
    Determine the output H5 file path based on the provided output path and input image path.
    """
    if output_path.lower().endswith((".h5", ".hdf5")):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        return os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)
    return os.path.abspath(os.path.join(output_path, f"{Path(input_path).stem}_detections.h5"))


def save_h5(path, coords, scores, metadata, predet_candidates):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "coordinates", data=coords.astype(np.float32), compression="gzip",
            shuffle=True, chunks=True
        )
        handle.create_dataset(
            "scores", data=scores.astype(np.float32), compression="gzip",
            shuffle=True, chunks=True
        )
        for key, value in metadata.items():
            if value is not None:
                handle.attrs[key] = value
        if predet_candidates:
            group = handle.create_group("pre_detection")
            group.create_dataset(
                "counts", data=np.asarray([item[0] for item in predet_candidates], dtype=np.int64)
            )
            group.create_dataset(
                "scale_factors", data=np.asarray([item[1] for item in predet_candidates], dtype=np.float64)
            )
            labels = [
                "original" if item[2] is None else f"max-side={item[2]}"
                for item in predet_candidates
            ]
            group.create_dataset("labels", data=np.asarray(labels, dtype=h5py.string_dtype()))


def densest_visualization_crop(coords, width, height, crop_size=1024):
    """
    Try to find a crop of the source image that contains the densest cluster of detected coordinates, up to 1024x1024 pixels.
    If no coordinates are available, return a center crop.
    """
    crop_width = min(crop_size, width)
    crop_height = min(crop_size, height)
    if len(coords) == 0:
        center_x, center_y = width / 2, height / 2
    else:
        bin_size = 32
        hist_width = max(1, int(np.ceil(width / bin_size)))
        hist_height = max(1, int(np.ceil(height / bin_size)))
        hist = np.zeros((hist_height, hist_width), dtype=np.uint32)
        bin_x = np.clip((coords[:, 0] / bin_size).astype(int), 0, hist_width - 1)
        bin_y = np.clip((coords[:, 1] / bin_size).astype(int), 0, hist_height - 1)
        np.add.at(hist, (bin_y, bin_x), 1)
        kernel_width = max(1, int(np.ceil(crop_width / bin_size)))
        kernel_height = max(1, int(np.ceil(crop_height / bin_size)))
        kernel_width = min(kernel_width, hist_width)
        kernel_height = min(kernel_height, hist_height)
        integral = np.pad(hist, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        density = (
            integral[kernel_height:, kernel_width:]
            - integral[:-kernel_height, kernel_width:]
            - integral[kernel_height:, :-kernel_width]
            + integral[:-kernel_height, :-kernel_width]
        )
        peak_y, peak_x = np.unravel_index(np.argmax(density), density.shape)
        coarse_x0 = peak_x * bin_size
        coarse_y0 = peak_y * bin_size
        in_peak = (
            (coords[:, 0] >= coarse_x0)
            & (coords[:, 0] < coarse_x0 + crop_width)
            & (coords[:, 1] >= coarse_y0)
            & (coords[:, 1] < coarse_y0 + crop_height)
        )
        if np.any(in_peak):
            center_x, center_y = np.median(coords[in_peak], axis=0)
        else:
            center_x = (peak_x + kernel_width / 2) * bin_size
            center_y = (peak_y + kernel_height / 2) * bin_size

    x0 = int(np.clip(round(center_x - crop_width / 2), 0, max(0, width - crop_width)))
    y0 = int(np.clip(round(center_y - crop_height / 2), 0, max(0, height - crop_height)))
    return x0, y0, crop_width, crop_height


def save_visualization(source, coords, scores, h5_path):
    """
    Save a compact PNG visualization of the densest crop of the source image with detected coordinates overlaid, color-coded by confidence score.
    """
    x0, y0, width, height = densest_visualization_crop(
        coords, source.width, source.height
    )
    image = source.read_scaled_region(x0, y0, width, height, scale=1.0)
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    inside = (
        (coords[:, 0] >= x0)
        & (coords[:, 0] < x0 + width)
        & (coords[:, 1] >= y0)
        & (coords[:, 1] < y0 + height)
    )
    for (x, y), score in zip(coords[inside], scores[inside]):
        local_x, local_y = float(x - x0), float(y - y0)
        radius = 3
        color = (255, int(255 * (1.0 - score)), 0)
        draw.ellipse(
            (local_x - radius, local_y - radius, local_x + radius, local_y + radius),
            outline=color, width=2
        )
    visualization_path = os.path.splitext(h5_path)[0] + "_visualization.png"
    canvas.save(visualization_path)
    print(f"Visualization saved to: {visualization_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Whole-image DeNuC detection for common images and SVS/WSI files."
    )
    parser.add_argument("--model", required=True, help="Checkpoint file or experiment directory.")
    parser.add_argument("--input", required=True, help="Input image or WSI path.")
    parser.add_argument("--output", required=True, help="Output .h5 file or output directory.")
    parser.add_argument("--visualize", action="store_true", help="Save a compact PNG visualization.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image does not exist: {args.input}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, model_size, checkpoint_path, arch = load_model(args.model, device)
    batch_size = autotune_batch_size(model, model_size, device)
    source = open_image_source(args.input)
    predet_candidates = []
    try:
        if source.mpp is not None:
            scale = source.mpp / TARGET_MPP
            scale_method = "metadata_mpp_to_40x"
            print(f"Source MPP: {source.mpp:.6g}; inference scale: {scale:.6g}x")
        else:
            print("Source MPP is unavailable; selecting scale by pre-detection.")
            scale, scale_method, predet_candidates = select_unknown_scale(
                source, model, model_size, device, batch_size
            )

        scaled_coords, scores = infer_scaled_image(
            source, scale, model, model_size, device,
            description="Whole-image detection", batch_size=batch_size
        )
        original_coords = scaled_coords / scale
        original_coords[:, 0] = np.clip(original_coords[:, 0], 0, source.width - 1)
        original_coords[:, 1] = np.clip(original_coords[:, 1], 0, source.height - 1)

        h5_path = output_h5_path(args.output, args.input)
        metadata = {
            "format_version": "denuc-anyimage-v1",
            "source_path": os.path.abspath(args.input),
            "source_width": source.width,
            "source_height": source.height,
            "source_mpp": source.mpp,
            "target_mpp": TARGET_MPP if source.mpp is not None else None,
            "inference_scale_factor": scale,
            "scale_method": scale_method,
            "checkpoint_path": checkpoint_path,
            "architecture": arch,
            "model_input_size": model_size,
            "inference_batch_size": batch_size,
            "window_size": WINDOW_SIZE,
            "overlap_ratio": OVERLAP_RATIO,
            "score_threshold": SCORE_THRESHOLD,
            "nms_distance_at_inference_scale": NMS_DISTANCE,
            "coordinate_system": "level-0/source-image pixels; columns are x,y",
        }
        save_h5(h5_path, original_coords, scores, metadata, predet_candidates)
        print(f"Detected {len(original_coords)} nuclei")
        print(f"Results saved to: {h5_path}")

        if args.visualize:
            save_visualization(source, original_coords, scores, h5_path)
    finally:
        source.close()


if __name__ == "__main__":
    main()
