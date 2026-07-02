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
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage


TARGET_MPP = 0.25
PATCH_SIZE = 256


def prepare_output(output_folder):
    if os.path.exists(output_folder):
        print(f"Warning: removing existing output folder: {output_folder}")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)


def instance_map_to_centers(instance_map):
    centers = []
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        ys, xs = np.nonzero(instance_map == instance_id)
        if len(xs):
            centers.append([int(np.round(xs.mean())), int(np.round(ys.mean()))])
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2)


def binary_map_to_centers(binary_map):
    if binary_map.ndim == 3:
        binary_map = np.any(binary_map > 0, axis=2)
    labels, count = ndimage.label(binary_map > 0)
    centers = []
    for instance_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == instance_id)
        if len(xs):
            centers.append([int(np.round(xs.mean())), int(np.round(ys.mean()))])
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2)


def split_train_samples(sample_names, seed=0, val_ratio=0.2):
    """Return a deterministic image-level train/validation mapping."""
    sample_names = sorted(sample_names)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sample_names))
    n_val = max(1, int(np.ceil(len(sample_names) * val_ratio)))
    val_names = {sample_names[i] for i in order[:n_val]}
    return {name: ("val" if name in val_names else "train") for name in sample_names}


def save_tiled_sample(
    image,
    centers,
    raw_sample_name,
    split,
    output_folder,
    mpp,
    metadata,
    save_preview=False,
):
    """Convert to 40x when possible, tile without overlap, and save DeNuC npy files."""
    image = np.asarray(image, dtype=np.uint8)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)

    if mpp > 0:
        scale = mpp / TARGET_MPP
        if not np.isclose(scale, 1.0):
            new_width = max(1, int(round(image.shape[1] * scale)))
            new_height = max(1, int(round(image.shape[0] * scale)))
            image = np.asarray(
                Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BICUBIC)
            )
            centers *= scale

    height, width = image.shape[:2]
    n_cols = int(np.ceil(width / PATCH_SIZE))
    n_rows = int(np.ceil(height / PATCH_SIZE))

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            x0 = col_idx * PATCH_SIZE
            y0 = row_idx * PATCH_SIZE
            x1 = min(x0 + PATCH_SIZE, width)
            y1 = min(y0 + PATCH_SIZE, height)

            patch = np.full((PATCH_SIZE, PATCH_SIZE, 3), 255, dtype=np.uint8)
            patch[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1]

            inside = (
                (centers[:, 0] >= x0)
                & (centers[:, 0] < x1)
                & (centers[:, 1] >= y0)
                & (centers[:, 1] < y1)
            )
            patch_centers = centers[inside] - np.array([x0, y0], dtype=np.float32)
            patch_centers = np.rint(patch_centers).astype(np.int32)
            patch_centers[:, 0] = np.clip(patch_centers[:, 0], 0, PATCH_SIZE - 1)
            patch_centers[:, 1] = np.clip(patch_centers[:, 1], 0, PATCH_SIZE - 1)
            annotations = np.column_stack(
                [patch_centers, np.zeros(len(patch_centers), dtype=np.int32)]
            ).astype(np.int32)

            sample_name = f"{raw_sample_name}_{col_idx}_{row_idx}"
            np.save(
                os.path.join(output_folder, f"{sample_name}.npy"),
                {"patch": patch, "ann": annotations},
            )
            metadata["sample_name"].append(sample_name)
            metadata["split"].append(split)

            if save_preview:
                preview = Image.fromarray(patch)
                draw = ImageDraw.Draw(preview)
                for x, y in patch_centers:
                    draw.ellipse((int(x) - 4, int(y) - 4, int(x) + 4, int(y) + 4), fill=(255, 0, 0))
                preview_folder = os.path.join(output_folder, "preview")
                os.makedirs(preview_folder, exist_ok=True)
                preview.save(os.path.join(preview_folder, f"{sample_name}.png"))


def save_metadata(metadata, output_folder):
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(output_folder, "meta.csv"), index=False)
    print(meta_df["split"].value_counts().sort_index())
    print(f"Saved {len(meta_df)} patches to {output_folder}")
