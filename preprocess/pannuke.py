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
import os

import numpy as np
import tqdm

from common_nuclei import instance_map_to_centers, prepare_output, save_metadata, save_tiled_sample


SPLIT_CONFIGS = {
    "pannuke123": {1: "train", 2: "val", 3: "test"},
    "pannuke231": {2: "train", 3: "val", 1: "test"},
    "pannuke312": {3: "train", 1: "val", 2: "test"},
}


def load_fold(folder, fold):
    images = np.load(
        os.path.join(folder, f"Fold {fold}", "images", f"fold{fold}", "images.npy"),
        mmap_mode="r",
    )
    masks = np.load(
        os.path.join(folder, f"Fold {fold}", "masks", f"fold{fold}", "masks.npy"),
        mmap_mode="r",
    )
    return images, masks


def mask_to_centers(mask):
    centers = []
    for class_idx in range(5):
        class_map = mask[:, :, class_idx]
        centers.append(instance_map_to_centers(class_map))
    nonempty = [item for item in centers if len(item)]
    return np.concatenate(nonempty, axis=0) if nonempty else np.empty((0, 2), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pannuke_folder", required=True)
    parser.add_argument("--output_root", default="./dataset")
    parser.add_argument("--save_preview", action="store_true")
    parser.add_argument("--max_samples_per_fold", type=int)
    args = parser.parse_args()

    outputs = {}
    metadata = {}
    for dataset_name in SPLIT_CONFIGS:
        output_folder = os.path.join(args.output_root, dataset_name)
        prepare_output(output_folder)
        outputs[dataset_name] = output_folder
        metadata[dataset_name] = {"sample_name": [], "split": []}

    for fold in (1, 2, 3):
        images, masks = load_fold(args.pannuke_folder, fold)
        count = len(images)
        if args.max_samples_per_fold is not None:
            count = min(count, args.max_samples_per_fold)
        for index in tqdm.tqdm(range(count), desc=f"PanNuke fold {fold}"):
            sample_name = f"Fold{fold}_Sample{index + 1:04d}"
            centers = mask_to_centers(masks[index])
            for dataset_name, split_map in SPLIT_CONFIGS.items():
                save_tiled_sample(
                    images[index], centers, sample_name, split_map[fold],
                    outputs[dataset_name], mpp=0.25, metadata=metadata[dataset_name],
                    save_preview=args.save_preview
                )

    for dataset_name in SPLIT_CONFIGS:
        save_metadata(metadata[dataset_name], outputs[dataset_name])


if __name__ == "__main__":
    main()
