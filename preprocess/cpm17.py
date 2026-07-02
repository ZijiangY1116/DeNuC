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

import scipy.io as sio
import tqdm
import numpy as np
from PIL import Image

from common_nuclei import (
    instance_map_to_centers,
    prepare_output,
    save_metadata,
    save_tiled_sample,
    split_train_samples,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpm17_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_preview", action="store_true")
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()

    prepare_output(args.output_folder)
    train_images = sorted(glob.glob(os.path.join(args.cpm17_folder, "train", "Images", "*.png")))
    test_images = sorted(glob.glob(os.path.join(args.cpm17_folder, "test", "Images", "*.png")))
    split_map = split_train_samples([os.path.splitext(os.path.basename(p))[0] for p in train_images], args.seed)
    jobs = [("train", p) for p in train_images] + [("test", p) for p in test_images]
    if args.max_samples is not None:
        jobs = jobs[: args.max_samples]
    metadata = {"sample_name": [], "split": []}
    for subset, image_path in tqdm.tqdm(jobs):
        name = os.path.splitext(os.path.basename(image_path))[0]
        split = split_map[name] if subset == "train" else "test"
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        inst_map = sio.loadmat(os.path.join(args.cpm17_folder, subset, "Labels", f"{name}.mat"))["inst_map"]
        save_tiled_sample(
            image, instance_map_to_centers(inst_map), f"{subset}_{name}", split,
            args.output_folder, mpp=-1, metadata=metadata, save_preview=args.save_preview
        )
    save_metadata(metadata, args.output_folder)


if __name__ == "__main__":
    main()
