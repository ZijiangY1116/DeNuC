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

import tqdm
import numpy as np
from PIL import Image

from common_nuclei import binary_map_to_centers, prepare_output, save_metadata, save_tiled_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tnbc_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--save_preview", action="store_true")
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()

    data_folder = args.tnbc_folder
    if os.path.isdir(os.path.join(data_folder, "TNBC_dataset")):
        data_folder = os.path.join(data_folder, "TNBC_dataset")
    prepare_output(args.output_folder)
    masks = sorted(glob.glob(os.path.join(data_folder, "GT_*", "*.png")))
    if args.max_samples is not None:
        masks = masks[: args.max_samples]
    metadata = {"sample_name": [], "split": []}
    for mask_path in tqdm.tqdm(masks):
        slide_id = os.path.basename(os.path.dirname(mask_path)).replace("GT_", "")
        name = os.path.splitext(os.path.basename(mask_path))[0]
        image_path = os.path.join(data_folder, f"Slide_{slide_id}", f"{name}.png")
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path))
        save_tiled_sample(
            image, binary_map_to_centers(mask), f"{slide_id}-{name}", "test",
            args.output_folder, mpp=0.25, metadata=metadata, save_preview=args.save_preview
        )
    save_metadata(metadata, args.output_folder)


if __name__ == "__main__":
    main()
