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

from common_nuclei import instance_map_to_centers, prepare_output, save_metadata, save_tiled_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cryonuseg_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--save_preview", action="store_true")
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()

    image_folder = os.path.join(args.cryonuseg_folder, "tissue images")
    label_folder = os.path.join(args.cryonuseg_folder, "Annotator 1 (biologist)", "label masks")
    prepare_output(args.output_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "*.tif")))
    if args.max_samples is not None:
        images = images[: args.max_samples]
    metadata = {"sample_name": [], "split": []}
    for image_path in tqdm.tqdm(images):
        name = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        inst_map = np.asarray(Image.open(os.path.join(label_folder, f"{name}.tif")))
        save_tiled_sample(
            image, instance_map_to_centers(inst_map), name, "test", args.output_folder,
            mpp=0.25, metadata=metadata, save_preview=args.save_preview
        )
    save_metadata(metadata, args.output_folder)


if __name__ == "__main__":
    main()
