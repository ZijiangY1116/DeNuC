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
#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="denuc_all_datasets"
ARCH="denuc_det_shufflenet_x2_0"

# Datasets without a native train split remain in this list so the resulting
# experiment records the complete evaluation suite. They contribute zero
# training samples; PanNuke uses the requested 1/2/3 train/val/test rotation.
DATASETS="puma,brcam2c,ocelot,cpm15,cpm17,tnbc,kumar,consep,cryonuseg,pannuke123"

usage() {
	echo "Usage: $0 [--exp_name EXP_NAME] [--arch ARCH] [extra denuc_train.py arguments]"
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
	case "$1" in
		--exp_name)
			EXP_NAME="$2"
			shift 2
			;;
		--arch)
			ARCH="$2"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			EXTRA_ARGS+=("$1")
			shift
			;;
	esac
done

echo "[INFO] Training on: ${DATASETS}"
python ./denuc_train.py \
	--arch "${ARCH}" \
	--exp_name "${EXP_NAME}" \
	--datasets "${DATASETS}" \
	"${EXTRA_ARGS[@]}"

