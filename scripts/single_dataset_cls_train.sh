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

GPU_ID="0"
EXP_NAME="cls_exp"
DET_EXP_NAME=""
DATASET=""

print_usage() {
	cat <<'EOF'
Usage:
	./scripts/single_dataset_cls_train.sh -i <GPU_ID> --exp_name <EXP_NAME> --det_exp_name <DET_EXP_NAME> --dataset <DATASET>

Examples:
	./scripts/single_dataset_cls_train.sh -i 0 --exp_name cls_exp --det_exp_name 20260209_160250_denuc_experiment --dataset puma
EOF
}

if [[ $# -eq 0 ]]; then
	print_usage
	exit 1
fi

while [[ $# -gt 0 ]]; do
	case "$1" in
		-i|--gpu)
			GPU_ID="$2"
			shift 2
			;;
		--exp_name)
			EXP_NAME="$2"
			shift 2
			;;
		--det_exp_name)
			DET_EXP_NAME="$2"
			shift 2
			;;
		--dataset)
			DATASET="$2"
			shift 2
			;;
		-h|--help)
			print_usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			print_usage
			exit 1
			;;
	esac
done

if [[ -z "$GPU_ID" || -z "$DET_EXP_NAME" || -z "$DATASET" ]]; then
	echo "Missing required arguments." >&2
	print_usage
	exit 1
fi

run_train() {
	local dataset="$1"
	echo "[INFO] Training dataset: ${dataset}"
	CUDA_VISIBLE_DEVICES="${GPU_ID}" \
		python cls_train.py \
			--exp_name "${EXP_NAME}_${dataset}" \
			--det_exp_name "${DET_EXP_NAME}" \
			--dataset "${dataset}"
}

run_train "${DATASET}"
