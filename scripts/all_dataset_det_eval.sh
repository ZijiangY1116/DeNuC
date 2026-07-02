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

EXP_NAME=""
EVAL_MODE="test"
NMS_DIST="12.0"

DATASETS=(
	puma
	brcam2c
	ocelot
	cpm15
	cpm17
	tnbc
	kumar
	consep
	cryonuseg
	pannuke123
	pannuke231
	pannuke312
)

usage() {
	echo "Usage: $0 --exp_name EXP_NAME [--eval_mode train|val|test] [--nms_dist PIXELS]"
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--exp_name)
			EXP_NAME="$2"
			shift 2
			;;
		--eval_mode)
			EVAL_MODE="$2"
			shift 2
			;;
		--nms_dist)
			NMS_DIST="$2"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage
			exit 1
			;;
	esac
done

if [[ -z "${EXP_NAME}" ]]; then
	usage
	exit 1
fi

for dataset_name in "${DATASETS[@]}"; do
	echo "[INFO] Evaluating ${dataset_name} (${EVAL_MODE})"
	python ./denuc_eval.py \
		--exp_name "${EXP_NAME}" \
		--eval_dataset "${dataset_name}" \
		--eval_mode "${EVAL_MODE}" \
		--nms_dist "${NMS_DIST}"
done

