#!/bin/bash

# Usage example (adjust paths and iteration):
#   bash scripts/run_render_get_datasets.sh \
#     /data2/peilincai/InstantSplat/assets/sora/example \
#     /data2/peilincai/InstantSplat/output_infer/sora/example/24_views \
#     24 \
#     8000

set -e

SOURCE_PATH=${1:-"/data2/peilincai/InstantSplat/assets/sora/example"}
MODEL_PATH=${2:-"/data2/peilincai/InstantSplat/output_infer/sora/example/24_views"}
N_VIEWS=${3:-24}
ITER=${4:-8000}

export PYTHONWARNINGS="ignore"

python -W ignore ./render_get_datasets.py \
  -s ${SOURCE_PATH} \
  -m ${MODEL_PATH} \
  -r 1 \
  --n_views ${N_VIEWS} \
  --iterations ${ITER} \
  --get_datasets

echo "GET_DATASETS rendering complete. Ghosts and mapping saved under: ${MODEL_PATH}/train/ours_${ITER}/"


