#!/bin/bash

set -euo pipefail

# Config
ASSETS_BASE="/data2/peilincai/InstantSplat/assets/sora/example/streetview_outputs"
TMP_BASE="/data2/peilincai/InstantSplat/assets/sora/example/tmp_pairs"
MODEL_PATH="/data2/peilincai/InstantSplat/output_infer/sora/example/16_views"
GHOST_DATASETS="/data2/peilincai/InstantSplat/output_infer/sora/example/ghost_datasets"

N_VIEWS=16
ITER=4000

mkdir -p "$MODEL_PATH" "$GHOST_DATASETS/ghosts" "$TMP_BASE"

# GPU picker (same threshold logic as run_infer.sh)
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '$2 < threshold { print $1; exit }'
}

# Merge mapping json
merge_mapping() {
  local dst_json="$1"   # $GHOST_DATASETS/ghosts_mapping.json
  local src_json="$2"   # $MODEL_PATH/train/ours_${ITER}/ghosts_mapping.json
  python - "$dst_json" "$src_json" <<'PY'
import json, sys, os
dst, src = sys.argv[1], sys.argv[2]
if os.path.exists(dst):
    try:
        with open(dst, 'r') as f:
            a = json.load(f)
        if not isinstance(a, list):
            a = []
    except Exception:
        a = []
else:
    a = []
with open(src, 'r') as f:
    b = json.load(f)
    if not isinstance(b, list):
        b = []
existing = { (d.get('ghost_image'), d.get('ground_truth')): d for d in a if isinstance(d, dict) }
for e in b:
    if isinstance(e, dict):
        existing[(e.get('ghost_image'), e.get('ground_truth'))] = e
merged = list(existing.values())
with open(dst, 'w') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
print(f"Merged {len(b)} entries; total {len(merged)}")
PY
}

# Build temporary dataset from two pt folders (8 + 8 = 16 images)
build_temp_dataset() {
  local pt1="$1"   # e.g., 001
  local pt2="$2"   # e.g., 002
  local tmp_dir="$TMP_BASE/pair_${pt1}_${pt2}"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir/images"

  local src1="$ASSETS_BASE/pt_${pt1}"
  local src2="$ASSETS_BASE/pt_${pt2}"

  # Link up to 8 images from each point (sorted)
  find "$src1" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -n 8 | while read -r f; do ln -s "$f" "$tmp_dir/images/"; done
  find "$src2" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -n 8 | while read -r f; do ln -s "$f" "$tmp_dir/images/"; done

  echo "$tmp_dir"
}

# Train and render ghosts for a temp dataset
process_pair() {
  local pt1="$1"
  local pt2="$2"
  local tmp_src
  tmp_src=$(build_temp_dataset "$pt1" "$pt2")

  # Clean previous model artifacts to avoid mixing outputs
  rm -rf "$MODEL_PATH/point_cloud" "$MODEL_PATH/pose" "$MODEL_PATH/train"
  mkdir -p "$MODEL_PATH"

  # GPU selection
  local GPU_ID
  GPU_ID=$(get_available_gpu || true)
  while [ -z "${GPU_ID:-}" ]; do
    echo "[$(date '+%F %T')] No GPU available, retry in 30s..."
    sleep 30
    GPU_ID=$(get_available_gpu || true)
  done

  echo "======================================================="
  echo "Pair (pt_${pt1}, pt_${pt2}) on GPU ${GPU_ID}"
  echo "Temp dataset: $tmp_src"
  echo "Model path:   $MODEL_PATH"
  echo "======================================================="

  # 1) init_geo
  echo "[$(date '+%F %T')] init_geo..."
  CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore /data2/peilincai/InstantSplat/init_geo.py \
    -s "$tmp_src" \
    -m "$MODEL_PATH" \
    --n_views ${N_VIEWS} \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking \
    --infer_video \
    > "$MODEL_PATH/01_init_geo_pt_${pt1}_${pt2}.log" 2>&1

  # 2) train
  echo "[$(date '+%F %T')] train (${ITER} iters)..."
  CUDA_VISIBLE_DEVICES=${GPU_ID} python /data2/peilincai/InstantSplat/train.py \
    -s "$tmp_src" \
    -m "$MODEL_PATH" \
    -r 1 \
    --n_views ${N_VIEWS} \
    --iterations ${ITER} \
    --position_lr_init 3e-5 \
    --position_lr_final 3e-7 \
    --position_lr_delay_mult 0.01 \
    --position_lr_max_steps ${ITER} \
    --feature_lr 0.0025 \
    --opacity_lr 0.05 \
    --scaling_lr 0.003 \
    --rotation_lr 3e-4 \
    --lambda_dssim 0.2 \
    --init_scale_from_view_depth \
    --pp_optimizer \
    --optim_pose \
    > "$MODEL_PATH/02_train_pt_${pt1}_${pt2}.log" 2>&1

  # 3) render ghosts
  echo "[$(date '+%F %T')] render ghosts..."
  CUDA_VISIBLE_DEVICES=${GPU_ID} python /data2/peilincai/InstantSplat/render_checkpoint.py \
    -m "$MODEL_PATH" \
    -s "$tmp_src" \
    --iterations ${ITER} \
    --n_views ${N_VIEWS} \
    --get_datasets \
    --neighbor_span 2 \
    --min_coverage 0.90

  # 4) collect outputs
  local SRC_GHOSTS="$MODEL_PATH/train/ours_${ITER}/ghosts"
  local SRC_JSON="$MODEL_PATH/train/ours_${ITER}/ghosts_mapping.json"
  if [ -d "$SRC_GHOSTS" ]; then
    cp -f "$SRC_GHOSTS"/*.png "$GHOST_DATASETS/ghosts/" || true
  fi
  if [ -f "$SRC_JSON" ]; then
    merge_mapping "$GHOST_DATASETS/ghosts_mapping.json" "$SRC_JSON"
  fi

  # 5) cleanup temp dataset of the pair (optional)
  rm -rf "$tmp_src"
}


main() {
  # Iterate pairs: (1,2), (3,4), ..., (49,50)
  for i in $(seq -w 001 2 049); do
    j=$(printf "%03d" $((10#${i}+1)))
    process_pair "$i" "$j"
  done

  echo "======================================================="
  echo "All pairs processed. Ghosts at: $GHOST_DATASETS/ghosts"
  echo "Merged mapping: $GHOST_DATASETS/ghosts_mapping.json"
  echo "======================================================="
}

main "$@"


