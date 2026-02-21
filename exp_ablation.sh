#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_ablations.sh <pkl_path> <base_save_dir>
# Example:
#   bash run_ablations.sh maze_trees_relative.pkl runs/ablations

PKL_PATH=${1:-maze_trees_relative.pkl}
BASE_SAVE=${2:-runs1/ablations}

mkdir -p "${BASE_SAVE}"

# Common args (EDIT THESE to match your baseline command)
COMMON_ARGS=(
  --pkl_path "${PKL_PATH}"
  --max_depth 80 --k 3 --num_types 3
  --expected_nodes 100
  --pad_to 200 --max_nodes 256
  --batch_size 256 --epochs 30000
  --sample_every_epochs 5 --sample_steps 500
  --profile_count_mode nodes
  --profile_mode exp
  --no_perm_invariant
  --schedule_mode profiled
)

# Optional: set a fixed seed for fair comparison
SEED=0

echo "Running baseline(edit), ablation1(insert_only), ablation2(depth_agnostic_kappa)"

CUDA_VISIBLE_DEVICES=1 python src/treedfm_edit.py "${COMMON_ARGS[@]}" --save_dir "${BASE_SAVE}/baseline_edit" --seed ${SEED} &
CUDA_VISIBLE_DEVICES=2 python src/treedfm_edit.py "${COMMON_ARGS[@]}" --save_dir "${BASE_SAVE}/ablation1_insert_only" --variant insert_only --seed ${SEED} &
CUDA_VISIBLE_DEVICES=3 python src/treedfm_edit.py "${COMMON_ARGS[@]}" --save_dir "${BASE_SAVE}/ablation2_depth_agnostic" --depth_agnostic_kappa --time_only_width 1.0 --seed ${SEED} &

wait
echo "Done."