#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_overlap_lag_ablation.sh <pkl_path> <base_save_dir>
# Example:
#   bash run_overlap_lag_ablation.sh maze_trees_relative.pkl runs/ablations_overlap_lag

PKL_PATH=${1:-maze_trees_relative.pkl}
BASE_SAVE=${2:-runs2/ablations_overlap_lag} # 이전과 폴더가 겹치지 않게 runs2 등으로 변경 가능

mkdir -p "${BASE_SAVE}"

# 기존 baseline_edit과 동일한 공통 파라미터 설정
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
  --variant edit
)

# 시드 고정
SEED=0

echo "Starting 2x2 Ablation Study for profile_overlap & type_lag..."

# 1. Overlap 0.0 / Lag 0.0 -> GPU 1
echo "  [GPU 1] Running overlap=0.0, lag=0.0"
CUDA_VISIBLE_DEVICES=1 python src/treedfm_edit.py "${COMMON_ARGS[@]}" \
  --profile_overlap 0.0 --type_lag 0.0 \
  --save_dir "${BASE_SAVE}/overlap0.0_lag0.0" --seed ${SEED} &

# 2. Overlap 0.0 / Lag 0.1 -> GPU 2
echo "  [GPU 2] Running overlap=0.0, lag=0.1"
CUDA_VISIBLE_DEVICES=2 python src/treedfm_edit.py "${COMMON_ARGS[@]}" \
  --profile_overlap 0.0 --type_lag 0.1 \
  --save_dir "${BASE_SAVE}/overlap0.0_lag0.1" --seed ${SEED} &

# 3. Overlap 0.1 / Lag 0.0 -> GPU 3
echo "  [GPU 3] Running overlap=0.1, lag=0.0"
CUDA_VISIBLE_DEVICES=3 python src/treedfm_edit.py "${COMMON_ARGS[@]}" \
  --profile_overlap 0.1 --type_lag 0.0 \
  --save_dir "${BASE_SAVE}/overlap0.1_lag0.0" --seed ${SEED} &

# 4. Overlap 0.1 / Lag 0.1 -> GPU 4 (Baseline 세팅과 거의 동일)
echo "  [GPU 4] Running overlap=0.1, lag=0.1"
CUDA_VISIBLE_DEVICES=4 python src/treedfm_edit.py "${COMMON_ARGS[@]}" \
  --profile_overlap 0.1 --type_lag 0.1 \
  --save_dir "${BASE_SAVE}/overlap0.1_lag0.1" --seed ${SEED} &

# 모든 백그라운드 프로세스가 끝날 때까지 대기
wait
echo "All ablation experiments are done!"