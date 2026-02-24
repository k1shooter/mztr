#!/usr/bin/env bash
set -euo pipefail

# 데이터셋 경로 및 결과 저장 경로 설정
BASE_SAVE="runs/synthetic_experiments"
mkdir -p "${BASE_SAVE}"

# =====================================================================
# 공통 하이퍼파라미터 세팅
# =====================================================================
COMMON_ARGS=(
  --epochs 10000                 # 충분한 룰 학습을 위한 에포크 수
  --batch_size 256               # max_nodes=256일 때 OOM 방지를 위해 256 권장
  --sample_every_epochs 50      # 100 에포크마다 샘플링 및 시각화 저장
  --sample_steps 200             # 샘플링 시 Diffusion 스텝 수
  --profile_count_mode nodes     # 트리 노드 밀도 기반 스케줄링
  --profile_mode exp             # 논문/실험 기본 세팅
  --schedule_mode profiled
  --variant edit                 # TreeDFM (insert+delete+sub) 기본 모델
  --no_perm_invariant            # 자식 순서(Left/Right 등)가 중요하므로 Permutation Invariance 해제
  --profile_overlap 0.0
)

echo "Starting 4 concurrent TreeDFM trainings tailored for each dataset..."

# =====================================================================
# 1. 수식 평가 트리 (Arithmetic Expr) -> GPU 0
# - max_depth: 10 -> 마진 두어 16
# - max_size: 59 -> 마진 두어 128
# - k: 항상 2개로 분할되므로 2
# - num_types: 1~11까지 쓰므로 0-index 감안하여 12
# =====================================================================
echo "  [GPU 0] Launching Arithmetic Expr Tree..."
CUDA_VISIBLE_DEVICES=1 python src/treedfm_edit.py "${COMMON_ARGS[@]}" \
  --pkl_path "synthetic_datasets/arithmetic_expr.pkl" \
  --save_dir "${BASE_SAVE}/arithmetic" \
  --max_depth 16 --max_nodes 128 --pad_to 128 \
  --k 2 --num_types 12 \
  --root_type 11 &

# =====================================================================
# 2. 자원 보존 트리 (Resource Flow) -> GPU 1
# - max_depth: 12 -> 마진 두어 16
# - max_size: 108 -> 마진 두어 256
# - k: 최대 3분기이므로 3
# - num_types: 자원량 1~60까지 쓰므로 넉넉히 64
# =====================================================================
echo "  [GPU 1] Launching Resource Flow Tree..."
CUDA_VISIBLE_DEVICES=2 python src/treedfm_edit.py "${COMMON_ARGS[@]}" \
  --pkl_path "synthetic_datasets/resource_flow.pkl" \
  --save_dir "${BASE_SAVE}/resource_flow" \
  --max_depth 16 --max_nodes 256 --pad_to 256 \
  --k 3 --num_types 61 \
  --root_type 60 &

# =====================================================================
# 3. 스코프 의존성 트리 (Scope Dependency) -> GPU 2
# - max_depth: 10 -> 마진 두어 16
# - max_size: 90 -> 마진 두어 128
# - k: 최대 3분기이므로 3
# - num_types: BLOCK, DECL(3), USE(3) 총 7가지 -> 8
# =====================================================================
echo "  [GPU 2] Launching Scope Dependency Tree..."
CUDA_VISIBLE_DEVICES=3 python src/treedfm_edit.py "${COMMON_ARGS[@]}" \
  --pkl_path "synthetic_datasets/scope_dependency.pkl" \
  --save_dir "${BASE_SAVE}/scope_dependency" \
  --max_depth 16 --max_nodes 128 --pad_to 128 \
  --k 3 --num_types 8 \
  --root_type 1 &

# =====================================================================
# 4. 구조적 대칭 트리 (Symmetric CFG) -> GPU 3
# - max_depth: 8 -> 마진 두어 16
# - max_size: 173 -> 마진 두어 256
# - k: 자식이 최대 2개이므로 2
# - num_types: 루트(1) + 랜덤서브트리(1~5) -> 넉넉히 8
# =====================================================================
echo "  [GPU 3] Launching Symmetric Tree..."
CUDA_VISIBLE_DEVICES=4 python src/treedfm_edit.py "${COMMON_ARGS[@]}" \
  --pkl_path "synthetic_datasets/symmetric_cfg.pkl" \
  --save_dir "${BASE_SAVE}/symmetric" \
  --max_depth 16 --max_nodes 256 --pad_to 256 \
  --k 2 --num_types 6 \
  --root_type 1 &

# 백그라운드 프로세스가 끝날 때까지 대기
wait
echo "All 4 models successfully finished training!"