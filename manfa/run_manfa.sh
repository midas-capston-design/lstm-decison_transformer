#!/usr/bin/env bash
set -euo pipefail

# 기본 경로/설정. 필요하면 인자로 덮어쓰기 가능.
CONFIG_PATH=${1:-manfa/configs/default.yaml}
FIELD_DIR=${2:-results/manfa_field}
FULL_DIR=${3:-results/manfa_full}
WINDOW_DIR=${4:-flow_matching/processed_data_flow_matching}
SPLIT=${5:-test}
OUTPUT_PATH=${6:-results/manfa/predictions.jsonl}

# CUDA 메모리 파편화를 줄이기 위한 기본 allocator 설정
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo ">>> Stage 1: Neural Field + Encoder pre-training"
python3 -m manfa.train \
  --config "${CONFIG_PATH}" \
  --stage field \
  --save-dir "${FIELD_DIR}"

echo ">>> Stage 2: Full MaNFA training (particle flow 포함)"
python3 -m manfa.train \
  --config "${CONFIG_PATH}" \
  --stage full \
  --save-dir "${FULL_DIR}"

# 최신 체크포인트 자동 탐색 (직접 지정하려면 7번째 인자 전달)
CHECKPOINT_PATH=${7:-}
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo ">>> 자동으로 최신 체크포인트를 찾습니다..."
  CHECKPOINT_PATH=$(ls "${FULL_DIR}"/epoch_*.pt | sort | tail -n 1)
fi
echo ">>> Inference checkpoint: ${CHECKPOINT_PATH}"

echo ">>> Stage 3: Streaming inference on ${SPLIT} split"
python3 -m manfa.inference \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --window-dir "${WINDOW_DIR}" \
  --split "${SPLIT}" \
  --out "${OUTPUT_PATH}"

echo "모든 단계 완료! 결과 파일: ${OUTPUT_PATH}"
