#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-manfa/configs/h100.yaml}
FIELD_DIR=${2:-results/manfa_h100_field}
FULL_DIR=${3:-results/manfa_h100_full}
WINDOW_DIR=${4:-flow_matching/processed_data_flow_matching}
SPLIT=${5:-test}
OUTPUT_PATH=${6:-results/manfa_h100/predictions.jsonl}

echo ">>> [H100] Stage 1: Neural Field + Encoder pre-training"
python -m manfa.train \
  --config "${CONFIG_PATH}" \
  --stage field \
  --save-dir "${FIELD_DIR}"

echo ">>> [H100] Stage 2: Full MaNFA training"
python -m manfa.train \
  --config "${CONFIG_PATH}" \
  --stage full \
  --save-dir "${FULL_DIR}"

CHECKPOINT_PATH=${7:-}
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo ">>> 최신 체크포인트 탐색 중..."
  CHECKPOINT_PATH=$(ls "${FULL_DIR}"/epoch_*.pt | sort | tail -n 1)
fi
echo ">>> [H100] Inference checkpoint: ${CHECKPOINT_PATH}"

echo ">>> [H100] Stage 3: Streaming inference on ${SPLIT}"
python -m manfa.inference \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --window-dir "${WINDOW_DIR}" \
  --split "${SPLIT}" \
  --out "${OUTPUT_PATH}"

echo "[H100] 파이프라인 완료! 결과: ${OUTPUT_PATH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
