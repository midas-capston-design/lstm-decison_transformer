#!/bin/bash
# 대형 모델 학습 스크립트 (RMSE < 2m 목표)

set -e  # 에러 발생 시 중단

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

echo "🧠 대형 모델 학습 시작..."

python3 src/pipeline.py train \
    --data-dir data/processed \
    --nodes data/nodes_final.csv \
    --epochs 100 \
    --batch-size 16 \
    --hidden-dim 256 \
    --depth 8 \
    --lr 1.5e-4 \
    --dropout 0.2 \
    --checkpoint-dir checkpoints \
    --patience 15

echo "✅ 학습 완료!"
