#!/bin/bash
# 모델 학습 스크립트

set -e  # 에러 발생 시 중단

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

echo "🧠 모델 학습 시작..."

python3 src/pipeline.py train \
    --data-dir data/processed \
    --nodes data/nodes_final.csv \
    --epochs 50 \
    --batch-size 16 \
    --hidden-dim 256 \
    --depth 8 \
    --lr 2e-4 \
    --dropout 0.15 \
    --checkpoint-dir checkpoints \
    --patience 10

echo "✅ 학습 완료!"
