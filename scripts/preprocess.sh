#!/bin/bash
# 데이터 전처리 스크립트

set -e  # 에러 발생 시 중단

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

echo "📊 데이터 전처리 시작..."

python3 src/pipeline.py preprocess \
    --law-dir data/raw \
    --nodes data/nodes_final.csv \
    --output data/processed \
    --min-samples-per-path 0

echo "✅ 전처리 완료!"
