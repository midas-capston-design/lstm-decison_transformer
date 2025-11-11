#!/bin/bash

# 실내 위치 추정 모델 학습 스크립트
# Usage: ./train.sh [model_number]

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 도움말 출력
print_help() {
    echo "======================================"
    echo "  실내 위치 추정 모델 학습 스크립트"
    echo "======================================"
    echo ""
    echo "사용법: ./train.sh [숫자]"
    echo ""
    echo "모델 선택:"
    echo "  1 - LSTM v1 (Baseline, 마커만 사용)"
    echo "  2 - LSTM v2 (마커 사이 데이터, Stride=10)"
    echo "  3 - LSTM v3 (Stride=5 + 데이터 증강)"
    echo "  4 - LSTM v4 (그리드 0.9m)"
    echo "  5 - Fuzzy Normalization + LSTM"
    echo "  6 - Decision Transformer (강화학습 기반)"
    echo "  7 - 전체 학습 (LSTM v4, LSTM v4 Fuzzy, DT v2, DT v2 Fuzzy)"
    echo ""
    echo "예제:"
    echo "  ./train.sh 4    # LSTM v4 학습"
    echo "  ./train.sh 7    # 4개 모델 모두 학습"
    echo ""
}

# 가상환경 활성화
activate_venv() {
    if [ ! -d "venv" ]; then
        echo -e "${RED}Error: 가상환경이 없습니다. 먼저 venv를 만들어주세요.${NC}"
        echo "명령어: python3 -m venv venv && source venv/bin/activate"
        exit 1
    fi

    source venv/bin/activate
    echo -e "${GREEN}✓ 가상환경 활성화 완료${NC}"
}

# v1 학습
train_v1() {
    echo -e "${YELLOW}[v1] LSTM Baseline 학습 시작${NC}"
    echo "- 데이터: 마커만 사용"
    echo "- 샘플 수: 12,179개"
    echo "- 클래스: 1,387개"
    echo ""

    cd v1

    # 전처리 데이터가 없으면 실행
    if [ ! -d "processed_data" ]; then
        echo -e "${YELLOW}전처리 데이터가 없습니다. preprocessing.py 실행...${NC}"
        python preprocessing.py
    fi

    # 학습 (v1은 train_lstm.py가 없으므로 v3 것을 사용)
    echo -e "${YELLOW}학습 시작...${NC}"
    if [ -f "train_lstm.py" ]; then
        python train_lstm.py
    else
        echo -e "${RED}Error: train_lstm.py가 없습니다.${NC}"
        echo "v1은 데이터가 부족하여 학습이 권장되지 않습니다."
        exit 1
    fi

    cd ..
}

# v2 학습
train_v2() {
    echo -e "${YELLOW}[v2] LSTM Stride=10 학습 시작${NC}"
    echo "- 데이터: 마커 + 마커 사이 (Stride=10)"
    echo "- 샘플 수: 38,747개"
    echo "- 클래스: 2,484개"
    echo ""

    cd v2

    # 전처리 데이터가 없으면 실행
    if [ ! -d "processed_data_v2" ]; then
        echo -e "${YELLOW}전처리 데이터가 없습니다. preprocessing_v2.py 실행...${NC}"
        python preprocessing_v2.py
    fi

    # 학습
    echo -e "${YELLOW}학습 시작...${NC}"
    if [ -f "train_lstm.py" ]; then
        python train_lstm.py
    else
        echo -e "${RED}Error: train_lstm.py가 없습니다.${NC}"
        echo "v2는 아직 학습 스크립트가 없습니다. v3를 사용하세요."
        exit 1
    fi

    cd ..
}

# v3 학습 (권장)
train_v3() {
    echo -e "${GREEN}[v3] LSTM Stride=5 + 증강 학습 시작 ⭐${NC}"
    echo "- 데이터: 마커 + 마커 사이 (Stride=5) + 데이터 증강"
    echo "- 샘플 수: 84,358개"
    echo "- 클래스: 2,152개"
    echo "- 클래스당 평균: 39.2개"
    echo ""

    cd v3

    # 전처리 데이터가 없으면 실행
    if [ ! -d "processed_data_v3" ]; then
        echo -e "${YELLOW}전처리 데이터가 없습니다. preprocessing_v3.py 실행...${NC}"
        python preprocessing_v3.py
    fi

    # 학습
    echo -e "${YELLOW}학습 시작...${NC}"
    python train_lstm.py

    cd ..
}

# Fuzzy Normalization 학습
train_fuzzy() {
    echo -e "${YELLOW}[Fuzzy] Fuzzy Normalization + LSTM 학습 시작${NC}"
    echo "- 정규화: Low/Medium/High 퍼지 멤버십"
    echo "- 입력 차원: (100, 18) - 6개 센서 × 3 fuzzy values"
    echo ""

    cd fuzzy_normalization

    # 전처리 데이터가 없으면 실행
    if [ ! -d "processed_data_v3_fuzzy" ]; then
        echo -e "${YELLOW}전처리 데이터가 없습니다. preprocessing_fuzzy.py 실행...${NC}"
        python preprocessing_fuzzy.py
    fi

    # 학습
    echo -e "${YELLOW}학습 시작...${NC}"
    if [ -f "train_lstm.py" ]; then
        python train_lstm.py
    else
        echo -e "${RED}Error: train_lstm.py가 없습니다.${NC}"
        echo "fuzzy_normalization 디렉토리에 학습 스크립트를 추가해주세요."
        exit 1
    fi

    cd ..
}

# v4 학습
train_v4() {
    echo -e "${YELLOW}[v4] LSTM 그리드 0.9m 학습 시작${NC}"
    echo "- 그리드 크기: 0.9m"
    echo "- 예상 클래스: ~538개"
    echo "- 클래스당 평균: ~156개"
    echo ""

    cd v4

    # 전처리 데이터가 없으면 실행
    if [ ! -d "processed_data_v4" ]; then
        echo -e "${YELLOW}전처리 데이터가 없습니다. preprocessing_v4.py 실행...${NC}"
        python preprocessing_v4.py
    fi

    # 학습
    echo -e "${YELLOW}학습 시작...${NC}"
    python train_lstm.py

    cd ..
}

# Fuzzy Normalization 학습
train_fuzzy() {
    echo -e "${YELLOW}[Fuzzy] Fuzzy Normalization + LSTM 학습 시작${NC}"
    echo "- 정규화: Low/Medium/High 퍼지 멤버십"
    echo "- 입력 차원: (100, 18) - 6개 센서 × 3 fuzzy values"
    echo ""

    cd fuzzy_normalization

    # 전처리 데이터가 없으면 실행
    if [ ! -d "processed_data_v3_fuzzy" ]; then
        echo -e "${YELLOW}전처리 데이터가 없습니다. preprocessing_fuzzy.py 실행...${NC}"
        python preprocessing_fuzzy.py
    fi

    # 학습
    echo -e "${YELLOW}학습 시작...${NC}"
    if [ -f "train_lstm.py" ]; then
        python train_lstm.py
    else
        echo -e "${RED}Error: train_lstm.py가 없습니다.${NC}"
        echo "fuzzy_normalization 디렉토리에 학습 스크립트를 추가해주세요."
        exit 1
    fi

    cd ..
}

# Decision Transformer 학습
train_dt() {
    echo -e "${YELLOW}[DT] Decision Transformer 학습 시작${NC}"
    echo "- 모델: GPT-style Causal Transformer"
    echo "- 논문: Decision Transformer (arXiv 2106.01345v2)"
    echo "- Context Length: 20 timesteps"
    echo ""

    # PyTorch 확인
    python -c "import torch" 2>/dev/null || {
        echo -e "${RED}Error: PyTorch가 설치되지 않았습니다.${NC}"
        echo "설치: pip install torch"
        exit 1
    }

    cd decision_transformer

    # 전처리 (v3 데이터 사용)
    if [ ! -d "../v3/processed_data_v3" ]; then
        echo -e "${YELLOW}v3 전처리 데이터가 필요합니다. v3/preprocessing_v3.py 실행...${NC}"
        cd ../v3
        python preprocessing_v3.py
        cd ../decision_transformer
    fi

    # 학습
    echo -e "${YELLOW}학습 시작...${NC}"
    python train_decision_transformer.py

    cd ..
}

# 전체 학습 (4개 모델)
train_all() {
    echo -e "${GREEN}[전체 학습] 4개 모델 학습 시작${NC}"
    echo "1. LSTM v4 (그리드 0.9m)"
    echo "2. LSTM v4 + Fuzzy"
    echo "3. Decision Transformer v2 (그리드 0.9m)"
    echo "4. Decision Transformer v2 + Fuzzy"
    echo ""

    # 1. LSTM v4
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[1/4] LSTM v4 학습${NC}"
    echo -e "${YELLOW}========================================${NC}"
    cd v4
    if [ ! -d "processed_data_v4" ]; then
        python preprocessing_v4.py
    fi
    python train_lstm.py
    cd ..

    # 2. LSTM v4 Fuzzy
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[2/4] LSTM v4 + Fuzzy 학습${NC}"
    echo -e "${YELLOW}========================================${NC}"
    cd v4_fuzzy
    if [ ! -d "processed_data_v4_fuzzy" ]; then
        python preprocessing_v4_fuzzy.py
    fi
    python train_lstm.py
    cd ..

    # 3. Decision Transformer v2
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[3/4] Decision Transformer v2 학습${NC}"
    echo -e "${YELLOW}========================================${NC}"

    python -c "import torch" 2>/dev/null || {
        echo -e "${RED}Error: PyTorch가 설치되지 않았습니다.${NC}"
        echo "설치: pip install torch"
        exit 1
    }

    if [ ! -d "v4/processed_data_v4" ]; then
        cd v4
        python preprocessing_v4.py
        cd ..
    fi

    cd decision_transformer_v2
    python train_decision_transformer.py
    cd ..

    # 4. Decision Transformer v2 Fuzzy
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[4/4] Decision Transformer v2 + Fuzzy 학습${NC}"
    echo -e "${YELLOW}========================================${NC}"

    if [ ! -d "v4_fuzzy/processed_data_v4_fuzzy" ]; then
        cd v4_fuzzy
        python preprocessing_v4_fuzzy.py
        cd ..
    fi

    cd decision_transformer_v2_fuzzy
    python train_decision_transformer.py
    cd ..

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}전체 학습 완료!${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# 메인 로직
main() {
    # 인자가 없으면 도움말
    if [ $# -eq 0 ]; then
        print_help
        exit 0
    fi

    MODEL_NUM=$1

    # 유효성 검사
    if ! [[ "$MODEL_NUM" =~ ^[1-7]$ ]]; then
        echo -e "${RED}Error: 1-7 사이의 숫자를 입력하세요.${NC}"
        print_help
        exit 1
    fi

    echo ""
    echo "======================================"
    echo "  모델 학습 시작"
    echo "======================================"
    echo ""

    # 가상환경 활성화
    activate_venv
    echo ""

    # 모델별 학습
    case $MODEL_NUM in
        1)
            train_v1
            ;;
        2)
            train_v2
            ;;
        3)
            train_v3
            ;;
        4)
            train_v4
            ;;
        5)
            train_fuzzy
            ;;
        6)
            train_dt
            ;;
        7)
            train_all
            ;;
    esac

    echo ""
    echo -e "${GREEN}======================================"
    echo "  학습 완료!"
    echo "======================================${NC}"
    echo ""
}

# 스크립트 실행
main "$@"
