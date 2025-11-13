# 지자기 기반 LSTM 실내 위치 추정 프로젝트

## 📋 프로젝트 개요
스마트폰의 지자기 센서와 자세 센서를 활용하여 실내 위치를 추정하는 딥러닝 프로젝트

## 🎯 핵심 전략
- **문제 정의**: 그리드 기반 분류 (0.45m × 0.45m 그리드)
- **입력**: 지자기(MagX/Y/Z) + 자세(Pitch/Roll/Yaw), 100 timesteps (2초)
- **출력**: 2,152개 그리드 셀 중 현재 위치 분류
- **모델**: 4-layer LSTM

## 📂 프로젝트 구조
```
.
├── law_data/              # 원본 센서 데이터 (187개 CSV)
│   └── {start}_{end}_{trial}.csv
├── lstm/
│   ├── v1/                # Version 1: 마커만 사용 (Baseline)
│   ├── v2/                # Version 2: Stride=10
│   ├── v3/ ⭐             # Version 3: Stride=5 + 증강
│   ├── v4/, v4_fuzzy/     # 0.9m grid 변형
│   ├── fuzzy_normalization/
│   └── train.sh           # 통합 학습 스크립트
├── archive/               # 분석 스크립트 및 중간 결과물
├── nodes_final.csv        # 29개 노드 좌표 (참고용)
├── FINAL_STRATEGY.md      # 최종 전처리 전략 문서
└── README.md              # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tqdm
```

### 2. 데이터 전처리 (v3 - 현재 버전)
```bash
cd lstm/v3
python preprocessing_v3.py
```

**현재 결과:**
- 총 샘플: 84,358개
- 클래스: 2,152개 그리드 셀
- 클래스당 평균: 39.2개 샘플
- Train/Val/Test: 60,524 / 10,686 / 13,148

### 3. LSTM 모델 학습
```bash
cd lstm/v3
python train_lstm.py
```

**모델 구조:**
- LSTM(128) → LSTM(256) → LSTM(256) → LSTM(128)
- Batch Normalization + Dropout (0.3)
- Dense(256) → Dense(128) → Softmax(2152)
- 총 파라미터: 1,432,288개 (5.46 MB)

## 📊 데이터셋 정보

### 원본 데이터
- **파일 개수**: 187개 CSV
- **경로**: 69개 서로 다른 노드 쌍
- **샘플링 레이트**: 평균 47.6 Hz
- **Ground Truth**: Highlighted 마커 (0.45m 간격)

### 센서 정보
| 센서 | 사용 여부 | 용도 |
|------|----------|------|
| MagX, MagY, MagZ | ✅ | 위치 정보 (방향 의존적) |
| Pitch, Roll, Yaw | ✅ | 자세/방향 정보 |
| AccX, AccY, AccZ | ❌ | 노이즈 과다 |
| GyroX, GyroY, GyroZ | ❌ | 드리프트 문제 |

### 버전별 데이터 품질 비교

| 항목 | v1 (Baseline) | v2 (Stride=10) | v3 (현재) |
|------|---------------|----------------|-----------|
| 총 샘플 | 12,179 | 38,747 | 84,358 |
| 클래스 수 | 1,387 | 2,484 | 2,152 |
| 클래스당 평균 | 5.2개 | 15.6개 | 39.2개 |
| 학습 가능성 | ❌ | ⚠️ | ✅ |

## 🔍 주요 발견 사항

### 1. 방향 의존성 문제
- **문제**: 1→11과 11→1은 같은 위치인데 지자기 값이 완전히 다름
- **원인**: 스마트폰 방향이 180° 반대
- **해결**: 자세 정보(Pitch/Roll/Yaw)를 함께 입력으로 사용

### 2. 경로 중복 문제
- **문제**: 1→11과 2→12는 10개 노드를 공유 → 같은 지자기 시퀀스가 다른 라벨
- **해결**: 문제를 "절대 위치 예측"으로 재정의 (그리드 분류)

### 3. 데이터 부족 문제
- **v1**: 마커만 사용 → 클래스당 5.2개 (학습 불가)
- **v2**: 마커 사이 데이터 추가 (Stride=10) → 15.6개 (부족)
- **v3**: Stride=5 + 보수적 증강 → 39.2개 (학습 가능!)

## 🎓 학습 전략

### 입력 Window
- **크기**: 100 샘플 (2초, ~2걸음)
- **방식**: Causal (과거 데이터만 사용)
- **실시간**: 앱 시작 후 2초부터 예측 가능

### 데이터 증강
- **비율**: 30% 샘플만 증강
- **지자기 노이즈**: Gaussian(μ=0, σ=0.8μT)
- **방향 노이즈**: Gaussian(μ=0, σ=1.5°)
- **철학**: 센서 노이즈 수준만 적용 (보수적)

### 클래스 필터링
- **기준**: 10개 미만 샘플 클래스 제거
- **이유**: 학습 불가능한 클래스 제외
- **결과**: 2,484 → 2,152 클래스

## 📈 버전 히스토리

### v1: Baseline (마커만 사용)
- Highlighted 마커 지점만 사용
- **문제**: 데이터 양 절대적 부족 (클래스당 5.2개)

### v2: 마커 사이 데이터 포함
- Sliding Window (Stride=10)
- **개선**: 샘플 3.2배 증가
- **한계**: 여전히 부족 (클래스당 15.6개)

### v3: Stride 축소 + 증강 (현재) ⭐
- Stride=5 + 보수적 데이터 증강
- **결과**: 클래스당 39.2개 → LSTM 학습 가능

## 🔬 실시간 추론 시나리오

```
앱 시작 (t=0초)
  ↓
센서 수집 시작
  ↓
t=2초 (100 샘플): 첫 예측 가능 (불안정)
  ↓
t=3초: 예측 업데이트
  ↓
t=5초 (5걸음): 안정적 예측
  ↓
계속 사용: Sliding Window로 실시간 업데이트
```

## 📝 참고 문서
- `FINAL_STRATEGY.md`: 센서 신뢰도 분석 및 전처리 전략
- `v1/CHANGELOG.md`: v1 상세 정보
- `v2/CHANGELOG.md`: v2 상세 정보
- `v3/CHANGELOG.md`: v3 상세 정보

## 🛠️ 진행 상황
1. 데이터 탐색 및 분석
2. 센서 신뢰도 검증
3. 전처리 전략 수립
4. v1, v2, v3 버전별 데이터 전처리
5. LSTM 모델 구축
6. 모델 학습 및 평가 (사용자가 직접 실행)

## 🔬 추가 실험 방법

### 1. Fuzzy Normalization (퍼지 정규화)
```bash
cd fuzzy_normalization
python preprocessing_fuzzy.py
```
- 각 센서 값을 Low/Medium/High 3개 멤버십 함수로 변환
- 노이즈에 강건하고 해석 가능한 특징 표현
- 입력 차원: (100, 6) → (100, 18)

### 2. Decision Transformer (강화학습 기반)
```bash
cd decision_transformer
pip install torch  # PyTorch 필요
python train_decision_transformer.py
```
- 논문: "Decision Transformer: RL via Sequence Modeling" (2106.01345v2)
- Return-to-go conditioning으로 목표 지향적 경로 생성
- GPT-style causal transformer 사용
- TD-learning 없이 supervised learning만으로 학습

---
**마지막 업데이트**: 2025-11-09
