# Indoor Positioning with Hyena - Project Summary

**Date**: 2025-01-21
**Goal**: 실시간 실내 측위 시스템 (Real-time indoor positioning)

---

## 📊 Current Status

### Dataset
- **Original**: 203 raw files
- **Added**: 271 good bad files (calibration corrected)
- **Total**: **474 CSV files** in `data/raw/`
- **Sampling rate**: 50Hz
- **Average length**: 500-3000 timesteps

### Data Quality Analysis Completed
- ✅ Bad 데이터 분석 완료 (`analyze_file_quality.py`)
- ✅ 58개 raw-style bad files → 직접 복사
- ✅ 213개 bad-style files → -40.3μT offset 보정 후 복사
- ✅ 38개 low-quality files 제외 (길이 < 500, 불안정)

---

## 🎯 Problem Definition

### Use Case
**실시간 보행자 추적** - 걸으면서 매 걸음(~50 timesteps)마다 위치 업데이트

### Requirements
1. **최소 context**: 250 timesteps부터 예측 가능
2. **정확도 증가**: Context 누적되면서 정확도 향상 (250 → 500 → 1000)
3. **Causal**: Position[t] 예측 시 과거 데이터(0:t)만 사용
4. **실시간**: 매 50 timesteps마다 위치 출력

---

## 🔧 Architecture Decision

### ❌ 이전 방식 (Seq2seq Full Sequence)
```python
# Training
Input: sensors[0:T] (전체 시퀀스, e.g., 2000 timesteps)
Output: positions[0:T] (전체 경로)
Problem: Non-causal (미래 센서도 사용 가능)

# Inference
Input: sensors[window] (고정 250 window)
Output: position[last]
Problem: Train-test mismatch!
```

### ✅ 새로운 방식 (Sliding Window Causal)
```python
# Training
Input: sensors[t-249:t] (250 window)
Output: position[t] (마지막 위치만)
→ Causal: 과거만 사용

# Inference
Same as training!
→ Train-test 일치
```

**핵심 변경점:**
- Full sequence → Sliding window (250, stride 50)
- Full trajectory → Single position (마지막 timestep)
- Non-causal → Causal (과거만 사용)

---

## 📁 File Structure

### 새로 추가된 파일
```
src/
├── preprocess_sliding.py    # Sliding window 전처리
└── train_sliding.py          # Causal training

scripts/
└── run_all.sh               # 3가지 feature 모드 비교 실험 (업데이트됨)

move_good_bad_to_raw.py      # Bad 파일 → Raw 이동 (완료)
analyze_file_quality.py       # 파일 품질 분석 (완료)
fundamental_analysis.py       # Bad vs Raw 근본 분석 (완료)
```

### 기존 파일 (유지)
```
src/
├── pipeline.py              # 기존 Seq2seq 방식 (참고용)
├── model.py                 # Hyena 모델
└── dataset.py               # Dataset 클래스들
```

---

## 🚀 How to Run

### 방법 1: 전체 실험 (추천)
```bash
# 3가지 feature 모드 비교 (mag3, mag4, full)
./scripts/run_all.sh
```

**실행 내용:**
1. **MAG3** (MagX, MagY, MagZ) - 3 features
2. **MAG4** (MagX, MagY, MagZ, Magnitude) - 4 features
3. **FULL** (MagX, MagY, MagZ, Pitch, Roll, Yaw) - 6 features

**출력:**
- `checkpoints_sliding_mag3/best.pt`
- `checkpoints_sliding_mag4/best.pt`
- `checkpoints_sliding_full/best.pt`

### 방법 2: 단일 Feature 모드
```bash
# 1. 전처리
python3 src/preprocess_sliding.py \
  --raw-dir data/raw \
  --nodes data/nodes_final.csv \
  --output data/sliding_mag3 \
  --feature-mode mag3 \
  --window-size 250 \
  --stride 50

# 2. 학습
python3 src/train_sliding.py \
  --data-dir data/sliding_mag3 \
  --epochs 50 \
  --batch-size 32 \
  --hidden-dim 256 \
  --depth 8
```

---

## 📦 Data Format

### Preprocessing Output
```python
# data/sliding_mag3/train.jsonl
{"features": [[f1, f2, f3], ...250 rows], "target": [x_norm, y_norm]}
{"features": [[f1, f2, f3], ...250 rows], "target": [x_norm, y_norm]}
...

# Shapes
X_train: [N_train, 250, 3]
y_train: [N_train, 2]
```

### Feature Modes
| Mode  | Features | Dim |
|-------|----------|-----|
| mag3  | MagX, MagY, MagZ | 3 |
| mag4  | MagX, MagY, MagZ, Magnitude | 4 |
| full  | MagX, MagY, MagZ, Pitch, Roll, Yaw | 6 |

### Normalization
```python
# Magnetometer
BASE_MAG = (-33.0, -15.0, -42.0)
mag_norm = (mag - base) / 10.0

# Magnitude
mag_magnitude = sqrt(MagX² + MagY² + MagZ²)
mag_magnitude_norm = (mag_magnitude - 50.0) / 10.0

# Coordinates
COORD_CENTER = (-41.0, 0.0)
COORD_SCALE = 50.0
x_norm = (x - center_x) / scale
y_norm = (y - center_y) / scale
```

---

## 🧠 Model Architecture

### Hyena Positioning
```python
HyenaPositioning(
    input_dim=3,        # Feature 개수
    hidden_dim=256,     # Hyena hidden dimension
    output_dim=2,       # (x, y)
    depth=8,            # Hyena layers
    dropout=0.1
)
```

### Training Configuration
```python
Optimizer: AdamW (lr=2e-4, weight_decay=0.01)
Scheduler: CosineAnnealingLR
Loss: MSE
Batch size: 32
Epochs: 50
Early stopping: patience=10 (RMSE 기준)
```

### Forward Pass
```python
# Input: [batch, 250, n_features]
# Output: [batch, 250, 2]
# Loss: Only last timestep
pred = model(features, edge_ids)[:, -1, :]  # [batch, 2]
loss = MSE(pred, target)
```

---

## 📈 Expected Results

### Metrics
- **RMSE**: Root Mean Square Error (m)
- **MAE**: Mean Absolute Error (m)
- **Median**: Median error (m)
- **P90**: 90th percentile error (m)

### Evaluation
- Train: Full sequence loss + last position distance
- Val: Same as training (causal)
- Test: Same as training (causal)

**Key**: Train/Val/Test 모두 동일한 방식으로 평가

---

## 🔍 Analysis Scripts (참고용)

### 실행 완료된 분석
```bash
# 1. Bad 데이터 품질 분석
python3 analyze_file_quality.py
# Output: good_bad_files.txt, raw_style_bad_files.txt, exclude_files.txt

# 2. Bad vs Raw 근본 분석
python3 fundamental_analysis.py
# Conclusion: Sensor calibration offset (40.3μT)

# 3. Bad 파일 전처리 및 이동
python3 move_good_bad_to_raw.py
# Result: 474 files in data/raw/
```

---

## 🚨 Important Notes

### Causal Training의 중요성
```python
# ❌ 잘못된 방식 (Non-causal)
Position[100] 예측 시 Sensors[0:2000] 모두 사용
→ 실제 추론 때는 Sensors[0:100]만 있음
→ Train-test mismatch!

# ✅ 올바른 방식 (Causal)
Position[100] 예측 시 Sensors[0:100]만 사용
→ 실제 추론 때도 Sensors[0:100]만 사용
→ Train-test 일치!
```

### Hyena의 역할
- Long-range dependency 학습
- 250 window 내에서도 장기 패턴 포착
- FFT 기반 효율적 long convolution
- 더 긴 context (500, 1000) 사용 시 정확도 향상 기대

### Context Length Strategy
```
t=250:  context[0:250]   → RMSE 높음 (최소 context)
t=500:  context[0:500]   → RMSE 중간
t=1000: context[0:1000]  → RMSE 낮음 (충분한 context)
```

사용자는 초기엔 부정확하지만 빠른 피드백, 시간 지나면서 정확도 향상.

---

## 📋 Next Steps

### 즉시 실행
```bash
# 실험 시작
./scripts/run_all.sh

# 예상 시간: ~2-3시간 (3개 모델)
# GPU 권장
```

### 실험 후 분석
1. 3개 모델 성능 비교 (mag3 vs mag4 vs full)
2. Pitch/Roll/Yaw의 실제 기여도 확인
3. Best model 선택

### 향후 개선 (필요시)
1. **Expanding window 실험**
   - 현재: 고정 250 window
   - 개선: 250 → 500 → 1000 expanding
   - 예상: 정확도 시간에 따라 향상

2. **Data augmentation**
   - Noise injection
   - Time warping
   - Mixup

3. **Hyperparameter tuning**
   - Hidden dim: 128, 256, 512
   - Depth: 6, 8, 10
   - Window size: 200, 250, 300

---

## 💾 Data Backup

### Raw Data
- `data/raw/`: 474 CSV files (원본 + 보정된 bad)
- `data/nodes_final.csv`: 노드 위치 정보

### Generated Files (재생성 가능)
- `data/sliding_mag3/`: MAG3 전처리 결과
- `data/sliding_mag4/`: MAG4 전처리 결과
- `data/sliding_full/`: FULL 전처리 결과
- `checkpoints_sliding_*/`: 학습된 모델

### Analysis Results
- `good_bad_files.txt`: 271개
- `raw_style_bad_files.txt`: 58개
- `exclude_files.txt`: 38개

---

## 🔑 Key Decisions Log

### 1. Dataset Expansion
- **Decision**: Bad 폴더 271개 파일 추가
- **Method**: Calibration offset correction (-40.3μT)
- **Result**: 203 → 474 files (133% increase)

### 2. Architecture Change
- **Decision**: Seq2seq → Sliding Window
- **Reason**: Real-time tracking requires causal inference
- **Impact**: Train-test alignment

### 3. Feature Selection (실험 중)
- **Options**: mag3 (3), mag4 (4), full (6)
- **Hypothesis**: Pitch/Roll은 노이즈, Yaw는 유용할 수 있음
- **Pending**: 실험 결과 확인

### 4. Window Size
- **Decision**: 250 timesteps (고정)
- **Reason**: 최소 의미 있는 context, 5초 (50Hz)
- **Future**: Expanding window 고려

### 5. Validation Method
- **Decision**: Full sequence (reverted back)
- **Reason**: Sliding window validation은 너무 느림
- **Note**: Test는 sliding window stride=50

---

## 📞 Contact & References

### Key Files to Check
- `src/preprocess_sliding.py`: 전처리 로직
- `src/train_sliding.py`: 학습 로직
- `src/model.py`: Hyena 모델
- `scripts/run_all.sh`: 실행 스크립트

### Debug Commands
```bash
# 전처리 결과 확인
head -1 data/sliding_mag3/train.jsonl | python3 -m json.tool

# 메타데이터 확인
cat data/sliding_mag3/meta.json

# 학습 재개 (체크포인트에서)
# → train_sliding.py에 resume 기능 추가 필요
```

---

## ✅ Checklist

### 완료
- [x] Bad 데이터 분석
- [x] 데이터셋 증가 (474 files)
- [x] Sliding window 전처리 구현
- [x] Causal training 구현
- [x] Feature 모드 3가지 지원
- [x] 실행 스크립트 작성

### 진행 중
- [ ] 실험 실행 (mag3 vs mag4 vs full)
- [ ] 성능 비교 및 분석

### 향후
- [ ] Expanding window 구현 (필요시)
- [ ] Best model inference 스크립트
- [ ] Real-time demo

---

**마지막 업데이트**: 2025-01-21
**다음 세션 시작**: `./scripts/run_all.sh` 실행 또는 결과 분석
