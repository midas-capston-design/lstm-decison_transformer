# 지자기 기반 LSTM 실내 위치 추정 시스템 프로젝트

## 📋 프로젝트 개요

### 목적
건물 내부에서 스마트폰의 지자기 센서를 활용하여 사용자의 위치를 정확하게 추정하는 딥러닝 기반 실내 위치 추정 시스템 개발

### 핵심 아이디어
- 건물 내 각 위치마다 고유한 지자기 패턴이 존재
- 사용자가 걸어갈 때 측정되는 지자기 값의 시퀀스는 위치별로 독특한 패턴을 형성
- LSTM 네트워크를 사용하여 이러한 시간적 패턴을 학습하고 위치를 분류

### 기술 스택
- **딥러닝 프레임워크**: TensorFlow / Keras
- **데이터 수집**: 스마트폰 (Android/iOS)
- **센서**: 3축 지자기 센서 (Magnetometer)
- **모델 아키텍처**: LSTM (Long Short-Term Memory)

---

## 🎯 문제 정의

### 접근 방식: 분류 문제 (Classification)

건물 내부 공간을 여러 개의 위치(Location)로 분할하고, 현재 측정되는 지자기 시퀀스가 어느 위치에 해당하는지 분류하는 문제로 정의

### 입력 (Input)
```
형태: (batch_size, sequence_length, features)

구체적 예시:
- batch_size: 32 (한 번에 학습할 샘플 수)
- sequence_length: 20 (연속된 측정값 개수)
- features: 3 (Bx, By, Bz - 3축 지자기 값)

최종 입력 shape: (32, 20, 3)
```

#### 입력 데이터 구성
```python
# 하나의 샘플 예시 (20개의 연속된 3차원 지자기 측정값)
sample = [
    [45.2, 23.1, -18.5],  # t=0: [Bx, By, Bz] (단위: μT)
    [45.5, 23.3, -18.2],  # t=1
    [45.8, 23.5, -17.9],  # t=2
    ...
    [47.1, 24.2, -16.8]   # t=19
]
```

### 출력 (Output)
```
형태: (batch_size, num_locations)

구체적 예시:
- batch_size: 32
- num_locations: 50 (건물을 50개 위치로 분할한 경우)

최종 출력 shape: (32, 50)
```

#### 출력 데이터 구성 (One-hot Encoding)
```python
# 현재 위치가 15번 위치라면
output = [0, 0, 0, ..., 1, 0, ..., 0]  # 50개 중 15번째만 1
```

---

## 🏗️ 시스템 아키텍처

### 1. 전체 시스템 구조

```
[데이터 수집] → [전처리] → [LSTM 모델] → [위치 예측]
     ↓            ↓           ↓            ↓
  스마트폰     정규화      학습/예측     위치 출력
   센서        윈도우      분류기
```

### 2. LSTM 모델 아키텍처

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    # 입력층: (batch_size, 20, 3)
    keras.layers.Input(shape=(20, 3)),
    
    # LSTM 레이어 1
    keras.layers.LSTM(40, return_sequences=True),
    
    # LSTM 레이어 2
    keras.layers.LSTM(40, return_sequences=True),
    
    # LSTM 레이어 3
    keras.layers.LSTM(40, return_sequences=True),
    
    # LSTM 레이어 4 (마지막 타임스텝만 출력)
    keras.layers.LSTM(40, return_sequences=False),
    
    # 출력층: Softmax를 사용한 분류
    keras.layers.Dense(50, activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 3. 모델 파라미터 설정 근거

| 파라미터 | 설정값 | 근거 |
|---------|--------|------|
| LSTM 레이어 수 | 4개 | 관련 연구에서 4개 레이어가 최적의 정확도와 학습 시간 균형 제공 |
| 히든 유닛 수 | 40개 | 충분한 특징 추출 능력과 과적합 방지 균형 |
| Sequence Length | 20-30개 | 위치별 패턴을 인식하기에 충분한 시간적 정보 |
| 학습 에폭 | 100-200회 | 손실 함수가 수렴하는 시점 |

---

## 📊 데이터 수집 및 준비

### 1. 건물 공간 분할

```
예시: 건물 1층을 5×10 그리드로 분할
- 총 위치 수: 50개
- 각 위치 크기: 2m × 2m
- 위치 라벨: Location 0 ~ Location 49
```

```
┌─────┬─────┬─────┬─────┬─────┐
│ L0  │ L1  │ L2  │ L3  │ L4  │
├─────┼─────┼─────┼─────┼─────┤
│ L5  │ L6  │ L7  │ L8  │ L9  │
├─────┼─────┼─────┼─────┼─────┤
│ L10 │ L11 │ L12 │ L13 │ L14 │
└─────┴─────┴─────┴─────┴─────┘
   ... (총 10행)
```

### 2. 데이터 수집 프로토콜

#### 수집 도구
- 스마트폰 (Android/iOS)
- 지자기 센서 앱 (예: Sensor Logger)

#### 수집 방법
```
1. 각 위치에서 데이터 수집
   - 각 위치당 최소 200개 샘플
   - 다양한 방향에서 수집 (동서남북)
   - 일정한 속도로 걷기 (약 1m/s)

2. 샘플링 주파수
   - 10Hz ~ 50Hz (초당 10~50회 측정)
   - 권장: 20Hz

3. 수집 시나리오
   - 위치 A → 위치 B로 이동하며 연속 측정
   - 슬라이딩 윈도우로 시퀀스 생성
```

#### 데이터 라벨링
```python
# 예시: Location 15를 지나는 경로에서 수집
데이터: [측정값 시퀀스]
라벨: 15

# 저장 형식
{
    "sequence": [[45.2, 23.1, -18.5], [45.5, 23.3, -18.2], ...],
    "label": 15,
    "timestamp": "2025-11-09 10:30:00",
    "location_name": "복도_A_15"
}
```

### 3. 데이터셋 구성

```python
# 전체 데이터셋 예시
총 샘플 수: 10,000개
- 각 위치당 200개 샘플 × 50개 위치

데이터 분할:
- 학습 데이터 (Training): 70% = 7,000개
- 검증 데이터 (Validation): 15% = 1,500개
- 테스트 데이터 (Test): 15% = 1,500개

형태:
X_train.shape = (7000, 20, 3)
y_train.shape = (7000, 50)  # one-hot encoded

X_val.shape = (1500, 20, 3)
y_val.shape = (1500, 50)

X_test.shape = (1500, 20, 3)
y_test.shape = (1500, 50)
```

---

## 💻 구현 단계

### Phase 1: 데이터 수집

```python
# 1. 스마트폰으로 지자기 데이터 수집
# 2. CSV 파일로 저장

# 데이터 형식:
# timestamp, bx, by, bz, location_id
# 2025-11-09 10:30:00.000, 45.2, 23.1, -18.5, 15
# 2025-11-09 10:30:00.050, 45.5, 23.3, -18.2, 15
```

### Phase 2: 데이터 전처리

```python
import numpy as np
import pandas as pd

def create_sequences(data, sequence_length=20):
    """
    원시 데이터를 LSTM 입력 시퀀스로 변환
    
    Args:
        data: DataFrame with columns [bx, by, bz, location_id]
        sequence_length: 시퀀스 길이
    
    Returns:
        X: (num_samples, sequence_length, 3)
        y: (num_samples, num_locations) - one-hot encoded
    """
    sequences = []
    labels = []
    
    for i in range(len(data) - sequence_length):
        # 20개의 연속된 측정값 추출
        seq = data.iloc[i:i+sequence_length][['bx', 'by', 'bz']].values
        label = data.iloc[i+sequence_length-1]['location_id']
        
        sequences.append(seq)
        labels.append(label)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    # One-hot encoding
    num_locations = len(np.unique(y))
    y_onehot = np.zeros((len(y), num_locations))
    y_onehot[np.arange(len(y)), y] = 1
    
    return X, y_onehot

# 데이터 정규화
def normalize_data(X):
    """
    지자기 데이터 정규화
    """
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1))
    
    X_normalized = (X - mean) / std
    
    return X_normalized, mean, std

# 사용 예시
df = pd.read_csv('magnetic_data.csv')
X, y = create_sequences(df, sequence_length=20)
X_norm, mean, std = normalize_data(X)
```

### Phase 3: 모델 학습

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 데이터 분할
X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 모델 구축
model = keras.Sequential([
    keras.layers.Input(shape=(20, 3)),
    keras.layers.LSTM(40, return_sequences=True),
    keras.layers.LSTM(40, return_sequences=True),
    keras.layers.LSTM(40, return_sequences=True),
    keras.layers.LSTM(40, return_sequences=False),
    keras.layers.Dense(50, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 콜백 설정
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# 학습
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=200,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {test_accuracy * 100:.2f}%")
```

### Phase 4: 실시간 예측

```python
def predict_location(model, new_sequence, mean, std, location_names):
    """
    실시간 위치 예측
    
    Args:
        model: 학습된 LSTM 모델
        new_sequence: (20, 3) 형태의 새로운 지자기 시퀀스
        mean, std: 정규화 파라미터
        location_names: 위치 이름 리스트
    
    Returns:
        predicted_location: 예측된 위치
        confidence: 예측 신뢰도
    """
    # 정규화
    new_sequence = (new_sequence - mean) / std
    
    # 배치 차원 추가
    new_sequence = np.expand_dims(new_sequence, axis=0)
    
    # 예측
    prediction = model.predict(new_sequence, verbose=0)
    
    # 가장 높은 확률의 위치
    predicted_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_idx]
    
    predicted_location = location_names[predicted_idx]
    
    return predicted_location, confidence

# 사용 예시
location_names = [f"Location_{i}" for i in range(50)]

# 새로운 20개의 측정값 (실시간으로 수집)
new_measurements = np.array([
    [46.1, 24.0, -17.2],
    [46.3, 24.1, -17.0],
    # ... 18개 더
])

location, conf = predict_location(model, new_measurements, mean, std, location_names)
print(f"예측 위치: {location}, 신뢰도: {conf*100:.2f}%")
```

---

## 📈 성능 평가

### 평가 지표

1. **정확도 (Accuracy)**
   - 전체 예측 중 정확한 예측의 비율
   - 목표: 85% 이상

2. **위치별 정확도**
   - 각 위치에서의 분류 정확도
   - 혼동 행렬(Confusion Matrix)로 시각화

3. **평균 거리 오차**
   - 예측 위치와 실제 위치 간의 물리적 거리
   - 목표: 2m 이하

### 성능 향상 기법

```python
# 1. 데이터 증강 (Data Augmentation)
def augment_sequence(sequence, noise_level=0.1):
    """지자기 데이터에 노이즈 추가"""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

# 2. 앙상블 방법
# 여러 모델의 예측을 결합하여 정확도 향상

# 3. 하이브리드 접근
# WiFi RSS, 가속도계 등 추가 센서 데이터 결합
```

---

## 🚀 배포 및 실사용

### 모바일 앱 통합

```python
# TensorFlow Lite로 모델 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 모델 저장
with open('indoor_localization.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 시스템 요구사항

- **스마트폰**: 지자기 센서 탑재
- **최소 Android 버전**: 6.0 이상
- **최소 iOS 버전**: 12.0 이상
- **모델 크기**: 약 500KB
- **예측 시간**: 50ms 이하

---

## 🔧 문제 해결 가이드

### 일반적인 문제

#### 1. 낮은 정확도
```
원인:
- 데이터 수집 부족
- 위치가 너무 많이 분할됨
- 지자기 간섭

해결:
- 각 위치당 샘플 수 증가 (200개 → 500개)
- 위치 개수 줄이기 (50개 → 30개)
- 금속 물체가 적은 환경 선택
```

#### 2. 과적합 (Overfitting)
```
원인:
- 학습 데이터 과다 학습

해결:
- Dropout 레이어 추가
- 정규화 적용 (L1, L2)
- Early Stopping 사용
```

#### 3. 위치 모호성
```
원인:
- 여러 위치가 유사한 지자기 패턴

해결:
- Sequence Length 늘리기 (20 → 30)
- Multi-scale TCN 추가
- 방향 정보 통합
```

---

## 📚 참고 문헌

### 주요 논문

1. **"Indoor Localization Using Smartphone Magnetic with Multi-Scale TCN and LSTM"**
   - Multi-scale TCN과 LSTM 결합 아키텍처
   - 다양한 이동 속도 대응

2. **"DeepML: Deep LSTM for Indoor Localization with Smartphone Magnetic and Light Sensors"**
   - 지자기 + 조도 센서 융합
   - 4-layer LSTM 아키텍처

3. **"A Hierarchical LSTM-Based Indoor Geomagnetic Localization Algorithm"**
   - 계층적 LSTM 구조
   - 위치 모호성 해결

### 핵심 인사이트

- **시간적 패턴의 중요성**: 단일 측정값이 아닌 시퀀스가 핵심
- **LSTM의 효과**: 시간적 의존성을 잘 포착
- **하이브리드 접근**: 다중 센서 융합으로 정확도 향상

---

## 📋 체크리스트

### 프로젝트 진행 체크리스트

- [ ] **1단계: 계획 및 준비**
  - [ ] 건물 평면도 확보
  - [ ] 위치 분할 계획 수립
  - [ ] 데이터 수집 도구 준비

- [ ] **2단계: 데이터 수집**
  - [ ] 각 위치에서 200개 이상 샘플 수집
  - [ ] 다양한 방향에서 수집
  - [ ] 데이터 품질 확인

- [ ] **3단계: 데이터 전처리**
  - [ ] 시퀀스 생성 코드 작성
  - [ ] 정규화 적용
  - [ ] Train/Val/Test 분할

- [ ] **4단계: 모델 개발**
  - [ ] LSTM 모델 구축
  - [ ] 학습 및 검증
  - [ ] 하이퍼파라미터 튜닝

- [ ] **5단계: 평가 및 최적화**
  - [ ] 테스트 데이터 평가
  - [ ] 성능 분석
  - [ ] 개선 작업

- [ ] **6단계: 배포**
  - [ ] 모델 경량화 (TFLite)
  - [ ] 모바일 앱 통합
  - [ ] 실사용 테스트

---

## 🔮 향후 개선 방향

### 단기 개선
1. **Multi-scale TCN 추가**
   - 다양한 이동 속도 대응
   - 특징 차원 확장

2. **센서 융합**
   - WiFi RSS 추가
   - 조도 센서 활용

3. **방향 정보 통합**
   - 자이로스코프 데이터 활용
   - 방향별 모델 학습

### 장기 개선
1. **전이 학습 (Transfer Learning)**
   - 다른 건물에 모델 재사용
   - 적은 데이터로 빠른 적응

2. **실시간 맵핑**
   - SLAM 기법 통합
   - 동적 환경 대응

3. **사용자 경험 최적화**
   - 배터리 소모 최소화
   - 예측 속도 향상

---

## 📞 문의 및 지원

### 개발 환경 설정 문의
- Python 버전: 3.8 이상
- TensorFlow 버전: 2.10 이상
- 필수 라이브러리: numpy, pandas, scikit-learn

### 추가 자료
- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [LSTM 튜토리얼](https://www.tensorflow.org/guide/keras/rnn)
- [실내 위치 추정 리뷰 논문](https://link.springer.com/)

---

**문서 버전**: 1.0  
**최종 수정일**: 2025-11-09  
**작성자**: Claude AI Assistant
