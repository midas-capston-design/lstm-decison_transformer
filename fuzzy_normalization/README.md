# Fuzzy Normalization for Indoor Localization

## 개요
퍼지 로직 기반 정규화를 사용한 지자기 센서 데이터 전처리

## 핵심 아이디어
기존의 min-max나 z-score 정규화 대신, 각 센서 값을 **Low/Medium/High** 3개의 퍼지 멤버십 함수로 변환합니다.

### Fuzzy Membership Functions
- **Low**: 낮은 값 (0-50 percentile)
- **Medium**: 중간 값 (25-75 percentile)
- **High**: 높은 값 (50-100 percentile)

각 센서 값은 3개의 멤버십 값으로 표현됩니다:
- 원본: `MagX = 25.3`
- Fuzzy: `[low=0.3, medium=0.7, high=0.0]`

## 장점
1. **노이즈 강건성**: 극값에 덜 민감
2. **해석 가능성**: "지자기가 중간 정도로 높음" 등 직관적
3. **비선형 패턴 포착**: 선형 정규화보다 풍부한 표현

## 입출력
- **입력**: (batch, 100, 6) - 6개 센서 (MagX/Y/Z, Pitch/Roll/Yaw)
- **출력**: (batch, 100, 18) - 각 센서 × 3 fuzzy values

## 사용법
```bash
cd fuzzy_normalization
python preprocessing_fuzzy.py
```

## 출력
- `processed_data_v3_fuzzy/X_train.npy` - 학습 데이터
- `processed_data_v3_fuzzy/y_train.npy` - 학습 라벨
- `processed_data_v3_fuzzy/metadata.pkl` - 메타데이터

## 다음 단계
- v3 LSTM 학습 스크립트를 fuzzy 데이터로 학습
- 일반 정규화와 성능 비교
