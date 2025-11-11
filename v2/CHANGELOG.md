# Version 2: 마커 사이 데이터 포함 (Stride=10)

## 📋 개요
Highlighted 마커뿐만 아니라 마커 사이의 모든 데이터도 활용 (Sliding Window)

## 🔧 전처리 전략
- **데이터 소스**: 마커 사이 모든 샘플 활용
- **Window 크기**: 100 샘플 (마커 이전 100 샘플)
- **Stride**: 10 샘플
- **데이터 증강**: 없음

## 📊 데이터셋 통계
- **총 샘플**: 38,747개 (v1 대비 3.2배 증가)
- **고유 클래스**: 2,484개 (그리드 셀)
- **클래스당 평균 샘플**: 15.6개
- **Train/Val/Test**: 48/10/11 경로

## 🔍 입력/출력 사양
- **입력 shape**: (batch, 100, 6)
  - 100 timesteps
  - 6 features: [MagX, MagY, MagZ, Pitch, Roll, Yaw]
- **출력**: 2,484개 클래스 (그리드 셀 ID)

## ⚠️ 개선점
- ✅ 샘플 수 3.2배 증가
- ✅ 클래스당 평균 15.6개로 개선

## ❌ 여전한 문제점
- **클래스당 평균 15.6개**: 여전히 목표(50-100개)에 크게 부족
- **620개 클래스가 10개 미만 샘플**: 학습 어려움
- **Stride=10이 너무 큼**: 더 촘촘한 샘플링 필요

## 📂 파일
- `preprocessing_v2.py`: 전처리 스크립트
- `processed_data_v2/`: 생성된 데이터셋
  - X_train.npy, y_train.npy
  - X_val.npy, y_val.npy
  - X_test.npy, y_test.npy
  - metadata.pkl

## 🔄 다음 단계
→ **v3**: Stride 축소 (10→5) + 보수적 데이터 증강 필요
