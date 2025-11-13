# Version 1: Baseline (마커만 사용)

## 📋 개요
Highlighted 마커 지점만 사용한 기본 전처리 버전

## 🔧 전처리 전략
- **데이터 소스**: Highlighted=True 마커만 사용
- **Window 크기**: 100 샘플 (마커 이전 100 샘플)
- **Stride**: N/A (마커 지점만 추출)
- **데이터 증강**: 없음

## 📊 데이터셋 통계
- **총 샘플**: 12,179개
- **고유 클래스**: 1,387개 (그리드 셀)
- **클래스당 평균 샘플**: 5.2개
- **Train/Val/Test**: 48/10/11 경로

## 🔍 입력/출력 사양
- **입력 shape**: (batch, 100, 6)
  - 100 timesteps
  - 6 features: [MagX, MagY, MagZ, Pitch, Roll, Yaw]
- **출력**: 1,387개 클래스 (그리드 셀 ID)

## ❌ 문제점
- **클래스당 평균 5.2개 샘플**: 딥러닝 학습에 크게 부족
- **458개 클래스가 1-2개 샘플만 보유**: 학습 불가능
- **데이터 양 부족**: LSTM 학습에 심각하게 부족

## 📂 파일
- `preprocessing.py`: 전처리 스크립트
- `processed_data/`: 생성된 데이터셋
  - X_train.npy, y_train.npy
  - X_val.npy, y_val.npy
  - X_test.npy, y_test.npy
  - metadata.pkl

## 🔄 다음 단계
→ **v2**: 마커 사이 데이터 활용으로 샘플 수 증가 필요
