# Decision Transformer v1 개요

`dt/decision_transformer/` 폴더는 Stride=5(LSTM v3) 데이터로 학습하는 DT 기본 모델을 담고 있습니다. 입력 센서는 지자기 3축 + 자세 3축이며, 목표까지의 return-to-go와 다음 위치(액션)를 번갈아가며 Transformer에 넣어 경로를 생성합니다.

## 핵심 개념 정리

| 구분 | 기존 LSTM 분류 | Decision Transformer |
|------|---------------|---------------------|
| 목적 | 현재 위치 분류 | 목표를 주면 경로 생성 |
| 입력 | 센서 시퀀스 | (Return-to-go, State, Action) 반복 |
| 학습 | Cross-Entropy | 시퀀스 예측 (MSE) |
| 장점 | 단순 | 목표 조건 일반화, 장기 의존성 처리 |

- **State(sₜ)**: 센서 6차원 (MagX/Y/Z, Pitch/Roll/Yaw)
- **Action(aₜ)**: 다음 위치 변화량(dx, dy)
- **Return(R̂ₜ)**: 목표까지 남은 거리(0에 가까울수록 목표 근처)
- **Context Length**: 20 스텝(약 0.4 초 @ 50 Hz)

## 모델 구성
```
(R̂ₜ, sₜ, aₜ) → 임베딩/포지셔널 인코딩 → GPT 계열 Causal Transformer → Action Head
```
출력은 `â_{t+1}`(다음 위치)이며, 학습 시 MSE로 지도학습을 수행합니다.

## 실행 방법
```bash
cd dt/decision_transformer
python train_decision_transformer.py
```
필요 전처리: `lstm/v3/preprocessing_v3.py` → `dt/preprocessing_decision_transformer.py`.

## 주요 하이퍼파라미터
```python
CONFIG = {
    "context_length": 20,
    "n_layer": 3,
    "n_head": 4,
    "n_embd": 128,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 100,
}
```

## 앞으로의 체크리스트
- LSTM 분류 대비 정확도/경로 품질 비교
- Return-to-go 계산 로직 고도화
- 실시간 추론 CLI/REST 엔드포인트 구현
- Context length, embedding 차원 등 ablation

논문: *Decision Transformer: Reinforcement Learning via Sequence Modeling* (arXiv:2106.01345). 필요한 부분은 이 구현에 맞게 변형해 사용했습니다.
