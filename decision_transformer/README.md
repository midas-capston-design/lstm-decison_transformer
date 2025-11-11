# Decision Transformer for Indoor Localization

## 개요
강화학습을 시퀀스 모델링 문제로 재구성한 Decision Transformer를 실내 위치 추정에 적용

## 참고 논문
- **Title**: Decision Transformer: Reinforcement Learning via Sequence Modeling
- **Authors**: Lili Chen, Kevin Lu, et al. (UC Berkeley)
- **arXiv**: 2106.01345v2
- **URL**: https://arxiv.org/abs/2106.01345

## 핵심 아이디어

### 기존 RL vs Decision Transformer
| 방식 | 기존 RL (LSTM) | Decision Transformer |
|------|---------------|---------------------|
| 문제 정의 | 분류 (현재 위치) | 시퀀스 생성 (목표까지 경로) |
| 입력 | 센서 시퀀스 | (Return-to-go, State, Action) 시퀀스 |
| 학습 | TD-learning, Value function | Supervised sequence modeling |
| 추론 | 현재 센서 → 위치 | 목표 설정 → 최적 경로 생성 |

### Trajectory Representation
```
τ = (R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₜ, sₜ, aₜ)
```

- **R̂ₜ**: Return-to-go (목표까지 남은 거리)
- **sₜ**: State (센서 값: MagX/Y/Z, Pitch/Roll/Yaw)
- **aₜ**: Action (다음 위치: dx, dy)

### Architecture
```
Input: (R̂ₜ, sₜ, aₜ) for t ∈ [1, K]
       ↓
  Linear Embeddings + Positional Encoding
       ↓
  GPT-style Causal Transformer
       ↓
  Action Prediction Head
       ↓
Output: â_{t+1} (다음 위치)
```

## 우리 프로젝트 적용

### 문제 설정
- **State**: 지자기 + 자세 센서 (6차원)
- **Action**: 다음 그리드 위치 (2차원: x, y)
- **Return**: 목표까지의 거리 (음수, 가까울수록 0에 가까움)

### Context Length (K)
- 20 timesteps (약 0.4초 @ 50Hz)
- 충분히 긴 컨텍스트로 패턴 학습

### 학습 방식
1. Offline trajectory 수집 (law_data)
2. (R, s, a) 시퀀스로 변환
3. Causal transformer로 다음 action 예측
4. MSE loss로 supervised learning

## 장점

### vs LSTM 분류
1. **목표 지향적**: "여기로 가고 싶어" → 경로 생성
2. **Credit Assignment**: Attention으로 장기 의존성 학습
3. **일반화**: 다양한 목표에 대해 하나의 모델

### vs TD-learning
1. **No Bootstrapping**: Value function 불필요
2. **No Discounting**: 장기 목표 유지
3. **Stable**: Deadly triad 회피

## 사용법

```bash
# 1. PyTorch 설치
pip install torch

# 2. 학습
cd decision_transformer
python train_decision_transformer.py

# 3. 추론 (구현 예정)
python infer.py --target_x 10.0 --target_y 5.0
```

## 하이퍼파라미터

```python
CONFIG = {
    'context_length': 20,    # K
    'n_layer': 3,            # Transformer layers
    'n_head': 4,             # Attention heads
    'n_embd': 128,           # Embedding dimension
    'dropout': 0.1,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 100,
}
```

## 성능 평가 (TODO)
- [ ] LSTM vs Decision Transformer 비교
- [ ] 목표 변경에 대한 일반화 능력
- [ ] Long-term credit assignment 평가
- [ ] 실시간 추론 속도 측정

## 다음 단계
1. 정확한 좌표 라벨링 (현재는 임시)
2. Return-to-go 계산 로직 개선
3. 추론 스크립트 작성
4. Ablation study (context length, architecture 등)
5. Online fine-tuning 실험

---
**참고**: 이 구현은 논문의 핵심 아이디어를 따르되, 실내 위치 추정 문제에 맞게 수정되었습니다.
