# 🎯 지자기 기반 실내 위치 추정 프로젝트 전략

## 📊 데이터 구조 분석 결과

### 데이터셋
- **187개 CSV 파일** (69개 경로 × 평균 2-3회)
- **29개 노드** (x, y 좌표)
- **평균 2,220행/파일**, 47.6 Hz 샘플링
- **센서**: 지자기(MagX/Y/Z), 가속도, 자이로, 자세

### 핵심 발견: `Highlighted` 마커
- **Ground Truth 라벨!**
- 각 노드 통과 시 약 9-10번씩 표시
- 예: 11→1 경로 (11개 노드) → 100개 마커

---

## ⚠️ 핵심 문제점

### 문제: 경로 중복 (Route Overlap)
```
1→11: 노드 [1,2,3,4,5,6,7,8,9,10,11]
2→12: 노드 [2,3,4,5,6,7,8,9,10,11,12]

중복: [2,3,4,5,6,7,8,9,10,11] = 10개 노드
```

→ 같은 지자기 시퀀스가 다른 경로 라벨을 가질 수 있음

### 해결: 문제 재정의
❌ **나쁜 접근**: "어느 경로인가?" (1→11 vs 2→12)
✅ **좋은 접근**: "현재 어느 노드에 있는가?" (1~29)

---

## 🚀 두 가지 접근법

### 방법 1: LSTM 기반 노드 위치 분류

**문제 정의**
- 입력: 지자기 시퀀스 `[MagX, MagY, MagZ]` × K timesteps
- 출력: 29개 노드 중 현재 위치

**라벨링 방법 (Highlighted 마커 활용)**
1. Highlighted=True 마커들을 노드 경계로 사용
2. 11→1 경로: 100개 마커 → 10개씩 그룹화 → 11개 노드
3. 각 그룹에 해당 노드 ID 할당

**데이터 전처리**
```python
# 예시: 11→1 경로 (노드 11,10,9,8,7,6,5,4,3,2,1)
마커 0-9    → 노드 11
마커 10-19  → 노드 10
마커 20-29  → 노드 9
...
마커 90-99  → 노드 1

# Sliding window로 시퀀스 생성
sequence_length = 20  # 0.4초 @ 50Hz
overlap = 10          # 50% 중복
```

**모델 아키텍처**
```python
Input: (batch, 20, 3)          # 20 timesteps × [MagX,Y,Z]
LSTM(40 units) × 4 layers
Dense(29, softmax)
Output: (batch, 29)            # 29개 노드 확률
```

**장점**
- ✅ 중복 구간 문제 해결 (같은 노드 = 같은 라벨)
- ✅ 단순하고 검증된 방법
- ✅ Highlighted 마커로 정확한 라벨 확보

---

### 방법 2: Decision Transformer

**문제 정의**
- 목표: 시작 → 목표 최단 경로 찾기
- 입력: `(R̂_t, s_t, a_t)` 시퀀스
  - `R̂_t`: Return-to-go (목표까지 남은 거리)
  - `s_t`: State (지자기 + 노드 ID)
  - `a_t`: Action (다음 노드)
- 출력: 다음 action (어느 노드로 이동할지)

**Return-to-go 정의**
```python
# 옵션 1: 물리적 거리 (m)
R̂_t = distance(current_node, goal_node)

# 옵션 2: 남은 홉 수
R̂_t = num_hops_to_goal

# 옵션 3: 음수 시간 (초)
R̂_t = -remaining_time_to_goal
```

**State 정의**
```python
s_t = {
    'mag': [MagX, MagY, MagZ],       # 3차원
    'node_id': current_node,          # 1차원 (optional)
    'position': (x, y)                # 2차원 (optional)
}
```

**Action 정의**
```python
a_t = next_node_id  # 다음 방문할 노드 (0~28)
```

**데이터 전처리**
```python
# 11→1 경로 예시
trajectory = {
    'returns': [10, 9, 8, ..., 1, 0],     # 남은 홉 수
    'states': [mag_seq_11, mag_seq_10, ...],  # 각 노드의 지자기
    'actions': [10, 9, 8, ..., 1]          # 다음 노드
}
```

**모델 아키텍처**
```python
Input: (R̂, s, a) × K timesteps
Embedding(R̂) + Embedding(s) + Embedding(a) + PositionalEncoding
GPT Transformer (causal masking)
Linear(action_dim)
Output: next_action
```

**장점**
- ✅ Context-aware (시작점 고려)
- ✅ Goal-conditioned (목표 명시 가능)
- ✅ 최적 경로 학습 가능
- ✅ 중복 구간에서도 목표를 고려하여 올바른 action 선택

---

## 📈 평가 지표

### 공통 지표
1. **정확도 (Accuracy)**: 전체 예측 중 정확한 비율
2. **혼동 행렬 (Confusion Matrix)**: 노드별 분류 성능
3. **평균 거리 오차**: 예측 노드 ↔ 실제 노드 물리적 거리

### Decision Transformer 추가
4. **경로 정확도**: 전체 경로가 정확한 비율
5. **최단 경로 비율**: 최적 경로를 찾은 비율

---

## 🔧 구현 우선순위

### Phase 1: LSTM Baseline (필수) ⭐⭐⭐
1. Highlighted 마커 기반 라벨링
2. 시퀀스 데이터 생성
3. LSTM 모델 학습
4. 성능 평가

**예상 소요 시간**: 2-3일

### Phase 2: Decision Transformer (옵션) ⭐⭐
1. Return-to-go 계산
2. Trajectory 데이터 구성
3. Transformer 모델 구현
4. 비교 평가

**예상 소요 시간**: 3-4일

---

## 💾 데이터 분할

```python
# 전체: 187개 파일
# Train/Val/Test = 70/15/15

Train: 131개 파일 (~48개 경로)
Val:   28개 파일  (~10개 경로)
Test:  28개 파일  (~11개 경로)

# 경로 기반 분할 (같은 경로가 다른 set에 안 들어가도록)
```

---

## 📝 다음 단계

1. ✅ 데이터 분석 완료
2. 🔄 **[진행중]** Highlighted 마커 기반 라벨링 전략 수립
3. ⏳ LSTM 데이터 전처리 구현
4. ⏳ LSTM 모델 구축 및 학습
5. ⏳ (Optional) Decision Transformer 구현
6. ⏳ 성능 비교 및 분석

---

**작성일**: 2025-11-09
**상태**: 전략 수립 완료, 구현 준비 중
