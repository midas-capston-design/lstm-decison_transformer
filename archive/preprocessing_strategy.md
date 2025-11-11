# 🔧 데이터 전처리 전략

## 📊 Ground Truth 정의

**확인된 사실**:
- `Highlighted=True` 마커 = 0.45m 간격의 정확한 위치
- 각 마커 간 거리: **정확히 0.45m**
- nodes_final.csv: 참고용 (정확하지 않음)

---

## 🎯 문제 정의 옵션

### 옵션 A: 절대 좌표 회귀 (Regression)
```
입력: 지자기 시퀀스 [MagX, MagY, MagZ] × K timesteps
출력: 절대 좌표 (x, y) - 연속값

예: 현재 위치 = (-22.5m, 0m)
```

**장점**:
- ✅ 중복 구간 문제 완전 해결
- ✅ 연속적인 위치 예측
- ✅ 새로운 경로에도 일반화 가능

**단점**:
- ❌ 회귀 문제는 분류보다 어려움
- ❌ 정확한 절대 좌표 계산 필요

---

### 옵션 B: 상대 위치 분류 (Classification - Relative Position)
```
입력: 지자기 시퀀스 [MagX, MagY, MagZ] × K timesteps
출력: 경로 내 마커 번호 (0, 1, 2, ..., N-1)

예: 11→1 경로에서 현재 마커 번호 = 42 (45m 중 18.9m 지점)
```

**장점**:
- ✅ 구현 간단
- ✅ 분류 문제로 학습 쉬움

**단점**:
- ❌ 경로마다 클래스 수 다름
- ❌ 다른 경로 간 일반화 어려움
- ❌ 중복 구간 문제 여전히 존재

---

### 옵션 C: 그리드 기반 분류 (Grid-based Classification) ⭐ 추천
```
입력: 지자기 시퀀스 [MagX, MagY, MagZ] × K timesteps
출력: 그리드 셀 ID

건물을 0.45m × 0.45m 그리드로 분할
각 그리드 셀 = 하나의 클래스
```

**예시**:
```
건물 범위: x ∈ [-85.5, 0], y ∈ [-9, 9]
그리드 크기: 0.45m × 0.45m

x 방향: 86m / 0.45m ≈ 191개
y 방향: 18m / 0.45m = 40개
총 그리드: 191 × 40 = 7,640개 셀

하지만 실제 방문한 셀만 사용하면 훨씬 적음!
```

**장점**:
- ✅ 중복 구간 완전 해결 (같은 그리드 = 같은 라벨)
- ✅ 분류 문제로 학습 쉬움
- ✅ 경로와 무관하게 일반화
- ✅ Highlighted 마커와 1:1 매칭

**단점**:
- ❌ 클래스 수가 많을 수 있음 (실제 방문한 셀만 사용하면 OK)

---

## 📋 추천: 옵션 C (그리드 기반 분류)

### 전처리 단계:

#### 1단계: 마커 절대 좌표 계산
```python
각 경로 파일에서:
1. Highlighted=True인 index 추출
2. 경로 정보 (start→end)로 경로 계획
3. 0.45m씩 이동하며 각 마커의 절대 좌표 계산

예: 11→1 경로
  마커 0:  (-45.00, 0)
  마커 1:  (-44.55, 0)
  마커 2:  (-44.10, 0)
  ...
  마커 99: (-0.45, 0)
```

#### 2단계: 그리드 매핑
```python
def coord_to_grid(x, y, grid_size=0.45):
    grid_x = int(round((x - x_min) / grid_size))
    grid_y = int(round((y - y_min) / grid_size))
    return grid_x, grid_y

# 그리드 ID = grid_y * num_x_grids + grid_x
```

#### 3단계: 시퀀스 생성
```python
for each Highlighted 마커:
    # 마커 위치 앞뒤 데이터로 시퀀스 생성
    start_idx = marker_idx - window_before
    end_idx = marker_idx + window_after

    sequence = data[start_idx:end_idx][['MagX', 'MagY', 'MagZ']]
    label = grid_id[marker_idx]

    dataset.append((sequence, label))
```

#### 4단계: 정규화
```python
# 전체 데이터셋에서 평균/표준편차 계산
mean = [MagX_mean, MagY_mean, MagZ_mean]
std = [MagX_std, MagY_std, MagZ_std]

# 정규화
normalized = (sequence - mean) / std
```

---

## ⚠️ 주의사항

### 1. 경로 계산 정확성
- 노드 위치는 참고용일 뿐
- 실제 경로는 Highlighted 마커 순서로 결정
- 직선 구간: 단순 계산
- 직각 구간: RightAngle 마커 활용

### 2. 시퀀스 윈도우 크기
```python
# 옵션 1: 마커 중심 (추천)
window_before = 10 samples  # 약 0.2초 @ 50Hz
window_after = 10 samples
total_length = 20

# 옵션 2: 마커 이전만
window_before = 20 samples
window_after = 0
```

### 3. 마커 간 샘플 부족 문제
```python
# 마커 간 평균 간격: ~30 samples
# Window 크기: 20 samples
# → 충분함!
```

---

## 🔍 검증 필요사항

다음을 구현 전에 확인:
1. ✅ 각 경로의 정확한 좌표 계산 방법
2. ✅ RightAngle 지점 처리
3. ⏳ 실제 방문한 그리드 셀 개수 (클래스 수)
4. ⏳ 마커 간 샘플 분포
5. ⏳ 시퀀스 윈도우 크기 최적화

---

## 다음 단계

**질문**:
1. 그리드 기반 분류 vs 절대 좌표 회귀 중 어느 것을 선호하시나요?
2. 시퀀스 윈도우: 마커 중심? 마커 이전만?
3. 바로 구현 시작할까요?
