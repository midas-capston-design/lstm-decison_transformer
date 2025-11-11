# 🎯 최종 전처리 전략

## 📊 센서 신뢰도 분석 결과

### ✅ 사용 가능 센서
| 센서 | 신뢰도 | 용도 |
|------|--------|------|
| **MagX, MagY, MagZ** | ⭐⭐⭐ | 위치 정보 (방향 의존) |
| **Pitch, Roll, Yaw** | ⭐⭐⭐ | 자세/방향 정보 |
| AccY, AccZ | ⭐ | 중력 방향 (SNR 낮음) |

### ❌ 사용 불가 센서
| 센서 | 이유 |
|------|------|
| AccX | SNR < 1.0, 노이즈 과다 |
| GyroX, GyroY, GyroZ | 평균 ≈ 0, 사용 불가 |

---

## 🔥 핵심 발견

### 1. 방향 의존성 문제
```
1→11 경로:
  MagX 평균: -30.60 μT
  Yaw 평균: 312.44°

11→1 경로 (같은 물리적 경로, 반대 방향):
  MagX 평균: +42.87 μT  ← 완전히 다름!
  Yaw 평균: 2.49°      ← 약 310° 차이

→ 스마트폰 방향이 180° 반대이므로 지자기 값도 반전됨!
```

### 2. 마커 간 데이터
```
• Highlighted 마커: 0.45m 간격
• 마커 간 평균 샘플: 30개 (~0.6초 @ 50Hz)
• Window 크기: 20 샘플 = 0.4초
• 충분한 데이터 확보 ✅
```

---

## 🎯 최종 전처리 전략 (3가지 옵션)

### 옵션 1: 지자기 + 자세 (6차원) ⭐⭐⭐ **추천**

```python
입력: [MagX, MagY, MagZ, Pitch, Roll, Yaw] × 20 timesteps
     = (batch, 20, 6)

출력: 그리드 셀 ID (0~N)
     = (batch, num_grids)
```

**전처리 과정**:
```python
1. 각 Highlighted 마커의 절대 좌표 계산
2. 0.45m 그리드로 매핑 → 그리드 ID
3. 마커 중심 ± 10 샘플 → 20 샘플 시퀀스
4. [MagX, MagY, MagZ, Pitch, Roll, Yaw] 추출
5. 정규화: (x - mean) / std
```

**장점**:
- ✅ 방향 정보 명시적 포함
- ✅ 1→11과 11→1 구분 가능
- ✅ 물리적으로 타당

**단점**:
- ❌ 입력 차원 증가 (6차원)
- ❌ Yaw의 360° 주기성 처리 필요

---

### 옵션 2: 지자기만 + 데이터 증강 (3차원) ⭐⭐

```python
입력: [MagX, MagY, MagZ] × 20 timesteps
     = (batch, 20, 3)

출력: 그리드 셀 ID

데이터 증강: 1→11과 11→1 모두 같은 위치로 라벨링
```

**전처리 과정**:
```python
1. 각 Highlighted 마커의 절대 좌표 계산
2. 0.45m 그리드로 매핑
3. 1→11의 마커 50과 11→1의 마커 50을 같은 그리드로
4. 모델이 방향 불변성 학습
```

**장점**:
- ✅ 입력 간단 (3차원)
- ✅ 모델이 자동으로 방향 불변성 학습
- ✅ 구현 간단

**단점**:
- ❌ 방향 정보 손실
- ❌ 학습 데이터 2배 필요

---

### 옵션 3: 지자기 회전 보정 (3차원) ⭐

```python
입력: [MagX_world, MagY_world, MagZ_world] × 20 timesteps
     = (batch, 20, 3)

# Yaw를 사용해 지자기를 월드 좌표계로 변환
MagX_world = MagX * cos(Yaw) - MagY * sin(Yaw)
MagY_world = MagX * sin(Yaw) + MagY * cos(Yaw)
MagZ_world = MagZ
```

**장점**:
- ✅ 방향 불변성 확보
- ✅ 물리적으로 정확
- ✅ 입력 3차원 유지

**단점**:
- ❌ Yaw 정확도에 의존
- ❌ 변환 과정 복잡

---

## 📋 추천 구현 순서

### Phase 1: 옵션 1 (지자기 + 자세)
```
1. Highlighted 마커 좌표 계산
2. 그리드 매핑 (0.45m)
3. 시퀀스 생성 [MagX/Y/Z, Pitch/Roll/Yaw]
4. LSTM 학습
```

### Phase 2 (Optional): 옵션 2 비교
```
1. 지자기만 사용
2. 데이터 증강
3. 성능 비교
```

---

## 🔧 구현 세부사항

### 1. 그리드 생성
```python
# 건물 범위 (nodes_final.csv 기준)
x_min, x_max = -85.5, 0.0
y_min, y_max = -9.0, 9.0

grid_size = 0.45  # m

def coord_to_grid(x, y):
    grid_x = int(round((x - x_min) / grid_size))
    grid_y = int(round((y - y_min) / grid_size))
    grid_id = grid_y * num_x_grids + grid_x
    return grid_id
```

### 2. 마커 좌표 계산
```python
def calculate_marker_coords(route_file):
    """
    경로 파일에서 Highlighted 마커의 절대 좌표 계산

    예: 11→1 경로
    - 노드 11: (-45.0, 0)
    - 노드 1: (0.0, 0)
    - 거리: 45m
    - 마커 100개 → 0.45m 간격

    마커 0: (-45.0, 0)
    마커 1: (-44.55, 0)
    ...
    마커 99: (-0.45, 0)
    """
    # 경로 정보 추출
    start_node, end_node = parse_filename(route_file)

    # 경로 계획 (nodes_final.csv + RightAngle 마커)
    path = plan_route(start_node, end_node)

    # 0.45m씩 이동하며 좌표 계산
    coords = []
    for i in range(num_markers):
        coord = interpolate_path(path, i * 0.45)
        coords.append(coord)

    return coords
```

### 3. 시퀀스 생성
```python
def create_sequences(df, marker_indices, coords, window=10):
    """
    각 마커를 중심으로 시퀀스 생성
    """
    sequences = []
    labels = []

    for idx, (marker_idx, (x, y)) in enumerate(zip(marker_indices, coords)):
        # 마커 중심 ± window
        start = max(0, marker_idx - window)
        end = min(len(df), marker_idx + window)

        # 센서 데이터 추출
        seq = df.iloc[start:end][['MagX', 'MagY', 'MagZ',
                                    'Pitch', 'Roll', 'Yaw']].values

        # 패딩 (20 샘플로 맞추기)
        if len(seq) < 20:
            seq = np.pad(seq, ((0, 20-len(seq)), (0, 0)), 'edge')

        # 그리드 ID 계산
        grid_id = coord_to_grid(x, y)

        sequences.append(seq)
        labels.append(grid_id)

    return np.array(sequences), np.array(labels)
```

---

## ❓ 결정 사항

**지금 결정해주세요**:

1. **어떤 옵션?**
   - `1` → 지자기 + 자세 (6차원) **추천**
   - `2` → 지자기만 + 데이터 증강 (3차원)
   - `3` → 지자기 회전 보정 (3차원)

2. **시퀀스 윈도우?**
   - `중심` → 마커 ± 10 샘플
   - `이전` → 마커 이전 20 샘플

3. **바로 구현 시작?**
   - `예` → 전처리 코드 작성 시작
   - `아니오` → 더 분석 필요

답변 주시면 바로 시작하겠습니다! 🚀
