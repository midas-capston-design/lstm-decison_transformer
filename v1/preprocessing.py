#!/usr/bin/env python3
"""
지자기 기반 실내 위치 추정 - 데이터 전처리
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import defaultdict

# 설정
WINDOW_SIZE = 100  # 샘플
GRID_SIZE = 0.45   # m
SENSOR_COLS = ['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']

print("="*70)
print("🔧 데이터 전처리 시작")
print("="*70)

# ============================================================================
# 1단계: 노드 정보 로드
# ============================================================================
print("\n[1/6] 노드 정보 로드...")
nodes_df = pd.read_csv('nodes_final.csv')
node_positions = {row['id']: (row['x_m'], row['y_m'])
                  for _, row in nodes_df.iterrows()}

print(f"  총 {len(node_positions)}개 노드 로드")

# 건물 범위 계산
x_coords = [pos[0] for pos in node_positions.values()]
y_coords = [pos[1] for pos in node_positions.values()]
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

print(f"  건물 범위: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

# ============================================================================
# 2단계: 경로 계획 함수
# ============================================================================

def plan_simple_route(start_node, end_node, node_positions):
    """
    간단한 경로 계획: 두 노드 사이를 직선 보간
    나중에 RightAngle 정보로 개선 가능
    """
    start_pos = node_positions[start_node]
    end_pos = node_positions[end_node]

    # 맨해튼 거리 기반 경로 (단순화)
    # 실제로는 복도 구조를 따라가야 하지만, 일단 직선으로
    return [start_pos, end_pos]


def calculate_marker_coordinates(start_pos, end_pos, num_markers):
    """
    시작점에서 끝점까지 num_markers 개의 좌표 생성 (0.45m 간격)
    """
    coords = []

    # 총 거리
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    total_distance = np.sqrt(dx**2 + dy**2)

    # 각 마커의 좌표 계산
    for i in range(num_markers):
        # 진행률 (0.0 ~ 1.0)
        progress = i / (num_markers - 1) if num_markers > 1 else 0

        x = start_pos[0] + dx * progress
        y = start_pos[1] + dy * progress
        coords.append((x, y))

    return coords


# ============================================================================
# 3단계: 그리드 매핑
# ============================================================================

def coord_to_grid(x, y):
    """절대 좌표를 그리드 ID로 변환"""
    grid_x = int(round((x - x_min) / GRID_SIZE))
    grid_y = int(round((y - y_min) / GRID_SIZE))

    # 그리드 범위
    num_x_grids = int(np.ceil((x_max - x_min) / GRID_SIZE)) + 1
    num_y_grids = int(np.ceil((y_max - y_min) / GRID_SIZE)) + 1

    # 범위 체크
    grid_x = max(0, min(grid_x, num_x_grids - 1))
    grid_y = max(0, min(grid_y, num_y_grids - 1))

    grid_id = grid_y * num_x_grids + grid_x
    return grid_id, (grid_x, grid_y)


# ============================================================================
# 4단계: 파일별 처리
# ============================================================================

def process_file(filepath):
    """
    하나의 경로 파일 처리

    Returns:
        sequences: (N, 100, 6) - N개의 시퀀스
        labels: (N,) - 그리드 ID
        coords: (N, 2) - 절대 좌표 (x, y)
    """
    filename = os.path.basename(filepath)
    parts = filename.replace('.csv', '').split('_')

    if len(parts) != 3:
        return None

    start_node = int(parts[0])
    end_node = int(parts[1])
    trial = parts[2]

    # 노드 존재 확인
    if start_node not in node_positions or end_node not in node_positions:
        return None

    # 데이터 로드
    df = pd.read_csv(filepath)

    # Highlighted 마커 인덱스
    highlighted_indices = df[df['Highlighted'] == True].index.tolist()
    num_markers = len(highlighted_indices)

    if num_markers == 0:
        return None

    # 마커 좌표 계산
    start_pos = node_positions[start_node]
    end_pos = node_positions[end_node]
    marker_coords = calculate_marker_coordinates(start_pos, end_pos, num_markers)

    # 시퀀스 및 라벨 생성
    sequences = []
    labels = []
    coords_list = []

    for marker_idx, (hl_idx, (x, y)) in enumerate(zip(highlighted_indices, marker_coords)):
        # 마커 직전 100 샘플
        start_idx = max(0, hl_idx - WINDOW_SIZE)
        end_idx = hl_idx

        # 센서 데이터 추출
        seq = df.iloc[start_idx:end_idx][SENSOR_COLS].values

        # 패딩 (100 샘플 미만인 경우)
        if len(seq) < WINDOW_SIZE:
            # 앞부분을 복사해서 패딩
            pad_len = WINDOW_SIZE - len(seq)
            if len(seq) > 0:
                seq = np.vstack([np.tile(seq[0], (pad_len, 1)), seq])
            else:
                continue  # 데이터가 아예 없으면 건너뛰기

        # 그리드 ID 계산
        grid_id, grid_xy = coord_to_grid(x, y)

        sequences.append(seq)
        labels.append(grid_id)
        coords_list.append((x, y))

    if len(sequences) == 0:
        return None

    return {
        'sequences': np.array(sequences),
        'labels': np.array(labels),
        'coords': np.array(coords_list),
        'route': f"{start_node}→{end_node}",
        'trial': trial
    }


# ============================================================================
# 5단계: 전체 데이터셋 생성
# ============================================================================

print("\n[2/6] 데이터 파일 처리...")

data_dir = Path('law_data')
files = sorted(list(data_dir.glob('*.csv')))

all_data = []
grid_stats = defaultdict(int)

for filepath in tqdm(files, desc="Processing files"):
    result = process_file(filepath)
    if result is not None:
        all_data.append(result)

        # 그리드 통계
        for label in result['labels']:
            grid_stats[label] += 1

print(f"\n  처리된 파일: {len(all_data)}/{len(files)}")
print(f"  고유 그리드 셀: {len(grid_stats)}")
print(f"  총 샘플 수: {sum(len(d['sequences']) for d in all_data)}")

# ============================================================================
# 6단계: 데이터 통합 및 정규화
# ============================================================================

print("\n[3/6] 데이터 통합 및 정규화...")

# 모든 시퀀스 통합
all_sequences = np.vstack([d['sequences'] for d in all_data])
all_labels = np.concatenate([d['labels'] for d in all_data])
all_coords = np.vstack([d['coords'] for d in all_data])

print(f"  통합 데이터 shape: {all_sequences.shape}")
print(f"  라벨 shape: {all_labels.shape}")

# 정규화 파라미터 계산
mean = all_sequences.mean(axis=(0, 1))
std = all_sequences.std(axis=(0, 1))

print(f"\n  정규화 파라미터:")
for i, col in enumerate(SENSOR_COLS):
    print(f"    {col}: mean={mean[i]:.4f}, std={std[i]:.4f}")

# 정규화 적용
all_sequences_norm = (all_sequences - mean) / (std + 1e-8)

# ============================================================================
# 7단계: Train/Val/Test 분할
# ============================================================================

print("\n[4/6] Train/Val/Test 분할...")

# 경로별로 그룹화
route_groups = defaultdict(list)
for i, data in enumerate(all_data):
    route_groups[data['route']].append(i)

# 경로를 70/15/15로 분할
routes = list(route_groups.keys())
np.random.seed(42)
np.random.shuffle(routes)

n_routes = len(routes)
n_train = int(0.7 * n_routes)
n_val = int(0.15 * n_routes)

train_routes = routes[:n_train]
val_routes = routes[n_train:n_train+n_val]
test_routes = routes[n_train+n_val:]

print(f"  Train 경로: {len(train_routes)}")
print(f"  Val 경로: {len(val_routes)}")
print(f"  Test 경로: {len(test_routes)}")

# 인덱스 수집
train_indices = []
val_indices = []
test_indices = []

idx_offset = 0
for data in all_data:
    route = data['route']
    n_samples = len(data['sequences'])
    indices = list(range(idx_offset, idx_offset + n_samples))

    if route in train_routes:
        train_indices.extend(indices)
    elif route in val_routes:
        val_indices.extend(indices)
    else:
        test_indices.extend(indices)

    idx_offset += n_samples

print(f"\n  Train 샘플: {len(train_indices)}")
print(f"  Val 샘플: {len(val_indices)}")
print(f"  Test 샘플: {len(test_indices)}")

# ============================================================================
# 8단계: 데이터셋 저장
# ============================================================================

print("\n[5/6] 데이터셋 저장...")

# 저장 디렉토리 생성
output_dir = Path('processed_data')
output_dir.mkdir(exist_ok=True)

# NumPy 형식으로 저장
np.save(output_dir / 'X_train.npy', all_sequences_norm[train_indices])
np.save(output_dir / 'y_train.npy', all_labels[train_indices])
np.save(output_dir / 'coords_train.npy', all_coords[train_indices])

np.save(output_dir / 'X_val.npy', all_sequences_norm[val_indices])
np.save(output_dir / 'y_val.npy', all_labels[val_indices])
np.save(output_dir / 'coords_val.npy', all_coords[val_indices])

np.save(output_dir / 'X_test.npy', all_sequences_norm[test_indices])
np.save(output_dir / 'y_test.npy', all_labels[test_indices])
np.save(output_dir / 'coords_test.npy', all_coords[test_indices])

# 메타데이터 저장
metadata = {
    'window_size': WINDOW_SIZE,
    'grid_size': GRID_SIZE,
    'sensor_cols': SENSOR_COLS,
    'mean': mean,
    'std': std,
    'num_classes': len(grid_stats),
    'grid_stats': dict(grid_stats),
    'x_range': (x_min, x_max),
    'y_range': (y_min, y_max),
}

with open(output_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"  저장 완료: {output_dir}")

# ============================================================================
# 9단계: 검증
# ============================================================================

print("\n[6/6] 데이터셋 검증...")

# 로드 테스트
X_train = np.load(output_dir / 'X_train.npy')
y_train = np.load(output_dir / 'y_train.npy')

print(f"\n  ✅ X_train shape: {X_train.shape}")
print(f"  ✅ y_train shape: {y_train.shape}")
print(f"  ✅ 클래스 수: {len(np.unique(y_train))}")
print(f"  ✅ 값 범위 확인:")
print(f"     X_train: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"     y_train: [{y_train.min()}, {y_train.max()}]")

# 클래스 분포
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n  📊 클래스 분포 (상위 10개):")
top_classes = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]
for cls, cnt in top_classes:
    print(f"     그리드 {cls}: {cnt}개 샘플")

print("\n" + "="*70)
print("✅ 전처리 완료!")
print("="*70)
print(f"""
다음 파일이 생성되었습니다:
  - processed_data/X_train.npy  ({X_train.shape})
  - processed_data/X_val.npy
  - processed_data/X_test.npy
  - processed_data/y_train.npy
  - processed_data/y_val.npy
  - processed_data/y_test.npy
  - processed_data/metadata.pkl

이제 LSTM 모델 학습을 시작할 수 있습니다!
""")
