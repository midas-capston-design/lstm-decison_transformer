#!/usr/bin/env python3
"""
Flow Matching용 데이터 전처리
- 증강 없이 원본 데이터만 처리
- Train/Val/Test split 후
- Train에만 시퀀셜 증강 적용
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

# 설정
WINDOW_SIZE = 100  # 샘플
STRIDE = 5         # Sliding window stride
GRID_SIZE = 0.9   # m (더 큰 그리드로 위치당 샘플 수 증가)
SENSOR_COLS = ['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']

# 증강 설정 (Train only)
AUGMENT_RATIO = 0.5  # Train 데이터의 50%에 증강 적용
MAG_DRIFT_STD = 0.4    # 지자기 drift 표준편차 (μT)
MAG_NOISE_STD = 0.1    # 지자기 smooth noise 표준편차 (μT)
ORIENT_DRIFT_STD = 0.8  # 방향 drift 표준편차 (도)
ORIENT_NOISE_STD = 0.2  # 방향 smooth noise 표준편차 (도)

print("="*70)
print("🔧 Flow Matching용 데이터 전처리")
print("="*70)
print(f"  Window: {WINDOW_SIZE} 샘플 (2초 @ 50Hz)")
print(f"  Stride: {STRIDE} 샘플 (0.1초)")
print(f"  Grid: {GRID_SIZE}m")
print(f"\n  증강 설정 (Train only):")
print(f"    증강 비율: {AUGMENT_RATIO*100:.0f}%")
print(f"    지자기 drift: std={MAG_DRIFT_STD}μT")
print(f"    지자기 noise: std={MAG_NOISE_STD}μT (smooth)")
print(f"    방향 drift: std={ORIENT_DRIFT_STD}°")
print(f"    방향 noise: std={ORIENT_NOISE_STD}° (smooth)")

# ============================================================================
# 증강 함수 (시퀀셜 특성 유지)
# ============================================================================

def augment_sequence_sequential(seq):
    """
    시퀀셜 특성을 유지하는 증강

    1. Sensor Drift (90%): 전체 시퀀스에 동일한 바이어스
       → 패턴 완전히 유지, 센서 캘리브레이션 오차 모사

    2. Smooth Noise (10%): 시간적으로 연속적인 노이즈
       → 측정 노이즈 모사, Gaussian filter로 smooth

    Args:
        seq: (100, 6) numpy array [MagX, MagY, MagZ, Pitch, Roll, Yaw]

    Returns:
        augmented seq: (100, 6)
    """
    seq_aug = seq.copy()

    # 1. Sensor Drift (전체 시퀀스에 동일)
    drift_mag = np.random.randn(3) * MAG_DRIFT_STD
    drift_orient = np.random.randn(3) * ORIENT_DRIFT_STD
    seq_aug[:, 0:3] += drift_mag
    seq_aug[:, 3:6] += drift_orient

    # 2. Smooth Noise (시간적으로 연속)
    noise_mag = np.random.randn(seq.shape[0], 3) * MAG_NOISE_STD
    noise_orient = np.random.randn(seq.shape[0], 3) * ORIENT_NOISE_STD

    # Gaussian filter로 smooth하게
    for i in range(3):
        noise_mag[:, i] = gaussian_filter1d(noise_mag[:, i], sigma=5)
        noise_orient[:, i] = gaussian_filter1d(noise_orient[:, i], sigma=5)

    seq_aug[:, 0:3] += noise_mag
    seq_aug[:, 3:6] += noise_orient

    return seq_aug


# ============================================================================
# 1단계: 노드 정보 로드
# ============================================================================
print("\n[1/8] 노드 정보 로드...")
nodes_df = pd.read_csv('nodes_final.csv')
node_positions = {row['id']: (row['x_m'], row['y_m'])
                  for _, row in nodes_df.iterrows()}

# 건물 범위 계산
x_coords = [pos[0] for pos in node_positions.values()]
y_coords = [pos[1] for pos in node_positions.values()]
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

print(f"  건물 범위: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

# ============================================================================
# 2단계: 그리드 매핑 함수
# ============================================================================

def coord_to_grid(x, y):
    """절대 좌표를 그리드 ID로 변환"""
    grid_x = int(round((x - x_min) / GRID_SIZE))
    grid_y = int(round((y - y_min) / GRID_SIZE))

    num_x_grids = int(np.ceil((x_max - x_min) / GRID_SIZE)) + 1
    num_y_grids = int(np.ceil((y_max - y_min) / GRID_SIZE)) + 1

    grid_x = max(0, min(grid_x, num_x_grids - 1))
    grid_y = max(0, min(grid_y, num_y_grids - 1))

    grid_id = grid_y * num_x_grids + grid_x
    return grid_id


def calculate_marker_coordinates(start_pos, end_pos, num_markers):
    """
    시작점에서 끝점까지 num_markers 개의 좌표 생성

    **중요**: X축 또는 Y축으로만 이동 (대각선 X)
    - dx와 dy 중 하나만 0이 아니어야 함
    - 둘 다 변하면 → X축 먼저 이동 후 Y축 이동
    """
    coords = []
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]

    # 직선 이동인지 확인
    is_straight_x = (abs(dy) < 0.01)  # Y 고정, X만 변화
    is_straight_y = (abs(dx) < 0.01)  # X 고정, Y만 변화

    if is_straight_x or is_straight_y:
        # 직선 이동: 단순 보간
        for i in range(num_markers):
            progress = i / (num_markers - 1) if num_markers > 1 else 0
            x = start_pos[0] + dx * progress
            y = start_pos[1] + dy * progress
            coords.append((x, y))
    else:
        # 대각선 경로: X축 먼저 → Y축 이동
        # 중간점 계산
        mid_pos = (end_pos[0], start_pos[1])

        # X축 이동 구간의 마커 수 (거리 비율로 결정)
        total_dist = abs(dx) + abs(dy)
        x_dist = abs(dx)
        num_x_markers = max(1, int(num_markers * x_dist / total_dist))
        num_y_markers = num_markers - num_x_markers

        # X축 이동 (start → mid)
        for i in range(num_x_markers):
            progress = i / num_x_markers if num_x_markers > 1 else 0
            x = start_pos[0] + dx * progress
            y = start_pos[1]
            coords.append((x, y))

        # Y축 이동 (mid → end)
        for i in range(num_y_markers):
            progress = i / (num_y_markers - 1) if num_y_markers > 1 else 1
            x = end_pos[0]
            y = mid_pos[1] + dy * progress
            coords.append((x, y))

    return coords


def interpolate_position(pos1, pos2, t1, t2, t_current):
    """두 마커 사이에서 현재 샘플의 위치를 선형 보간"""
    if t2 == t1:
        return pos1

    progress = (t_current - t1) / (t2 - t1)
    progress = max(0, min(1, progress))

    x = pos1[0] + (pos2[0] - pos1[0]) * progress
    y = pos1[1] + (pos2[1] - pos1[1]) * progress

    return (x, y)


# ============================================================================
# 3단계: 파일별 처리 (증강 없이 원본만)
# ============================================================================

def process_file_no_augment(filepath):
    """
    하나의 경로 파일 처리 (증강 없음)

    Returns:
        sequences: (N, 100, 6)
        labels: (N,)
        coords: (N, 2)
    """
    filename = os.path.basename(filepath)
    parts = filename.replace('.csv', '').split('_')

    if len(parts) != 3:
        return None

    start_node = int(parts[0])
    end_node = int(parts[1])

    if start_node not in node_positions or end_node not in node_positions:
        return None

    # 데이터 로드
    df = pd.read_csv(filepath)

    # 센서 컬럼 확인
    if not all(col in df.columns for col in SENSOR_COLS):
        return None

    # Highlighted 마커 인덱스
    highlighted_indices = df[df['Highlighted'] == True].index.tolist()
    num_markers = len(highlighted_indices)

    if num_markers < 2:
        return None

    # 마커 좌표 계산
    start_pos = node_positions[start_node]
    end_pos = node_positions[end_node]
    marker_coords = calculate_marker_coordinates(start_pos, end_pos, num_markers)

    sequences = []
    labels = []
    coords_list = []
    trajectories = []  # 전체 궤적 저장

    # 연속된 마커 쌍 순회
    for i in range(len(highlighted_indices) - 1):
        marker_idx_A = highlighted_indices[i]
        marker_idx_B = highlighted_indices[i + 1]
        coord_A = marker_coords[i]
        coord_B = marker_coords[i + 1]

        # 마커 A와 B 사이의 모든 샘플 순회
        for center_idx in range(marker_idx_A, marker_idx_B, STRIDE):
            start_idx = center_idx - WINDOW_SIZE
            end_idx = center_idx

            if start_idx < 0:
                continue
            if end_idx > len(df):
                break

            # 센서 데이터 추출
            seq = df.iloc[start_idx:end_idx][SENSOR_COLS].values

            if len(seq) != WINDOW_SIZE:
                continue

            # 각 timestep의 위치 계산 (선형 보간)
            traj = []
            for t_idx in range(start_idx, end_idx):
                if t_idx <= marker_idx_A:
                    pos = coord_A
                elif t_idx >= marker_idx_B:
                    pos = coord_B
                else:
                    pos = interpolate_position(coord_A, coord_B,
                                              marker_idx_A, marker_idx_B, t_idx)
                traj.append(pos)

            traj = np.array(traj)  # (100, 2)

            # 마지막 위치
            x, y = traj[-1]
            grid_id = coord_to_grid(x, y)

            # 원본 데이터만 추가
            sequences.append(seq)
            labels.append(grid_id)
            coords_list.append((x, y))
            trajectories.append(traj)

    if len(sequences) == 0:
        return None

    return {
        'sequences': np.array(sequences),
        'labels': np.array(labels),
        'coords': np.array(coords_list),
        'trajectories': np.array(trajectories),  # (N, 100, 2)
        'route': f"{start_node}→{end_node}",
    }


# ============================================================================
# 4단계: 전체 데이터셋 생성 (증강 없음)
# ============================================================================

print("\n[2/8] 데이터 파일 처리 (증강 없음)...")

data_dir = Path('law_data')
files = sorted(list(data_dir.glob('*.csv')))

all_data = []
grid_stats = defaultdict(int)

np.random.seed(42)

for filepath in tqdm(files, desc="Processing files"):
    result = process_file_no_augment(filepath)
    if result is not None:
        all_data.append(result)

        for label in result['labels']:
            grid_stats[label] += 1

total_samples = sum(len(d['sequences']) for d in all_data)

print(f"\n  처리된 파일: {len(all_data)}/{len(files)}")
print(f"  고유 그리드 셀: {len(grid_stats)}")
print(f"  총 샘플 수 (원본): {total_samples:,}")

# ============================================================================
# 5단계: 데이터 통합 및 정규화
# ============================================================================

print("\n[3/8] 데이터 통합 및 정규화...")

all_sequences = np.vstack([d['sequences'] for d in all_data])
all_labels = np.concatenate([d['labels'] for d in all_data])
all_coords = np.vstack([d['coords'] for d in all_data])
all_trajectories = np.vstack([d['trajectories'] for d in all_data])

print(f"  통합 데이터 shape: {all_sequences.shape}")
print(f"  궤적 shape: {all_trajectories.shape}")

# 정규화
mean = all_sequences.mean(axis=(0, 1))
std = all_sequences.std(axis=(0, 1))

print(f"\n  정규화 파라미터:")
for i, col in enumerate(SENSOR_COLS):
    print(f"    {col}: mean={mean[i]:.4f}, std={std[i]:.4f}")

all_sequences_norm = (all_sequences - mean) / (std + 1e-8)

# 좌표 정규화 (-1, 1)
coords_min = all_coords.min(axis=0)
coords_max = all_coords.max(axis=0)
coords_range = coords_max - coords_min
all_coords_norm = 2 * (all_coords - coords_min) / (coords_range + 1e-8) - 1

# 궤적도 동일하게 정규화
all_trajectories_norm = 2 * (all_trajectories - coords_min) / (coords_range + 1e-8) - 1

# ============================================================================
# 6단계: Train/Val/Test 분할 (경로 기준)
# ============================================================================

print("\n[4/8] Train/Val/Test 분할 (경로 기준)...")

# 경로별 그룹화
route_groups = defaultdict(list)
for i, data in enumerate(all_data):
    route_groups[data['route']].append(i)

routes = list(route_groups.keys())
np.random.seed(42)
np.random.shuffle(routes)

n_routes = len(routes)
n_train = int(0.7 * n_routes)
n_val = int(0.15 * n_routes)

train_routes = routes[:n_train]
val_routes = routes[n_train:n_train + n_val]
test_routes = routes[n_train + n_val:]

print(f"  Train 경로: {len(train_routes)}")
print(f"  Val 경로: {len(val_routes)}")
print(f"  Test 경로: {len(test_routes)}")

# 인덱스 매핑
route_to_indices = {}
current_idx = 0
for data in all_data:
    route = data['route']
    n_samples = len(data['sequences'])
    route_to_indices[route] = list(range(current_idx, current_idx + n_samples))
    current_idx += n_samples

# Split
train_indices = []
for route in train_routes:
    train_indices.extend(route_to_indices[route])

val_indices = []
for route in val_routes:
    val_indices.extend(route_to_indices[route])

test_indices = []
for route in test_routes:
    test_indices.extend(route_to_indices[route])

states_train = all_sequences_norm[train_indices]
states_val = all_sequences_norm[val_indices]
states_test = all_sequences_norm[test_indices]

traj_train = all_trajectories_norm[train_indices]
traj_val = all_trajectories_norm[val_indices]
traj_test = all_trajectories_norm[test_indices]

labels_train = all_labels[train_indices]
labels_val = all_labels[val_indices]
labels_test = all_labels[test_indices]

coords_train = all_coords_norm[train_indices]
coords_val = all_coords_norm[val_indices]
coords_test = all_coords_norm[test_indices]

print(f"\n  Train: {len(train_indices):,} 샘플")
print(f"  Val:   {len(val_indices):,} 샘플")
print(f"  Test:  {len(test_indices):,} 샘플")

# ============================================================================
# 7단계: Train에만 증강 적용
# ============================================================================

print(f"\n[5/8] Train 데이터 증강 (시퀀셜 유지)...")
print(f"  증강 비율: {AUGMENT_RATIO*100:.0f}%")

n_train_samples = len(states_train)
n_augment = int(n_train_samples * AUGMENT_RATIO)

print(f"  원본: {n_train_samples:,}개")
print(f"  증강: {n_augment:,}개")
print(f"  최종: {n_train_samples + n_augment:,}개")

# 증강할 샘플 랜덤 선택
augment_indices = np.random.choice(n_train_samples, n_augment, replace=False)

augmented_states = []
augmented_traj = []
augmented_labels = []
augmented_coords = []

for idx in tqdm(augment_indices, desc="Augmenting"):
    # 센서 데이터 증강
    seq_aug = augment_sequence_sequential(states_train[idx])
    augmented_states.append(seq_aug)

    # 위치는 그대로 (센서만 증강)
    augmented_traj.append(traj_train[idx])
    augmented_labels.append(labels_train[idx])
    augmented_coords.append(coords_train[idx])

# Train 데이터에 증강 추가
states_train_final = np.vstack([states_train, np.array(augmented_states)])
traj_train_final = np.vstack([traj_train, np.array(augmented_traj)])
labels_train_final = np.concatenate([labels_train, np.array(augmented_labels)])
coords_train_final = np.vstack([coords_train, np.array(augmented_coords)])

print(f"\n  ✅ Train 최종: {len(states_train_final):,}개 (증강 포함)")
print(f"  ✅ Val: {len(states_val):,}개 (원본만)")
print(f"  ✅ Test: {len(states_test):,}개 (원본만)")

# ============================================================================
# 8단계: 저장
# ============================================================================

print("\n[6/8] 데이터 저장...")

output_dir = Path(__file__).parent / 'processed_data_flow_matching'
output_dir.mkdir(exist_ok=True)

# Train
np.save(output_dir / 'states_train.npy', states_train_final)
np.save(output_dir / 'trajectories_train.npy', traj_train_final)
np.save(output_dir / 'labels_train.npy', labels_train_final)
np.save(output_dir / 'coords_train.npy', coords_train_final)

# Val
np.save(output_dir / 'states_val.npy', states_val)
np.save(output_dir / 'trajectories_val.npy', traj_val)
np.save(output_dir / 'labels_val.npy', labels_val)
np.save(output_dir / 'coords_val.npy', coords_val)

# Test
np.save(output_dir / 'states_test.npy', states_test)
np.save(output_dir / 'trajectories_test.npy', traj_test)
np.save(output_dir / 'labels_test.npy', labels_test)
np.save(output_dir / 'coords_test.npy', coords_test)

# 메타 정보
metadata = {
    'window_size': WINDOW_SIZE,
    'stride': STRIDE,
    'grid_size': GRID_SIZE,
    'sensor_cols': SENSOR_COLS,
    'normalization': {
        'sensor_mean': mean.tolist(),
        'sensor_std': std.tolist(),
        'coords_min': coords_min.tolist(),
        'coords_max': coords_max.tolist(),
    },
    'augmentation': {
        'train_only': True,
        'augment_ratio': AUGMENT_RATIO,
        'mag_drift_std': MAG_DRIFT_STD,
        'mag_noise_std': MAG_NOISE_STD,
        'orient_drift_std': ORIENT_DRIFT_STD,
        'orient_noise_std': ORIENT_NOISE_STD,
    },
    'splits': {
        'train': len(states_train_final),
        'val': len(states_val),
        'test': len(states_test),
    }
}

with open(output_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"  저장 위치: {output_dir}")

print("\n" + "="*70)
print("✅ 전처리 완료!")
print("="*70)
print(f"""
📊 최종 데이터셋:
  Train: {len(states_train_final):,}개 (원본 {n_train_samples:,} + 증강 {n_augment:,})
  Val:   {len(states_val):,}개 (원본만)
  Test:  {len(states_test):,}개 (원본만)

🔥 증강 방식:
  ✅ Train에만 적용
  ✅ Sensor Drift (전체 시퀀스 동일) - 패턴 유지
  ✅ Smooth Noise (시간적 연속) - 측정 오차 모사
  ✅ Val/Test는 원본 데이터만

📁 출력:
  {output_dir}/
    ├── states_train.npy
    ├── trajectories_train.npy
    ├── states_val.npy
    ├── trajectories_val.npy
    ├── states_test.npy
    ├── trajectories_test.npy
    └── metadata.pkl
""")
