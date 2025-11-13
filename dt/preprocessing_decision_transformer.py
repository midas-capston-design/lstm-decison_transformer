#!/usr/bin/env python3
"""
Decision Transformer용 완전한 데이터 전처리
각 timestep의 위치를 포함한 trajectory 데이터 생성
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from collections import defaultdict

print("="*70)
print("🚀 Decision Transformer용 데이터 전처리 (데이터 증강 포함)")
print("="*70)

# 설정
WINDOW_SIZE = 100  # 샘플 (2초 @ 50Hz)
STRIDE = 5         # Sliding window stride
SENSOR_COLS = ['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']

print(f"  Window: {WINDOW_SIZE} 샘플")
print(f"  Stride: {STRIDE} 샘플")

# ============================================================================
# 1단계: 노드 정보 로드
# ============================================================================
print("\n[1/7] 노드 정보 로드...")
nodes_df = pd.read_csv('nodes_final.csv')
node_positions = {row['id']: (row['x_m'], row['y_m'])
                  for _, row in nodes_df.iterrows()}

x_coords = [pos[0] for pos in node_positions.values()]
y_coords = [pos[1] for pos in node_positions.values()]
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

print(f"  건물 범위: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

# ============================================================================
# 2단계: 유틸리티 함수
# ============================================================================

def calculate_marker_coordinates(start_pos, end_pos, num_markers):
    """시작점에서 끝점까지 균등 분할"""
    coords = []
    for i in range(num_markers):
        progress = i / (num_markers - 1) if num_markers > 1 else 0
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        coords.append((x, y))
    return coords

def interpolate_position(pos1, pos2, t1, t2, t_current):
    """두 마커 사이 위치 선형 보간"""
    if t2 == t1:
        return pos1
    progress = (t_current - t1) / (t2 - t1)
    progress = max(0, min(1, progress))
    x = pos1[0] + (pos2[0] - pos1[0]) * progress
    y = pos1[1] + (pos2[1] - pos1[1]) * progress
    return (x, y)

# ============================================================================
# 3단계: 파일 처리 (Trajectory 포함)
# ============================================================================

def process_file_for_dt(filepath):
    """
    Decision Transformer용 데이터 추출

    Returns:
        sequences: (N, 100, 6) - 센서 데이터
        trajectories: (N, 100, 2) - 각 timestep의 위치
        route: 경로명
    """
    filename = filepath.name
    parts = filename.replace('.csv', '').split('_')

    if len(parts) != 3:
        return None

    start_node = int(parts[0])
    end_node = int(parts[1])

    if start_node not in node_positions or end_node not in node_positions:
        return None

    # 데이터 로드
    df = pd.read_csv(filepath)

    if not all(col in df.columns for col in SENSOR_COLS):
        return None

    # 마커 인덱스
    highlighted_indices = df[df['Highlighted'] == True].index.tolist()
    num_markers = len(highlighted_indices)

    if num_markers < 2:
        return None

    # 마커 좌표
    start_pos = node_positions[start_node]
    end_pos = node_positions[end_node]
    marker_coords = calculate_marker_coordinates(start_pos, end_pos, num_markers)

    sequences = []
    trajectories = []

    # 마커 쌍 순회
    for i in range(len(highlighted_indices) - 1):
        marker_idx_A = highlighted_indices[i]
        marker_idx_B = highlighted_indices[i + 1]
        coord_A = marker_coords[i]
        coord_B = marker_coords[i + 1]

        # 윈도우 순회
        for center_idx in range(marker_idx_A, marker_idx_B, STRIDE):
            start_idx = center_idx - WINDOW_SIZE
            end_idx = center_idx

            if start_idx < 0 or end_idx > len(df):
                continue

            # 센서 데이터
            seq = df.iloc[start_idx:end_idx][SENSOR_COLS].values
            if len(seq) != WINDOW_SIZE:
                continue

            # 각 timestep의 위치 계산
            trajectory = []
            for t_idx in range(start_idx, end_idx):
                x, y = interpolate_position(coord_A, coord_B,
                                           marker_idx_A, marker_idx_B,
                                           t_idx)
                trajectory.append([x, y])

            sequences.append(seq)
            trajectories.append(trajectory)

    if len(sequences) == 0:
        return None

    return {
        'sequences': np.array(sequences),
        'trajectories': np.array(trajectories),
        'route': f"{start_node}→{end_node}"
    }

# ============================================================================
# 4단계: 전체 데이터 처리
# ============================================================================

print("\n[2/7] Raw 데이터 처리...")
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / 'law_data'
files = sorted(list(data_dir.glob('*.csv')))

all_data = []
for filepath in tqdm(files, desc="파일 처리"):
    result = process_file_for_dt(filepath)
    if result is not None:
        all_data.append(result)

total_samples = sum(len(d['sequences']) for d in all_data)

print(f"\n  처리된 파일: {len(all_data)}/{len(files)}")
print(f"  총 샘플 수: {total_samples:,}")

# ============================================================================
# 5단계: 데이터 통합 및 정규화
# ============================================================================

print("\n[3/7] 데이터 통합 및 정규화...")

all_sequences = np.vstack([d['sequences'] for d in all_data])
all_trajectories = np.vstack([d['trajectories'] for d in all_data])

print(f"  센서 데이터: {all_sequences.shape}")
print(f"  Trajectory: {all_trajectories.shape}")

# 센서 데이터 정규화
sensor_mean = all_sequences.mean(axis=(0, 1))
sensor_std = all_sequences.std(axis=(0, 1))

print(f"\n  센서 정규화 파라미터:")
for i, col in enumerate(SENSOR_COLS):
    print(f"    {col}: mean={sensor_mean[i]:.4f}, std={sensor_std[i]:.4f}")

all_sequences_norm = (all_sequences - sensor_mean) / (sensor_std + 1e-8)

# 좌표 정규화 ([-1, 1])
coords_flat = all_trajectories.reshape(-1, 2)
coords_min = coords_flat.min(axis=0)
coords_max = coords_flat.max(axis=0)
coords_range = coords_max - coords_min

print(f"\n  좌표 정규화 파라미터:")
print(f"    Min: {coords_min}")
print(f"    Max: {coords_max}")
print(f"    Range: {coords_range}")

all_trajectories_norm = (all_trajectories - coords_min) / coords_range * 2 - 1

print(f"  정규화 후 좌표 범위: [{all_trajectories_norm.min():.2f}, {all_trajectories_norm.max():.2f}]")

# ============================================================================
# 6단계: Train/Val/Test 분할 (경로 기반)
# ============================================================================

print("\n[4/7] Train/Val/Test 분할...")

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

print(f"\n  Train 샘플: {len(train_indices):,}")
print(f"  Val 샘플: {len(val_indices):,}")
print(f"  Test 샘플: {len(test_indices):,}")

# ============================================================================
# 7단계: 데이터 증강 (Train만)
# ============================================================================

print("\n[5/9] 데이터 증강 (Train 세트만)...")

def augment_magnetic_data(seq, noise_range=(1.0, 3.0)):
    """
    지자기 데이터 증강

    MagX, MagY, MagZ에 1-3uT 범위의 Gaussian 노이즈 추가
    Pitch, Roll, Yaw는 그대로 유지

    Args:
        seq: (100, 6) 센서 데이터
        noise_range: (min_uT, max_uT) 노이즈 범위

    Returns:
        augmented_seq: (100, 6) 증강된 센서 데이터
    """
    seq_aug = seq.copy()

    # 노이즈 강도 랜덤 선택 (1-3uT)
    noise_std = np.random.uniform(noise_range[0], noise_range[1])

    # 지자기 3축에만 노이즈 추가
    mag_noise = np.random.normal(0, noise_std, size=(seq.shape[0], 3))
    seq_aug[:, :3] += mag_noise  # MagX, MagY, MagZ

    return seq_aug

# Train 데이터 추출
train_sequences_orig = all_sequences_norm[train_indices]
train_trajectories_orig = all_trajectories_norm[train_indices]

# 원본 + 증강 (2배)
train_sequences_aug = []
train_trajectories_aug = []

print(f"  원본 Train: {len(train_sequences_orig):,}")

for seq, traj in tqdm(zip(train_sequences_orig, train_trajectories_orig),
                     total=len(train_sequences_orig),
                     desc="증강 중"):
    # 원본
    train_sequences_aug.append(seq)
    train_trajectories_aug.append(traj)

    # 증강 (지자기 노이즈)
    seq_aug = augment_magnetic_data(seq, noise_range=(1.0, 3.0))
    train_sequences_aug.append(seq_aug)
    train_trajectories_aug.append(traj)  # 위치는 그대로

train_sequences_final = np.array(train_sequences_aug)
train_trajectories_final = np.array(train_trajectories_aug)

print(f"  증강 후 Train: {len(train_sequences_final):,} (2배)")

# ============================================================================
# 8단계: Returns-to-go 계산
# ============================================================================

print("\n[6/9] Returns-to-go 계산...")

def calculate_returns_to_go(trajectories):
    """
    각 timestep에서 목표까지의 거리를 음수로 (가까울수록 높은 return)

    Args:
        trajectories: (N, 100, 2)

    Returns:
        returns_to_go: (N, 100, 1)
    """
    N, T, _ = trajectories.shape
    rtg = np.zeros((N, T, 1))

    for i in range(N):
        goal = trajectories[i, -1]  # 마지막 위치 = 목표

        for t in range(T):
            current = trajectories[i, t]
            distance = np.linalg.norm(goal - current)
            rtg[i, t, 0] = -distance  # 음수 거리

    return rtg

rtg_train = calculate_returns_to_go(train_trajectories_final)
rtg_val = calculate_returns_to_go(all_trajectories_norm[val_indices])
rtg_test = calculate_returns_to_go(all_trajectories_norm[test_indices])

print(f"  Train RTG: {rtg_train.shape}")
print(f"  Val RTG: {rtg_val.shape}")
print(f"  Test RTG: {rtg_test.shape}")

# ============================================================================
# 9단계: 데이터 저장
# ============================================================================

print("\n[7/9] 데이터 저장...")

output_dir = Path('processed_data_dt')
output_dir.mkdir(exist_ok=True)

# Train (증강 포함)
np.save(output_dir / 'states_train.npy', train_sequences_final)
np.save(output_dir / 'trajectories_train.npy', train_trajectories_final)
np.save(output_dir / 'rtg_train.npy', rtg_train)

# Val
np.save(output_dir / 'states_val.npy', all_sequences_norm[val_indices])
np.save(output_dir / 'trajectories_val.npy', all_trajectories_norm[val_indices])
np.save(output_dir / 'rtg_val.npy', rtg_val)

# Test
np.save(output_dir / 'states_test.npy', all_sequences_norm[test_indices])
np.save(output_dir / 'trajectories_test.npy', all_trajectories_norm[test_indices])
np.save(output_dir / 'rtg_test.npy', rtg_test)

# Metadata
metadata = {
    'window_size': WINDOW_SIZE,
    'stride': STRIDE,
    'sensor_cols': SENSOR_COLS,
    'sensor_mean': sensor_mean,
    'sensor_std': sensor_std,
    'coords_min': coords_min,
    'coords_max': coords_max,
    'coords_range': coords_range,
    'building_range': {
        'x': (x_min, x_max),
        'y': (y_min, y_max)
    },
    'augmented': True,
    'augment_noise_range': (1.0, 3.0),
    'data_shapes': {
        'states_train': train_sequences_final.shape,
        'states_val': all_sequences_norm[val_indices].shape,
        'states_test': all_sequences_norm[test_indices].shape,
    }
}

with open(output_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"  저장 완료: {output_dir}/")

# ============================================================================
# 10단계: 검증
# ============================================================================

print("\n[8/9] 검증...")

states_train = np.load(output_dir / 'states_train.npy')
traj_train = np.load(output_dir / 'trajectories_train.npy')
rtg_train = np.load(output_dir / 'rtg_train.npy')

print(f"  states_train: {states_train.shape}")
print(f"  trajectories_train: {traj_train.shape}")
print(f"  rtg_train: {rtg_train.shape}")

print("\n" + "="*70)
print("✅ Decision Transformer용 데이터 전처리 완료!")
print("="*70)
print(f"""
📊 최종 데이터셋 (데이터 증강 적용):
  Train: {len(train_sequences_final):,} 샘플 (증강 2배)
  Val:   {len(val_indices):,} 샘플
  Test:  {len(test_indices):,} 샘플
  총합:  {len(train_sequences_final) + len(val_indices) + len(test_indices):,} 샘플

🔊 데이터 증강:
  지자기 노이즈: 1-3 uT (Gaussian)
  적용 대상: Train 세트만 (원본 + 증강 = 2배)
  영향 범위: MagX, MagY, MagZ만

📦 데이터 형태:
  states: (N, 100, 6) - 센서 시계열
  trajectories: (N, 100, 2) - 각 timestep의 위치 (정규화)
  rtg: (N, 100, 1) - returns-to-go (목표까지 거리)

📁 저장 위치: {output_dir}/
  - states_train.npy, states_val.npy, states_test.npy
  - trajectories_train.npy, trajectories_val.npy, trajectories_test.npy
  - rtg_train.npy, rtg_val.npy, rtg_test.npy
  - metadata.pkl

🎯 Flow Matching 학습 준비 완료!
  각 timestep마다:
    - State: 센서 측정값 (6차원)
    - Action: 위치 (2차원, 정규화됨)
    - Return-to-go: 목표까지 남은 거리
""")
