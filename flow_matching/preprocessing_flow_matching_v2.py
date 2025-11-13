#!/usr/bin/env python3
"""
Flow Matching용 데이터 전처리 V2 - 노드 그래프 기반

핵심 변경:
- 노드 그래프로 실제 경로 계산
- Highlighted 마커를 경로 상의 위치로 매핑
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
import networkx as nx

# 설정
WINDOW_SIZE = 250  # 5초
STRIDE = 50        # 1초 (API 호출 주기)
GRID_SIZE = 0.9
SENSOR_COLS = ['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']

# 증강 설정
AUGMENT_RATIO = 0.5
MAG_DRIFT_STD = 0.4
MAG_NOISE_STD = 0.1
ORIENT_DRIFT_STD = 0.8
ORIENT_NOISE_STD = 0.2

print("="*70)
print("🔧 Flow Matching 데이터 전처리 V2 (노드 그래프 기반)")
print("="*70)

# ============================================================================
# 1단계: 노드 그래프 생성
# ============================================================================
print("\n[1/9] 노드 그래프 생성...")

nodes_df = pd.read_csv('nodes_final.csv')
node_positions = {row['id']: (row['x_m'], row['y_m']) for _, row in nodes_df.iterrows()}

x_coords = [pos[0] for pos in node_positions.values()]
y_coords = [pos[1] for pos in node_positions.values()]
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

print(f"  노드 수: {len(node_positions)}")
print(f"  건물 범위: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

# 노드 그래프 생성
G = nx.Graph()
for node_id in node_positions:
    G.add_node(node_id)

# 가까운 노드들 연결 (5m 이하)
CONNECTION_THRESHOLD = 5.0
for node1, pos1 in node_positions.items():
    for node2, pos2 in node_positions.items():
        if node1 >= node2:
            continue
        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        if dist <= CONNECTION_THRESHOLD:
            G.add_edge(node1, node2, weight=dist)

# 잘못된 연결 제거 (실제로는 연결되지 않음)
wrong_connections = [(10, 28), (24, 25)]
for n1, n2 in wrong_connections:
    if G.has_edge(n1, n2):
        G.remove_edge(n1, n2)

print(f"  연결된 엣지: {G.number_of_edges()}개")

# ============================================================================
# 증강 함수
# ============================================================================

def augment_sequence_sequential(seq):
    """시퀀셜 특성 유지 증강"""
    seq_aug = seq.copy()

    # Sensor Drift
    drift_mag = np.random.randn(3) * MAG_DRIFT_STD
    drift_orient = np.random.randn(3) * ORIENT_DRIFT_STD
    seq_aug[:, 0:3] += drift_mag
    seq_aug[:, 3:6] += drift_orient

    # Smooth Noise
    noise_mag = np.random.randn(seq.shape[0], 3) * MAG_NOISE_STD
    noise_orient = np.random.randn(seq.shape[0], 3) * ORIENT_NOISE_STD

    for i in range(3):
        noise_mag[:, i] = gaussian_filter1d(noise_mag[:, i], sigma=5)
        noise_orient[:, i] = gaussian_filter1d(noise_orient[:, i], sigma=5)

    seq_aug[:, 0:3] += noise_mag
    seq_aug[:, 3:6] += noise_orient

    return seq_aug

# ============================================================================
# Grid 함수
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

# ============================================================================
# 경로 좌표 계산 (노드 그래프 기반)
# ============================================================================

def calculate_path_coordinates(node_path, num_markers):
    """
    노드 경로를 따라 num_markers 개의 좌표 생성

    Args:
        node_path: [1, 2, 3, 4, 22, 23] 같은 노드 ID 리스트
        num_markers: 마커 개수

    Returns:
        좌표 리스트 [(x, y), ...]
    """
    # 경로 상의 모든 위치 계산 (노드 간 선형 보간)
    path_positions = []
    path_distances = []  # 누적 거리

    cumulative_dist = 0.0
    path_positions.append(node_positions[node_path[0]])
    path_distances.append(0.0)

    for i in range(len(node_path) - 1):
        node1 = node_path[i]
        node2 = node_path[i + 1]
        pos1 = node_positions[node1]
        pos2 = node_positions[node2]

        # 두 노드 사이 거리
        segment_dist = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

        # 노드2 위치 추가
        cumulative_dist += segment_dist
        path_positions.append(pos2)
        path_distances.append(cumulative_dist)

    total_distance = cumulative_dist

    # 마커들을 경로를 따라 균등 배치
    marker_coords = []

    for i in range(num_markers):
        # 마커의 목표 거리
        target_dist = (i / (num_markers - 1)) * total_distance if num_markers > 1 else 0

        # 해당 거리에 있는 구간 찾기
        for j in range(len(path_distances) - 1):
            if path_distances[j] <= target_dist <= path_distances[j + 1]:
                # 구간 내 보간
                d1, d2 = path_distances[j], path_distances[j + 1]
                p1, p2 = path_positions[j], path_positions[j + 1]

                if d2 - d1 > 0:
                    t = (target_dist - d1) / (d2 - d1)
                else:
                    t = 0

                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                marker_coords.append((x, y))
                break
        else:
            # 마지막 위치
            marker_coords.append(path_positions[-1])

    return marker_coords

# ============================================================================
# 파일 처리
# ============================================================================

def process_file_no_augment(filepath):
    """파일 처리 (노드 그래프 기반)"""
    filename = os.path.basename(filepath)
    parts = filename.replace('.csv', '').split('_')

    if len(parts) != 3:
        return None

    start_node = int(parts[0])
    end_node = int(parts[1])

    if start_node not in node_positions or end_node not in node_positions:
        return None

    # 경로 계산
    if not nx.has_path(G, start_node, end_node):
        return None

    node_path = nx.shortest_path(G, start_node, end_node, weight='weight')

    # 데이터 로드
    df = pd.read_csv(filepath)

    if not all(col in df.columns for col in SENSOR_COLS):
        return None

    # Highlighted 마커
    highlighted_indices = df[df['Highlighted'] == True].index.tolist()
    num_markers = len(highlighted_indices)

    if num_markers < 2:
        return None

    # 마커 좌표 계산 (노드 경로 기반)
    marker_coords = calculate_path_coordinates(node_path, num_markers)

    # Step 1: 전체 파일 궤적 생성 (모든 timestep에 대해)
    full_trajectory = []
    for idx in range(len(df)):
        # 해당 timestep이 속한 마커 구간 찾기
        marker_pair_idx = None
        for i in range(len(highlighted_indices) - 1):
            if highlighted_indices[i] <= idx < highlighted_indices[i + 1]:
                marker_pair_idx = i
                break

        if marker_pair_idx is None:
            # 첫 마커 이전 or 마지막 마커 이후
            if idx < highlighted_indices[0]:
                pos = marker_coords[0]
            else:
                pos = marker_coords[-1]
        else:
            # 마커 사이: 보간
            marker_idx_A = highlighted_indices[marker_pair_idx]
            marker_idx_B = highlighted_indices[marker_pair_idx + 1]
            coord_A = marker_coords[marker_pair_idx]
            coord_B = marker_coords[marker_pair_idx + 1]

            progress = (idx - marker_idx_A) / (marker_idx_B - marker_idx_A)
            x = coord_A[0] + (coord_B[0] - coord_A[0]) * progress
            y = coord_A[1] + (coord_B[1] - coord_A[1]) * progress
            pos = [x, y]

        full_trajectory.append(pos)

    full_trajectory = np.array(full_trajectory)

    # Step 2: Sliding window로 샘플 생성
    sequences = []
    labels = []
    coords_list = []
    trajectories = []

    for start_idx in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
        end_idx = start_idx + WINDOW_SIZE

        # 센서 시퀀스
        seq = df.iloc[start_idx:end_idx][SENSOR_COLS].values
        if len(seq) != WINDOW_SIZE:
            continue

        # 궤적 (전체 궤적에서 슬라이싱)
        trajectory = full_trajectory[start_idx:end_idx]

        # 최종 위치
        final_pos = trajectory[-1]
        grid_id = coord_to_grid(final_pos[0], final_pos[1])

        sequences.append(seq)
        labels.append(grid_id)
        coords_list.append(final_pos)
        trajectories.append(trajectory)

    if len(sequences) == 0:
        return None

    return {
        'sequences': np.array(sequences),
        'labels': np.array(labels),
        'coords': np.array(coords_list),
        'trajectories': np.array(trajectories),
        'route': f"{start_node}→{end_node}",
        'node_path': node_path
    }

# ============================================================================
# 파일 처리
# ============================================================================
print("\n[2/9] 데이터 파일 처리...")

data_dir = Path('law_data')
files = sorted(data_dir.glob('*.csv'))

all_sequences = []
all_labels = []
all_coords = []
all_trajectories = []
route_data = defaultdict(lambda: {'sequences': [], 'trajectories': []})

for filepath in tqdm(files, desc="Processing files"):
    result = process_file_no_augment(filepath)
    if result is None:
        continue

    all_sequences.append(result['sequences'])
    all_labels.append(result['labels'])
    all_coords.append(result['coords'])
    all_trajectories.append(result['trajectories'])

    route = result['route']
    route_data[route]['sequences'].append(result['sequences'])
    route_data[route]['trajectories'].append(result['trajectories'])

print(f"\n  처리된 파일: {len(files)}/{len(files)}")

# 통합
all_sequences = np.vstack(all_sequences)
all_labels = np.concatenate(all_labels)
all_coords = np.vstack(all_coords)
all_trajectories = np.vstack(all_trajectories)

unique_grids = len(set(all_labels))

print(f"  고유 그리드 셀: {unique_grids}")
print(f"  총 샘플 수: {len(all_sequences):,}")

# ============================================================================
# 정규화
# ============================================================================
print("\n[3/9] 데이터 통합 및 정규화...")

mean = all_sequences.mean(axis=(0, 1))
std = all_sequences.std(axis=(0, 1))
all_sequences = (all_sequences - mean) / (std + 1e-8)

# 좌표 정규화
all_coords_normalized = np.zeros_like(all_coords)
all_coords_normalized[:, 0] = (all_coords[:, 0] - x_min) / (x_max - x_min) * 2 - 1
all_coords_normalized[:, 1] = (all_coords[:, 1] - y_min) / (y_max - y_min) * 2 - 1

# 궤적 정규화
all_trajectories_normalized = np.zeros_like(all_trajectories)
all_trajectories_normalized[:, :, 0] = (all_trajectories[:, :, 0] - x_min) / (x_max - x_min) * 2 - 1
all_trajectories_normalized[:, :, 1] = (all_trajectories[:, :, 1] - y_min) / (y_max - y_min) * 2 - 1

print(f"  통합 데이터 shape: {all_sequences.shape}")
print(f"  궤적 shape: {all_trajectories_normalized.shape}")

# ============================================================================
# Train/Val/Test 분할
# ============================================================================
print("\n[4/9] Train/Val/Test 분할 (경로 기준)...")

routes = list(route_data.keys())
np.random.seed(42)
np.random.shuffle(routes)

n_routes = len(routes)
n_train = int(n_routes * 0.7)
n_val = int(n_routes * 0.15)

train_routes = routes[:n_train]
val_routes = routes[n_train:n_train + n_val]
test_routes = routes[n_train + n_val:]

print(f"  Train 경로: {len(train_routes)}")
print(f"  Val 경로: {len(val_routes)}")
print(f"  Test 경로: {len(test_routes)}")

# 경로별로 분할
train_indices = [i for i, coord in enumerate(all_coords)
                 for route in train_routes
                 if any(np.array_equal(coord, c) for seqs in route_data[route]['sequences'] for c in seqs)]

# 간단한 방법: 인덱스 기반
route_to_indices = defaultdict(list)
idx = 0
for route in routes:
    for seqs in route_data[route]['sequences']:
        n_samples = len(seqs)
        route_to_indices[route].extend(range(idx, idx + n_samples))
        idx += n_samples

train_indices = [i for route in train_routes for i in route_to_indices[route]]
val_indices = [i for route in val_routes for i in route_to_indices[route]]
test_indices = [i for route in test_routes for i in route_to_indices[route]]

states_train = all_sequences[train_indices]
coords_train = all_coords_normalized[train_indices]
traj_train = all_trajectories_normalized[train_indices]
labels_train = all_labels[train_indices]

states_val = all_sequences[val_indices]
coords_val = all_coords_normalized[val_indices]
traj_val = all_trajectories_normalized[val_indices]
labels_val = all_labels[val_indices]

states_test = all_sequences[test_indices]
coords_test = all_coords_normalized[test_indices]
traj_test = all_trajectories_normalized[test_indices]
labels_test = all_labels[test_indices]

print(f"\n  Train: {len(states_train):,} 샘플")
print(f"  Val:   {len(states_val):,} 샘플")
print(f"  Test:  {len(states_test):,} 샘플")

# ============================================================================
# 증강
# ============================================================================
print(f"\n[5/9] Train 데이터 증강...")

n_augment = int(len(states_train) * AUGMENT_RATIO)
print(f"  증강 비율: {AUGMENT_RATIO*100:.0f}%")
print(f"  원본: {len(states_train):,}개")
print(f"  증강: {n_augment:,}개")

aug_indices = np.random.choice(len(states_train), n_augment, replace=False)
aug_sequences = []
aug_coords = []
aug_traj = []
aug_labels = []

for idx in tqdm(aug_indices, desc="Augmenting"):
    seq_aug = augment_sequence_sequential(states_train[idx])
    aug_sequences.append(seq_aug)
    aug_coords.append(coords_train[idx])
    aug_traj.append(traj_train[idx])
    aug_labels.append(labels_train[idx])

states_train = np.vstack([states_train, aug_sequences])
coords_train = np.vstack([coords_train, aug_coords])
traj_train = np.vstack([traj_train, aug_traj])
labels_train = np.concatenate([labels_train, aug_labels])

print(f"  최종: {len(states_train):,}개")

# ============================================================================
# 저장
# ============================================================================
print(f"\n[6/9] 데이터 저장...")

output_dir = Path(__file__).parent / 'processed_data_flow_matching'
output_dir.mkdir(exist_ok=True)

np.save(output_dir / 'states_train.npy', states_train)
np.save(output_dir / 'coords_train.npy', coords_train)
np.save(output_dir / 'trajectories_train.npy', traj_train)
np.save(output_dir / 'labels_train.npy', labels_train)

np.save(output_dir / 'states_val.npy', states_val)
np.save(output_dir / 'coords_val.npy', coords_val)
np.save(output_dir / 'trajectories_val.npy', traj_val)
np.save(output_dir / 'labels_val.npy', labels_val)

np.save(output_dir / 'states_test.npy', states_test)
np.save(output_dir / 'coords_test.npy', coords_test)
np.save(output_dir / 'trajectories_test.npy', traj_test)
np.save(output_dir / 'labels_test.npy', labels_test)

# 메타데이터
metadata = {
    'window_size': WINDOW_SIZE,
    'stride': STRIDE,
    'grid_size': GRID_SIZE,
    'sensor_cols': SENSOR_COLS,
    'normalization': {
        'sensor_mean': mean.tolist(),
        'sensor_std': std.tolist(),
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    },
    'node_graph': {
        'nodes': len(node_positions),
        'edges': G.number_of_edges(),
        'connection_threshold': CONNECTION_THRESHOLD
    }
}

with open(output_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"  저장 위치: {output_dir}")

print("\n" + "="*70)
print("✅ 전처리 완료!")
print("="*70)
print(f"\n📊 최종 데이터셋:")
print(f"  Train: {len(states_train):,}개")
print(f"  Val:   {len(states_val):,}개")
print(f"  Test:  {len(states_test):,}개")
