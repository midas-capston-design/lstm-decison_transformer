#!/usr/bin/env python3
"""
Decision Transformer용 올바른 전처리
각 timestep의 실제 위치를 저장
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

WINDOW_SIZE = 100
STRIDE = 5
SENSOR_COLS = ['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']

print("="*70)
print("🔧 Decision Transformer용 올바른 전처리")
print("="*70)

# 노드 정보 로드
nodes_df = pd.read_csv('nodes_final.csv')
node_positions = {row['id']: (row['x_m'], row['y_m'])
                  for _, row in nodes_df.iterrows()}

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
    """두 마커 사이 위치 보간"""
    if t2 == t1:
        return pos1
    progress = (t_current - t1) / (t2 - t1)
    progress = max(0, min(1, progress))
    x = pos1[0] + (pos2[0] - pos1[0]) * progress
    y = pos1[1] + (pos2[1] - pos1[1]) * progress
    return (x, y)

def process_file_trajectory(filepath):
    """
    각 timestep의 위치를 모두 저장

    Returns:
        sequences: (N, 100, 6)
        trajectories: (N, 100, 2)  # 각 timestep의 위치!
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

            # 🔥 각 timestep의 위치 계산!
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
        'trajectories': np.array(trajectories),  # (N, 100, 2)
        'route': f"{start_node}→{end_node}"
    }

print("\n[1/3] Raw 데이터 처리...")
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / 'law_data'
files = sorted(list(data_dir.glob('*.csv')))

all_data = []
for filepath in tqdm(files[:5], desc="파일 처리 (샘플)"):  # 처음 5개만 테스트
    result = process_file_trajectory(filepath)
    if result is not None:
        all_data.append(result)

if len(all_data) == 0:
    print("❌ 데이터 처리 실패")
    exit(1)

print(f"\n처리된 파일: {len(all_data)}")
print(f"첫 번째 파일:")
print(f"  sequences: {all_data[0]['sequences'].shape}")
print(f"  trajectories: {all_data[0]['trajectories'].shape}")

print("\n✅ 이제 (N, 100, 2) 형태의 trajectory가 생성됨!")
print("   전체 데이터로 실행하려면 files[:5]를 files로 변경")
