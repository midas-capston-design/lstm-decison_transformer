#!/usr/bin/env python3
"""
마커 구간별 샘플 분석 (수정 버전)
각 구간의 경로상에 있는 모든 그리드를 포함
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import pickle

# 데이터 로드
data_dir = Path('hyena/processed_data_hyena')
states_train = np.load(data_dir / 'states_train.npy')
positions_train = np.load(data_dir / 'positions_train.npy')

with open(data_dir / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

coords_min = np.array(metadata['normalization']['coords_min'])
coords_max = np.array(metadata['normalization']['coords_max'])

# Denormalize
def denormalize_coords(coords_norm, coords_min, coords_max):
    coords_range = coords_max - coords_min
    return (coords_norm + 1) / 2 * coords_range + coords_min

positions_train_real = denormalize_coords(positions_train, coords_min, coords_max)

# 노드 좌표
nodes_df = pd.read_csv('nodes_final.csv')
node_coords = {}
for _, row in nodes_df.iterrows():
    node_coords[row['id']] = (row['x_m'], row['y_m'])

# Grid + Direction
GRID_SIZE = 0.9
TARGET_SAMPLES = 15

def coord_to_grid(x, y):
    return (int(x / GRID_SIZE), int(y / GRID_SIZE))

def get_direction_bin(yaw_mean):
    angle = yaw_mean * 180
    if angle < 0:
        angle += 360
    bin_idx = int((angle + 22.5) / 45) % 8
    return bin_idx

def get_direction_to_target(x1, y1, x2, y2):
    """두 점 사이의 방향을 8방향 bin으로 반환"""
    dx = x2 - x1
    dy = y2 - y1
    angle = np.arctan2(dy, dx) * 180 / np.pi
    if angle < 0:
        angle += 360
    bin_idx = int((angle + 22.5) / 45) % 8
    return bin_idx

def get_grids_on_path(x1, y1, x2, y2):
    """두 노드 사이 경로상의 모든 그리드 셀 반환"""
    grids = set()

    # 경로를 0.1m 간격으로 샘플링하여 지나가는 모든 그리드 찾기
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    steps = int(dist / 0.1) + 1

    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        grid = coord_to_grid(x, y)
        grids.add(grid)

    return list(grids)

# 현재 샘플 수 계산 (grid, direction별)
grid_dir_samples = defaultdict(int)
for i, (x, y) in enumerate(positions_train_real):
    grid_id = coord_to_grid(x, y)
    yaw_mean = states_train[i, :, 5].mean()
    direction = get_direction_bin(yaw_mean)
    grid_dir_samples[(grid_id, direction)] += 1

# 왕복당 생성 윈도우 계산
def calc_windows_per_trip(distance_m):
    WALKING_SPEED = 1.0
    SAMPLING_RATE = 50
    WINDOW_SIZE = 250
    STRIDE = 50

    round_trip = distance_m * 2
    time_sec = round_trip / WALKING_SPEED
    samples = time_sec * SAMPLING_RATE

    if samples < WINDOW_SIZE:
        return 0

    return int((samples - WINDOW_SIZE) / STRIDE) + 1

# 모든 노드 쌍 분석
all_nodes = sorted(node_coords.keys())
results = []

for node_from in all_nodes:
    for node_to in all_nodes:
        if node_from == node_to:
            continue

        x1, y1 = node_coords[node_from]
        x2, y2 = node_coords[node_to]
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # 인접 노드만 (거리 < 5m)
        if distance > 5:
            continue

        # 이 경로상의 모든 그리드
        path_grids = get_grids_on_path(x1, y1, x2, y2)

        # 이 경로의 예상 방향
        expected_direction = get_direction_to_target(x1, y1, x2, y2)

        # 인접 방향도 포함 (±1 bin)
        accepted_directions = [
            expected_direction,
            (expected_direction - 1) % 8,
            (expected_direction + 1) % 8
        ]

        # 현재 샘플 수: 경로상 그리드 + 맞는 방향
        current_total = 0
        for grid in path_grids:
            for direction in accepted_directions:
                current_total += grid_dir_samples.get((grid, direction), 0)

        # 부족 샘플 계산
        # 각 그리드당 각 방향당 TARGET_SAMPLES가 목표
        # 하지만 실제로는 하나의 방향만 주로 사용되므로
        # 경로당 TARGET_SAMPLES * len(path_grids) 정도가 적정
        target_total = TARGET_SAMPLES * len(path_grids)
        needed_total = max(0, target_total - current_total)

        # 왕복당 윈도우
        windows_per_trip = calc_windows_per_trip(distance)

        # 필요 왕복
        if windows_per_trip > 0:
            trips_needed = int(np.ceil(needed_total / windows_per_trip))
        else:
            trips_needed = 9999

        results.append({
            'from': node_from,
            'to': node_to,
            'distance': distance,
            'current': current_total,
            'needed': needed_total,
            'per_trip': windows_per_trip,
            'trips': trips_needed
        })

# 마커 순서대로 정렬
results.sort(key=lambda x: (x['from'], x['to']))

# 출력
print("=" * 80)
print("마커 구간별 데이터 분석 (수정 버전)")
print("=" * 80)
print(f"\n총 분석 구간: {len(results)}개")
print(f"\n{'구간':<10} {'거리':<8} {'현재':<8} {'부족':<8} {'왕복당':<8} {'필요왕복':<10}")
print("-" * 80)

for r in results:
    route = f"{r['from']} → {r['to']}"
    print(f"{route:<10} {r['distance']:.1f}m   {r['current']:<6}개  {r['needed']:<6}개  {r['per_trip']:<6}개  {r['trips']:<8}회")

# 통계
total_current = sum(r['current'] for r in results)
total_needed = sum(r['needed'] for r in results)
total_trips = sum(r['trips'] for r in results if r['trips'] < 9999)

print("\n" + "=" * 80)
print("📊 요약")
print("=" * 80)
print(f"총 구간: {len(results)}개")
print(f"현재 총 샘플: {total_current:,}개")
print(f"부족 샘플: {total_needed:,}개")
print(f"필요 왕복: {total_trips:,}회")
print("=" * 80)
