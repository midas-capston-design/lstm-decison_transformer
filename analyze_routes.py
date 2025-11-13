#!/usr/bin/env python3
"""
Route별 실제 샘플 수 분석
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import pickle

print("=" * 70)
print("Route별 샘플 분석")
print("=" * 70)

# 데이터 로드
data_dir = Path('hyena/processed_data_hyena')

if not data_dir.exists():
    print("❌ 전처리 데이터가 없습니다.")
    exit(1)

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

print(f"\n총 샘플: {len(states_train):,}개")

# ============================================================================
# Grid + Direction 분석
# ============================================================================
GRID_SIZE = 0.9

def coord_to_grid(x, y):
    return (int(x / GRID_SIZE), int(y / GRID_SIZE))

def get_direction_bin(yaw_mean):
    """8방향 구분"""
    angle = yaw_mean * 180
    if angle < 0:
        angle += 360
    bin_idx = int((angle + 22.5) / 45) % 8
    return bin_idx

# (grid, direction) 조합별 샘플 수
grid_dir_samples = defaultdict(int)
grid_dir_details = defaultdict(list)

for i, (x, y) in enumerate(positions_train_real):
    grid_id = coord_to_grid(x, y)
    yaw_mean = states_train[i, :, 5].mean()
    direction = get_direction_bin(yaw_mean)

    grid_dir_samples[(grid_id, direction)] += 1
    grid_dir_details[(grid_id, direction)].append(i)

print(f"\n총 (grid, direction) 조합: {len(grid_dir_samples)}개")

# ============================================================================
# Route 추출 (인접 그리드 기반)
# ============================================================================
print("\n" + "=" * 70)
print("Route별 샘플 수 (실제 데이터 기반)")
print("=" * 70)

# 노드 좌표 로드
nodes_df = pd.read_csv('nodes_final.csv')
print(f"\n노드 수: {len(nodes_df)}")

# 각 노드의 그리드 계산
node_to_grid = {}
for _, row in nodes_df.iterrows():
    node_id = row['id']
    x, y = row['x_m'], row['y_m']
    grid = coord_to_grid(x, y)
    node_to_grid[node_id] = grid

print(f"노드→그리드 매핑: {len(node_to_grid)}개")

# 인접 노드 route 분석
route_samples = defaultdict(int)

for node_from in range(1, 19):  # 1~18
    for node_to in range(1, 19):
        if node_from == node_to:
            continue

        # 인접 노드인지 확인 (거리 기준)
        if node_from not in node_to_grid or node_to not in node_to_grid:
            continue

        grid_from = node_to_grid[node_from]
        grid_to = node_to_grid[node_to]

        # 그리드 거리 계산
        dist = ((grid_from[0] - grid_to[0])**2 + (grid_from[1] - grid_to[1])**2)**0.5

        # 인접 노드 (그리드 거리 < 3)
        if dist < 3:
            # 이 route에 해당하는 샘플 수
            for direction in range(8):
                count = grid_dir_samples.get((grid_from, direction), 0)
                route_samples[(node_from, node_to)] += count

# 샘플이 있는 routes만 필터링
existing_routes = {route: count for route, count in route_samples.items() if count > 0}

print(f"\n실제 샘플이 있는 routes: {len(existing_routes)}개")

# ============================================================================
# Route별 상세 분석
# ============================================================================
print("\n" + "=" * 70)
print("실제 수집된 Routes (샘플 많은 순)")
print("=" * 70)

sorted_routes = sorted(existing_routes.items(), key=lambda x: -x[1])

print(f"\n{'순위':<5} {'Route':<10} {'현재 샘플':<12}")
print("-" * 70)

for rank, ((node_from, node_to), count) in enumerate(sorted_routes, 1):
    print(f"{rank:<5} {node_from:2d} → {node_to:2d}   {count:>6}개")

# ============================================================================
# 필요 샘플 계산
# ============================================================================
print("\n" + "=" * 70)
print("Route별 필요 샘플 계산")
print("=" * 70)

TARGET_SAMPLES = 15  # 목표: 각 (grid, direction)당 최소 15개

route_needs = {}

for node_from in range(1, 19):
    for node_to in range(1, 19):
        if node_from == node_to:
            continue

        if node_from not in node_to_grid or node_to not in node_to_grid:
            continue

        grid_from = node_to_grid[node_from]
        grid_to = node_to_grid[node_to]

        dist = ((grid_from[0] - grid_to[0])**2 + (grid_from[1] - grid_to[1])**2)**0.5

        if dist < 3:  # 인접 노드
            total_need = 0
            for direction in range(8):
                current = grid_dir_samples.get((grid_from, direction), 0)
                need = max(0, TARGET_SAMPLES - current)
                total_need += need

            if total_need > 0:
                route_needs[(node_from, node_to)] = total_need

print(f"\n추가 수집이 필요한 routes: {len(route_needs)}개")

sorted_needs = sorted(route_needs.items(), key=lambda x: -x[1])

print(f"\n{'순위':<5} {'Route':<10} {'현재':<10} {'필요':<10}")
print("-" * 70)

for rank, ((node_from, node_to), need) in enumerate(sorted_needs[:50], 1):
    current = existing_routes.get((node_from, node_to), 0)
    print(f"{rank:<5} {node_from:2d} → {node_to:2d}   {current:>6}개   {need:>6}개")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 70)
print("📊 요약")
print("=" * 70)

total_current = sum(existing_routes.values())
total_need = sum(route_needs.values())

print(f"\n현재 상태:")
print(f"  수집된 routes: {len(existing_routes)}개")
print(f"  총 샘플: {total_current:,}개")

print(f"\n추가 필요:")
print(f"  수집 필요 routes: {len(route_needs)}개")
print(f"  추가 샘플: {total_need:,}개")

print(f"\n목표 달성 후:")
print(f"  총 샘플: {total_current + total_need:,}개")
print(f"  평균 샘플/조합: {(total_current + total_need) / len(grid_dir_samples):.1f}개")

print("\n" + "=" * 70)
