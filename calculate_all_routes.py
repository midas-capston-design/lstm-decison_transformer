#!/usr/bin/env python3
"""
모든 조합 경로별 왕복 횟수 계산
"""
import numpy as np
import pandas as pd

# 노드 좌표 로드
nodes_df = pd.read_csv('nodes_final.csv')

# 노드 좌표 딕셔너리
node_coords = {}
for _, row in nodes_df.iterrows():
    node_coords[row['id']] = (row['x_m'], row['y_m'])

# Window 생성 계산
SAMPLING_RATE = 50  # Hz
WALKING_SPEED = 1.0  # m/s
WINDOW_SIZE = 250  # samples
STRIDE = 50  # samples

def calc_distance_path(path):
    """경로의 총 거리 계산"""
    total_dist = 0
    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i+1]
        if n1 not in node_coords or n2 not in node_coords:
            return 0
        x1, y1 = node_coords[n1]
        x2, y2 = node_coords[n2]
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        total_dist += dist
    return total_dist

def calc_windows_per_trip(distance_m):
    """한 번 왕복 시 생성되는 window 개수"""
    round_trip_distance = distance_m * 2
    round_trip_time = round_trip_distance / WALKING_SPEED
    total_samples = round_trip_time * SAMPLING_RATE

    if total_samples < WINDOW_SIZE:
        return 0

    windows = int((total_samples - WINDOW_SIZE) / STRIDE) + 1
    return windows

# 최대 경로 조합 (필요 샘플과 경로)
combo_routes = [
    (197, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),  # 7 → 16
    (194, [16, 15, 14, 13, 12, 11, 10, 9, 8, 7]),  # 16 → 7
    (142, [8, 9, 10, 11, 12, 13, 14, 15, 16]),     # 8 → 16
    (135, [16, 15, 14, 13, 12, 11, 10, 9, 8]),     # 16 → 8
    (129, [7, 8, 9, 10, 11, 12, 13, 14, 15]),      # 7 → 15
    (123, [15, 14, 13, 12, 11, 10, 9, 8, 7]),      # 15 → 7
    (116, [9, 10, 11, 12, 13, 14, 15, 16]),        # 9 → 16
    (113, [16, 15, 14, 13, 12, 11, 10, 9]),        # 16 → 9
    (110, [7, 8, 9, 10, 11, 12, 13, 14]),          # 7 → 14
    (108, [14, 13, 12, 11, 10, 9, 8, 7]),          # 14 → 7
    (105, [10, 11, 12, 13, 14, 15, 16]),           # 10 → 16
    (102, [16, 15, 14, 13, 12, 11, 10]),           # 16 → 10
    (99, [7, 8, 9, 10, 11, 12, 13]),               # 7 → 13
    (95, [13, 12, 11, 10, 9, 8, 7]),               # 13 → 7
    (92, [11, 12, 13, 14, 15, 16]),                # 11 → 16
    (88, [16, 15, 14, 13, 12, 11]),                # 16 → 11
    (85, [7, 8, 9, 10, 11, 12]),                   # 7 → 12
    (81, [12, 11, 10, 9, 8, 7]),                   # 12 → 7
    (80, [1, 2, 3, 4, 5, 6, 7]),                   # 1 → 7
    (78, [12, 13, 14, 15, 16]),                    # 12 → 16
    (78, [7, 6, 5, 4, 3, 2, 1]),                   # 7 → 1
    (74, [16, 15, 14, 13, 12]),                    # 16 → 12
    (71, [7, 8, 9, 10, 11]),                       # 7 → 11
    (68, [11, 10, 9, 8, 7]),                       # 11 → 7
    (64, [1, 2, 3, 4, 5, 6]),                      # 1 → 6
    (64, [13, 14, 15, 16]),                        # 13 → 16
    (61, [6, 5, 4, 3, 2, 1]),                      # 6 → 1
    (61, [16, 15, 14, 13]),                        # 16 → 13
    (59, [2, 3, 4, 5, 6, 7]),                      # 2 → 7
    (59, [7, 8, 9, 10]),                           # 7 → 10
    (58, [7, 6, 5, 4, 3, 2]),                      # 7 → 2
    (58, [10, 9, 8, 7]),                           # 10 → 7
    (56, [1, 2, 3, 4, 5]),                         # 1 → 5
    (56, [13, 14, 15]),                            # 13 → 15
    (52, [5, 4, 3, 2, 1]),                         # 5 → 1
    (52, [15, 14, 13]),                            # 15 → 13
    (49, [14, 15, 16]),                            # 14 → 16
    (45, [16, 17, 18]),                            # 16 → 18
    (45, [16, 15, 14]),                            # 16 → 14
    (42, [18, 17, 16]),                            # 18 → 16
    (42, [8, 9, 10, 11]),                          # 8 → 11
    (39, [11, 10, 9, 8]),                          # 11 → 8
]

# 계산
results = []

for needed, path in combo_routes:
    distance = calc_distance_path(path)
    windows_per_trip = calc_windows_per_trip(distance)

    if windows_per_trip == 0:
        trips_needed = 9999
    else:
        trips_needed = int(np.ceil(needed / windows_per_trip))

    route_str = f"{path[0]} → {path[-1]}"

    results.append({
        'route': route_str,
        'distance': distance,
        'needed': needed,
        'per_trip': windows_per_trip,
        'trips': trips_needed
    })

# 출력
print("=" * 70)
print("최대 경로 조합 - 왕복 횟수")
print("=" * 70)
print(f"\n{'순위':<5} {'조합 경로':<12} {'거리':<10} {'필요':<8} {'왕복당':<8} {'필요왕복':<8}")
print("-" * 70)

for rank, r in enumerate(results, 1):
    print(f"{rank:<5} {r['route']:<12} {r['distance']:.1f}m     {r['needed']:<6}개  {r['per_trip']:<6}개  {r['trips']:<6}회")

print("\n" + "=" * 70)
print(f"총 필요 왕복: {sum(r['trips'] for r in results):,}회")
print("=" * 70)
