#!/usr/bin/env python3
"""
Route별 왕복 횟수 계산
"""
import numpy as np
import pandas as pd

# 노드 좌표 로드
nodes_df = pd.read_csv('nodes_final.csv')

# 필요 샘플 (이전 분석 결과)
needed_samples = {
    (13, 14): 197,
    (14, 13): 194,
    (14, 15): 142,
    (8, 7): 135,
    (10, 9): 129,
    (7, 8): 123,
    (9, 10): 116,
    (15, 14): 113,
    (11, 10): 110,
    (12, 11): 108,
    (10, 11): 105,
    (11, 12): 102,
    (12, 13): 99,
    (9, 8): 95,
    (13, 12): 92,
    (15, 16): 88,
    (16, 15): 85,
    (8, 9): 81,
    (2, 3): 80,
    (7, 6): 78,
    (3, 2): 75,
    (6, 7): 74,
    (16, 17): 71,
    (17, 16): 68,
    (6, 5): 64,
    (5, 6): 61,
    (5, 4): 59,
    (1, 2): 58,
    (4, 5): 56,
    (3, 4): 52,
    (4, 3): 49,
    (17, 18): 45,
    (18, 17): 42,
    (2, 1): 39,
}

# 노드 좌표 딕셔너리
node_coords = {}
for _, row in nodes_df.iterrows():
    node_coords[row['id']] = (row['x_m'], row['y_m'])

# 거리 계산 함수
def calc_distance(node1, node2):
    x1, y1 = node_coords[node1]
    x2, y2 = node_coords[node2]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# Window 생성 계산
SAMPLING_RATE = 50  # Hz
WALKING_SPEED = 1.0  # m/s
WINDOW_SIZE = 250  # samples
STRIDE = 50  # samples

def calc_windows_per_trip(distance_m):
    """한 번 왕복 시 생성되는 window 개수"""
    # 왕복 거리
    round_trip_distance = distance_m * 2
    # 왕복 시간 (초)
    round_trip_time = round_trip_distance / WALKING_SPEED
    # 샘플 수
    total_samples = round_trip_time * SAMPLING_RATE

    # Window 개수
    if total_samples < WINDOW_SIZE:
        return 0

    windows = int((total_samples - WINDOW_SIZE) / STRIDE) + 1
    return windows

# 각 route별 계산
results = []

for (node_from, node_to), needed in needed_samples.items():
    if node_from not in node_coords or node_to not in node_coords:
        continue

    distance = calc_distance(node_from, node_to)
    windows_per_trip = calc_windows_per_trip(distance)

    if windows_per_trip == 0:
        trips_needed = 9999  # 불가능
    else:
        trips_needed = int(np.ceil(needed / windows_per_trip))

    results.append({
        'route': f"{node_from} → {node_to}",
        'distance': distance,
        'needed_windows': needed,
        'windows_per_trip': windows_per_trip,
        'trips_needed': trips_needed
    })

# 정렬 (필요 샘플 많은 순)
results.sort(key=lambda x: -x['needed_windows'])

# 출력
print("=" * 70)
print("Route별 왕복 횟수")
print("=" * 70)
print(f"\n{'순위':<5} {'Route':<10} {'거리':<8} {'필요':<8} {'왕복당':<8} {'필요왕복':<8}")
print("-" * 70)

for rank, r in enumerate(results, 1):
    print(f"{rank:<5} {r['route']:<10} {r['distance']:.1f}m   {r['needed_windows']:<6}개  {r['windows_per_trip']:<6}개  {r['trips_needed']:<6}회")

print("\n" + "=" * 70)
print(f"총 필요 왕복: {sum(r['trips_needed'] for r in results):,}회")
print("=" * 70)
