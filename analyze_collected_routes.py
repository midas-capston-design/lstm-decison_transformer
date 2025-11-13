#!/usr/bin/env python3
"""
실제 수집된 경로 분석
"""
import os
from pathlib import Path
from collections import defaultdict

# law_data 디렉토리의 모든 csv 파일
law_data_dir = Path('law_data')
csv_files = list(law_data_dir.glob('*.csv'))

print(f"총 파일: {len(csv_files)}개\n")

# 경로별로 그룹화
routes = defaultdict(list)

for f in csv_files:
    name = f.stem  # 파일명 (확장자 제외)
    parts = name.split('_')

    if len(parts) >= 3:
        try:
            start = int(parts[0])
            end = int(parts[1])
            num = int(parts[2])

            routes[(start, end)].append(num)
        except:
            pass

# 정렬
sorted_routes = sorted(routes.items(), key=lambda x: (x[0][0], x[0][1]))

print("=" * 80)
print("실제 수집된 경로")
print("=" * 80)
print(f"\n{'경로':<15} {'파일 수':<10} {'파일 번호'}")
print("-" * 80)

for (start, end), nums in sorted_routes:
    nums_str = ', '.join(map(str, sorted(nums)))
    print(f"{start} → {end:<10} {len(nums)}개        [{nums_str}]")

print("\n" + "=" * 80)
print(f"총 경로 종류: {len(routes)}개")
print(f"총 파일: {sum(len(nums) for nums in routes.values())}개")
print("=" * 80)

# 인접 노드 (거리 < 5m) 확인
import pandas as pd
import numpy as np

nodes_df = pd.read_csv('nodes_final.csv')
node_coords = {}
for _, row in nodes_df.iterrows():
    node_coords[row['id']] = (row['x_m'], row['y_m'])

print("\n" + "=" * 80)
print("인접 노드 vs 장거리 경로")
print("=" * 80)

adjacent = []
long_distance = []

for (start, end), nums in routes.items():
    if start in node_coords and end in node_coords:
        x1, y1 = node_coords[start]
        x2, y2 = node_coords[end]
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        if dist < 5:
            adjacent.append((start, end, len(nums), dist))
        else:
            long_distance.append((start, end, len(nums), dist))

print(f"\n인접 노드 경로 ({len(adjacent)}개):")
print(f"{'경로':<15} {'파일 수':<10} {'거리'}")
print("-" * 80)
for start, end, count, dist in sorted(adjacent)[:20]:
    print(f"{start} → {end:<10} {count}개        {dist:.1f}m")
if len(adjacent) > 20:
    print(f"... 외 {len(adjacent)-20}개")

print(f"\n장거리 경로 ({len(long_distance)}개):")
print(f"{'경로':<15} {'파일 수':<10} {'거리'}")
print("-" * 80)
for start, end, count, dist in sorted(long_distance, key=lambda x: -x[2])[:20]:
    print(f"{start} → {end:<10} {count}개        {dist:.1f}m")
if len(long_distance) > 20:
    print(f"... 외 {len(long_distance)-20}개")
