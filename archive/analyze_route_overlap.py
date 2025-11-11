#!/usr/bin/env python3
"""
경로 중복 분석: 같은 구간이 서로 다른 경로에 포함되는지 확인
"""
import pandas as pd
import numpy as np
import os
from collections import defaultdict

# 노드 정보 로드
nodes_df = pd.read_csv('nodes_final.csv')
nodes = {row['id']: (row['x_m'], row['y_m']) for _, row in nodes_df.iterrows()}

print("="*70)
print("🔍 경로 중복 구간 분석")
print("="*70)

# 노드 위치 출력
print("\n📍 노드 위치 정보 (처음 15개):")
for node_id in sorted(nodes.keys())[:15]:
    x, y = nodes[node_id]
    print(f"   노드 {node_id:2d}: ({x:7.2f}, {y:5.2f})")

# 파일 분석
files = sorted([f for f in os.listdir('law_data') if f.endswith('.csv')])
routes_info = defaultdict(list)

print(f"\n📂 경로 파일 정보:")
for f in files[:20]:
    parts = f.replace('.csv', '').split('_')
    if len(parts) == 3:
        start, end, trial = parts
        start_pos = nodes.get(int(start), None)
        end_pos = nodes.get(int(end), None)

        if start_pos and end_pos:
            routes_info[(int(start), int(end))].append(f)
            if len(routes_info[(int(start), int(end))]) == 1:  # 첫 번째만 출력
                print(f"   {start}→{end}: {start_pos} → {end_pos}")

# 메인 복도 노드들 (y=0)
main_corridor = [i for i in range(1, 21) if nodes.get(i, (0,1))[1] == 0]
print(f"\n🛤️  메인 복도 노드 (y=0): {main_corridor}")

# 경로 중복 분석
print(f"\n⚠️  잠재적 문제 케이스 분석:")
print(f"{'='*70}")

# 예시: 1→11과 2→12는 몇 개의 노드를 공유하는가?
def get_path_nodes(start, end):
    """메인 복도 상의 경로 노드들 반환 (간단한 가정)"""
    if start < end:
        return list(range(start, end + 1))
    else:
        return list(range(start, end - 1, -1))

# 샘플 경로들의 노드 중복 확인
sample_routes = [
    (1, 11),
    (2, 12),
    (11, 1),
    (12, 2),
    (1, 20),
    (2, 20),
]

overlaps = []
for i, route1 in enumerate(sample_routes):
    for route2 in sample_routes[i+1:]:
        path1 = set(get_path_nodes(route1[0], route1[1]))
        path2 = set(get_path_nodes(route2[0], route2[1]))
        overlap = path1 & path2

        if len(overlap) > 1:  # 2개 이상 중복
            overlaps.append({
                'route1': f"{route1[0]}→{route1[1]}",
                'route2': f"{route2[0]}→{route2[1]}",
                'overlap_nodes': sorted(overlap),
                'overlap_count': len(overlap)
            })

# 중복이 많은 순서로 정렬
overlaps.sort(key=lambda x: x['overlap_count'], reverse=True)

print(f"\n🔴 중복 구간이 있는 경로 쌍 (Top 10):")
for i, ov in enumerate(overlaps[:10], 1):
    print(f"\n   [{i}] {ov['route1']} ↔ {ov['route2']}")
    print(f"       중복 노드 {ov['overlap_count']}개: {ov['overlap_nodes'][:10]}...")

# 실제 데이터로 검증
print(f"\n\n📊 실제 데이터 샘플 분석:")
print(f"{'='*70}")

# 1→11 데이터 일부 확인
if os.path.exists('law_data/1_11_1.csv'):
    df1 = pd.read_csv('law_data/1_11_1.csv')
    print(f"\n📁 1→11 경로 (1_11_1.csv):")
    print(f"   총 {len(df1)}개 샘플")
    print(f"   지속시간: {(pd.to_datetime(df1['Timestamp'].iloc[-1]) - pd.to_datetime(df1['Timestamp'].iloc[0])).total_seconds():.1f}초")
    print(f"   지자기 범위:")
    print(f"     MagX: [{df1['MagX'].min():.2f}, {df1['MagX'].max():.2f}]")
    print(f"     MagY: [{df1['MagY'].min():.2f}, {df1['MagY'].max():.2f}]")
    print(f"     MagZ: [{df1['MagZ'].min():.2f}, {df1['MagZ'].max():.2f}]")

# 2→12 데이터 일부 확인
if os.path.exists('law_data/2_12_1.csv'):
    df2 = pd.read_csv('law_data/2_12_1.csv')
    print(f"\n📁 2→12 경로 (2_12_1.csv):")
    print(f"   총 {len(df2)}개 샘플")
    print(f"   지속시간: {(pd.to_datetime(df2['Timestamp'].iloc[-1]) - pd.to_datetime(df2['Timestamp'].iloc[0])).total_seconds():.1f}초")
    print(f"   지자기 범위:")
    print(f"     MagX: [{df2['MagX'].min():.2f}, {df2['MagX'].max():.2f}]")
    print(f"     MagY: [{df2['MagY'].min():.2f}, {df2['MagY'].max():.2f}]")
    print(f"     MagZ: [{df2['MagZ'].min():.2f}, {df2['MagZ'].max():.2f}]")

print(f"\n\n💡 핵심 문제점:")
print(f"{'='*70}")
print(f"""
1. 서로 다른 경로가 동일한 물리적 구간을 공유
   → 같은 지자기 시퀀스가 서로 다른 라벨을 가질 수 있음

2. 지자기 데이터만으로는 "어디서 시작했는지" 알 수 없음
   → Context 정보 필요

3. 문제 정의 재검토 필요:
   ❌ 나쁜 접근: "어느 경로인가?" (1→11 vs 2→12)
   ✅ 좋은 접근:
      - "현재 어느 노드에 있는가?" (노드 위치 예측)
      - "다음 어디로 갈 것인가?" (action 예측, Decision Transformer)
      - "어떤 궤적을 따를 것인가?" (trajectory modeling)
""")

print(f"{'='*70}\n")
