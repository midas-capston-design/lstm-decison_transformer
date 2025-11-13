#!/usr/bin/env python3
"""
Route 최적화 - 기존 route 조합으로 새 route 생성
예: 1→2 + 2→3 = 1→3
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

print("=" * 70)
print("Route 조합 최적화 분석")
print("=" * 70)

# ============================================================================
# 필요한 routes (이전 분석 결과)
# ============================================================================
needed_routes = {
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
    (7, 6): 78,
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

# ============================================================================
# 기존에 수집된 routes (가정: 인접 노드 중 일부만 수집됨)
# ============================================================================
# 실제로는 데이터를 확인해야 하지만, 일반적으로 연속된 노드들은 수집되어 있을 가능성이 높음
existing_routes = set()

# 추정: 연속된 노드들은 수집했을 가능성이 높음
# 예: 1-2, 2-3, 3-4, ... 등
for i in range(1, 18):
    existing_routes.add((i, i+1))
    existing_routes.add((i+1, i))

print(f"\n가정: 기존 수집 routes {len(existing_routes)}개")
print("(실제 데이터 확인 필요)")

# ============================================================================
# Route 조합 가능성 분석
# ============================================================================
print("\n" + "=" * 70)
print("Route 조합 가능성")
print("=" * 70)

# 그래프 구축 (인접 리스트)
graph = defaultdict(set)
for start, end in existing_routes:
    graph[start].add(end)

def find_path_bfs(start, end, graph, max_length=3):
    """BFS로 경로 찾기 (최대 길이 제한)"""
    from collections import deque

    queue = deque([(start, [start])])
    paths = []

    while queue:
        node, path = queue.popleft()

        if len(path) > max_length:
            continue

        if node == end and len(path) > 1:
            paths.append(path)
            continue

        for neighbor in graph[node]:
            if neighbor not in path:  # 순환 방지
                queue.append((neighbor, path + [neighbor]))

    return paths

# 각 필요 route에 대해 조합 가능성 분석
combinable_routes = []
uncombinableRoutes = []

for (start, end), samples_needed in needed_routes.items():
    # 이미 수집된 route인지 확인
    if (start, end) in existing_routes:
        print(f"✅ {start}→{end}: 이미 수집됨 (추가 수집 {samples_needed}개 필요)")
        continue

    # 조합 가능한 경로 찾기
    paths = find_path_bfs(start, end, graph, max_length=4)

    if paths:
        # 가장 짧은 경로 선택
        shortest_path = min(paths, key=len)

        # 필요한 기존 routes
        sub_routes = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]

        combinable_routes.append({
            'target': (start, end),
            'samples_needed': samples_needed,
            'path': shortest_path,
            'sub_routes': sub_routes,
            'num_segments': len(sub_routes)
        })

        route_str = " → ".join(map(str, shortest_path))
        sub_routes_str = " + ".join([f"{s}→{e}" for s, e in sub_routes])
        print(f"✅ {start}→{end} ({samples_needed}개): {route_str}")
        print(f"   조합: {sub_routes_str}")
    else:
        uncombinableRoutes.append({
            'route': (start, end),
            'samples_needed': samples_needed
        })
        print(f"❌ {start}→{end} ({samples_needed}개): 조합 불가능 (신규 수집 필요)")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 70)
print("📊 조합 요약")
print("=" * 70)

total_needed = len(needed_routes)
total_combinable = len(combinable_routes)
total_new = len(uncombinableRoutes)

print(f"\n총 필요 routes: {total_needed}개")
print(f"  조합 가능: {total_combinable}개 ({total_combinable/total_needed*100:.1f}%)")
print(f"  신규 수집: {total_new}개 ({total_new/total_needed*100:.1f}%)")

samples_combinable = sum(r['samples_needed'] for r in combinable_routes)
samples_new = sum(r['samples_needed'] for r in uncombinableRoutes)
total_samples = sum(needed_routes.values())

print(f"\n필요 샘플:")
print(f"  조합으로 해결: {samples_combinable:,}개 ({samples_combinable/total_samples*100:.1f}%)")
print(f"  신규 수집 필요: {samples_new:,}개 ({samples_new/total_samples*100:.1f}%)")

# ============================================================================
# 최적화된 수집 계획
# ============================================================================
print("\n" + "=" * 70)
print("🎯 최적화된 수집 계획")
print("=" * 70)

print("\n[A] 조합으로 해결 가능한 routes")
print("-" * 70)

# 세그먼트 수로 정렬 (짧은 것부터)
combinable_routes.sort(key=lambda x: (x['num_segments'], -x['samples_needed']))

for r in combinable_routes[:10]:  # Top 10
    route_str = " → ".join(map(str, r['path']))
    sub_routes_str = " + ".join([f"{s}→{e}" for s, e in r['sub_routes']])
    start, end = r['target']
    print(f"{start:2d} → {end:2d} ({r['samples_needed']:3d}개)")
    print(f"        경로: {route_str}")
    print(f"        조합: {sub_routes_str}")
    print()

if len(combinable_routes) > 10:
    print(f"... 외 {len(combinable_routes)-10}개 routes")

print("\n[B] 신규 수집 필요 routes")
print("-" * 70)

# 샘플 수로 정렬 (많은 것부터)
uncombinableRoutes.sort(key=lambda x: -x['samples_needed'])

for r in uncombinableRoutes[:20]:  # Top 20
    start, end = r['route']
    print(f"{start:2d} → {end:2d}: {r['samples_needed']:3d}개 ← 신규 수집")

if len(uncombinableRoutes) > 20:
    print(f"... 외 {len(uncombinableRoutes)-20}개 routes")

# ============================================================================
# 가장 효율적인 수집 전략
# ============================================================================
print("\n" + "=" * 70)
print("💡 효율적 수집 전략")
print("=" * 70)

print("\n1. 기존 routes를 조합하여 다음 routes 생성:")
print(f"   → {total_combinable}개 routes ({samples_combinable:,}개 샘플)")
print("   → 추가 수집 불필요 (데이터 재활용)")

print(f"\n2. 신규 수집이 필요한 routes:")
print(f"   → {total_new}개 routes ({samples_new:,}개 샘플)")
print(f"   → 실제 현장 수집 필요")

print("\n3. 우선순위:")
print("   ① 신규 routes 중 샘플 수가 많은 것부터")
print("   ② 조합 routes는 전처리 단계에서 자동 생성")

# ============================================================================
# 실행 계획
# ============================================================================
print("\n" + "=" * 70)
print("📋 실행 계획")
print("=" * 70)

print("\n[Step 1] 기존 데이터 확인")
print("  - 실제 수집된 routes 파악")
print("  - 어떤 routes가 조합 가능한지 재계산")

print("\n[Step 2] 신규 routes 수집")
print(f"  - {total_new}개 routes 현장 수집")
print(f"  - 총 {samples_new:,}개 샘플 필요")
print(f"  - 예상 시간: {samples_new / 60 / 60:.1f}시간 (1초당 1샘플)")

print("\n[Step 3] Route 조합 스크립트 작성")
print("  - 1→2 + 2→3 = 1→3 자동 생성")
print("  - 전처리 파이프라인에 통합")

print("\n[Step 4] 전체 재전처리 및 학습")

print("\n" + "=" * 70)
print("✅ 분석 완료")
print("=" * 70)
