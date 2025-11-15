#!/usr/bin/env python3
"""
마커 데이터를 노드 단위로 전처리
- 각 마커를 0.45m 간격의 노드로 세분화
- 각 노드에 센서 데이터 매핑 (지자기, 방향벡터)
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

# 설정
NODE_INTERVAL = 0.45  # 노드 간격 (미터)

def load_marker_info():
    """마커 정보 로드 (nodes_final.csv는 실제로 마커 정보)"""
    markers_df = pd.read_csv('nodes_final.csv')
    return markers_df

def calculate_nodes_for_marker_pair(marker1, marker2, interval=NODE_INTERVAL):
    """
    두 마커 사이를 interval 간격으로 노드 생성

    Returns:
        list of dict: [{'x': x, 'y': y, 'from_marker': id1, 'to_marker': id2, 'node_index': i}, ...]
    """
    x1, y1 = marker1['x_m'], marker1['y_m']
    x2, y2 = marker2['x_m'], marker2['y_m']

    # 거리 계산
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # 노드 개수 계산 (시작점 제외, 끝점 포함)
    num_nodes = int(np.ceil(distance / interval))

    # 노드 생성
    nodes = []
    for i in range(num_nodes + 1):  # 시작점부터 끝점까지
        t = i / num_nodes if num_nodes > 0 else 0
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        nodes.append({
            'x': x,
            'y': y,
            'from_marker': marker1['id'],
            'to_marker': marker2['id'],
            'node_index': i,
            'total_nodes': num_nodes + 1
        })

    return nodes

def load_sensor_data(from_marker, to_marker):
    """
    law_data 디렉토리에서 마커 간 이동 센서 데이터 로드
    파일 형식: {from}_{to}_{count}.csv
    """
    law_data_dir = Path('law_data')

    # 해당 마커 쌍의 모든 파일 찾기
    pattern = f"{from_marker}_{to_marker}_*.csv"
    files = list(law_data_dir.glob(pattern))

    if not files:
        print(f"  ⚠️  센서 데이터 없음: {from_marker} → {to_marker}")
        return None

    # 여러 파일이 있으면 가장 최근 것 사용 (숫자가 큰 것)
    files.sort()
    selected_file = files[-1]

    try:
        df = pd.read_csv(selected_file)
        print(f"  ✓ 로드: {selected_file.name} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  ❌ 에러: {selected_file.name} - {e}")
        return None

def map_sensor_to_nodes(nodes, sensor_df):
    """
    센서 데이터를 노드에 균등하게 매핑
    - 각 노드에 해당하는 센서 데이터 구간의 평균값 할당
    """
    if sensor_df is None or len(sensor_df) == 0:
        return nodes

    total_nodes = len(nodes)
    rows_per_node = len(sensor_df) / total_nodes

    for i, node in enumerate(nodes):
        # 해당 노드에 해당하는 센서 데이터 구간
        start_idx = int(i * rows_per_node)
        end_idx = int((i + 1) * rows_per_node)

        if end_idx > len(sensor_df):
            end_idx = len(sensor_df)

        # 구간 데이터 추출
        segment = sensor_df.iloc[start_idx:end_idx]

        if len(segment) > 0:
            # 지자기 벡터 평균
            node['mag_x'] = segment['MagX'].mean()
            node['mag_y'] = segment['MagY'].mean()
            node['mag_z'] = segment['MagZ'].mean()

            # 방향 벡터 평균
            node['pitch'] = segment['Pitch'].mean()
            node['roll'] = segment['Roll'].mean()
            node['yaw'] = segment['Yaw'].mean()
        else:
            # 데이터 없으면 0으로 채움
            node['mag_x'] = 0
            node['mag_y'] = 0
            node['mag_z'] = 0
            node['pitch'] = 0
            node['roll'] = 0
            node['yaw'] = 0

    return nodes

def process_all_markers():
    """모든 마커 쌍에 대해 노드 생성 및 센서 데이터 매핑"""
    print("=" * 60)
    print("마커 → 노드 전처리 시작")
    print("=" * 60)

    # 마커 정보 로드
    markers_df = load_marker_info()
    print(f"\n총 마커 수: {len(markers_df)}")

    # law_data 디렉토리의 모든 경로 파악
    law_data_dir = Path('law_data')
    all_files = list(law_data_dir.glob('*.csv'))

    # 파일명에서 마커 쌍 추출
    marker_pairs = set()
    for f in all_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            try:
                from_m = int(parts[0])
                to_m = int(parts[1])
                marker_pairs.add((from_m, to_m))
            except ValueError:
                continue

    print(f"발견된 마커 경로 수: {len(marker_pairs)}")
    print()

    # 모든 노드 데이터 수집
    all_nodes = []
    node_id = 0

    for from_marker, to_marker in sorted(marker_pairs):
        print(f"[{from_marker:2d} → {to_marker:2d}] 처리 중...")

        # 마커 정보 가져오기
        m1 = markers_df[markers_df['id'] == from_marker]
        m2 = markers_df[markers_df['id'] == to_marker]

        if len(m1) == 0 or len(m2) == 0:
            print(f"  ⚠️  마커 정보 없음")
            continue

        m1 = m1.iloc[0]
        m2 = m2.iloc[0]

        # 노드 생성
        nodes = calculate_nodes_for_marker_pair(m1, m2)
        print(f"  생성된 노드 수: {len(nodes)}")

        # 센서 데이터 로드
        sensor_df = load_sensor_data(from_marker, to_marker)

        # 센서 데이터 매핑
        nodes = map_sensor_to_nodes(nodes, sensor_df)

        # 노드 ID 부여
        for node in nodes:
            node['node_id'] = node_id
            node_id += 1

        all_nodes.extend(nodes)
        print()

    # DataFrame으로 변환
    nodes_df = pd.DataFrame(all_nodes)

    # 열 순서 정리
    columns = ['node_id', 'x', 'y', 'mag_x', 'mag_y', 'mag_z', 'pitch', 'roll', 'yaw',
               'from_marker', 'to_marker', 'node_index', 'total_nodes']
    nodes_df = nodes_df[columns]

    # 저장
    output_file = 'nodes_with_sensors.csv'
    nodes_df.to_csv(output_file, index=False)

    print("=" * 60)
    print(f"✅ 완료!")
    print(f"   총 노드 수: {len(nodes_df)}")
    print(f"   저장 파일: {output_file}")
    print("=" * 60)

    # 통계 출력
    print("\n[통계]")
    print(f"  고유 마커 쌍: {len(marker_pairs)}")
    print(f"  평균 노드/경로: {len(nodes_df) / len(marker_pairs):.1f}")
    print(f"  지자기 범위:")
    print(f"    MagX: [{nodes_df['mag_x'].min():.2f}, {nodes_df['mag_x'].max():.2f}]")
    print(f"    MagY: [{nodes_df['mag_y'].min():.2f}, {nodes_df['mag_y'].max():.2f}]")
    print(f"    MagZ: [{nodes_df['mag_z'].min():.2f}, {nodes_df['mag_z'].max():.2f}]")

    return nodes_df

if __name__ == '__main__':
    nodes_df = process_all_markers()
