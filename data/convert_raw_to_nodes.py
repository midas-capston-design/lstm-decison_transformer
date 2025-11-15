#!/usr/bin/env python3
"""
원본 센서 데이터(law_data)를 노드 단위로 변환
- 입력: law_data/{from}_{to}_{count}.csv (마커 간 이동 데이터)
- 출력: node_data/{from}_{to}_{count}.csv (노드 단위 데이터)
- Highlighted=True인 행을 기준으로 노드 구간 분할
- 각 노드의 센서 데이터는 구간 평균값 사용
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

def calculate_node_positions(from_marker, to_marker, num_nodes, markers_df, right_angle_node=None):
    """
    마커 간 노드 좌표 계산 (직각 경로, 0.45m 간격)

    Args:
        from_marker: 시작 마커 ID
        to_marker: 도착 마커 ID
        num_nodes: 노드 개수 (Highlighted=True 행의 개수)
        markers_df: 마커 정보 DataFrame
        right_angle_node: 직각 회전이 발생하는 노드 인덱스 (None이면 직선 경로)

    Returns:
        list of (x, y): 각 노드의 좌표
    """
    m1 = markers_df[markers_df['id'] == from_marker]
    m2 = markers_df[markers_df['id'] == to_marker]

    if len(m1) == 0 or len(m2) == 0:
        return None

    x1, y1 = m1.iloc[0]['x_m'], m1.iloc[0]['y_m']
    x2, y2 = m2.iloc[0]['x_m'], m2.iloc[0]['y_m']

    dx = x2 - x1
    dy = y2 - y1

    # 직선 경로 (x 또는 y 중 하나만 변화)
    if dx == 0 or dy == 0:
        # 단순 직선
        total_distance = abs(dx) + abs(dy)
        if total_distance > 0:
            dir_x = dx / total_distance if dx != 0 else 0
            dir_y = dy / total_distance if dy != 0 else 0
        else:
            dir_x, dir_y = 0, 0

        positions = []
        for i in range(num_nodes):
            distance = i * 0.45
            x = round(x1 + dir_x * distance, 2)
            y = round(y1 + dir_y * distance, 2)
            positions.append((x, y))
        return positions

    # 직각 경로 (x와 y 모두 변화)
    if right_angle_node is None:
        # RightAngle 정보가 없으면 중간 지점을 추정
        right_angle_node = num_nodes // 2

    positions = []

    # 먼저 x 방향으로 이동한다고 가정
    # right_angle_node까지는 x 방향
    for i in range(right_angle_node + 1):
        distance = i * 0.45
        # x 방향 이동
        if dx != 0:
            sign_x = 1 if dx > 0 else -1
            x = round(x1 + sign_x * distance, 2)
        else:
            x = x1
        y = y1
        positions.append((x, y))

    # right_angle_node 이후는 y 방향
    corner_x = positions[right_angle_node][0]
    corner_y = positions[right_angle_node][1]

    for i in range(right_angle_node + 1, num_nodes):
        distance = (i - right_angle_node) * 0.45
        x = corner_x
        if dy != 0:
            sign_y = 1 if dy > 0 else -1
            y = round(corner_y + sign_y * distance, 2)
        else:
            y = corner_y
        positions.append((x, y))

    return positions

def convert_file(input_file, output_file, markers_df):
    """
    하나의 파일을 노드 단위로 변환

    Args:
        input_file: 입력 파일 경로
        output_file: 출력 파일 경로
        markers_df: 마커 정보 DataFrame
    """
    # 데이터 로드
    df = pd.read_csv(input_file)

    # 파일명에서 마커 정보 추출
    filename = input_file.stem
    parts = filename.split('_')
    from_marker = int(parts[0])
    to_marker = int(parts[1])

    # Highlighted=True인 행 찾기 (노드 구간 경계)
    highlighted_indices = df[df['Highlighted'] == True].index.tolist()

    if len(highlighted_indices) == 0:
        print(f"  ⚠️  Highlighted 행 없음: {input_file.name}")
        return None

    # RightAngle이 있는 노드 찾기
    right_angle_node_idx = None
    for i, highlight_idx in enumerate(highlighted_indices):
        if i == 0:
            start_idx = 0
        else:
            start_idx = highlighted_indices[i-1] + 1
        end_idx = highlight_idx + 1

        segment = df.iloc[start_idx:end_idx]
        if segment['RightAngle'].any():
            right_angle_node_idx = i
            break

    # 노드 좌표 계산
    num_nodes = len(highlighted_indices)
    node_positions = calculate_node_positions(from_marker, to_marker, num_nodes, markers_df, right_angle_node_idx)

    if node_positions is None:
        print(f"  ⚠️  마커 정보 없음: {from_marker} → {to_marker}")
        return None

    # 노드 데이터 생성
    node_data = []

    for i, highlight_idx in enumerate(highlighted_indices):
        # 구간 정의: 이전 Highlighted부터 현재 Highlighted까지
        if i == 0:
            start_idx = 0
        else:
            start_idx = highlighted_indices[i-1] + 1

        end_idx = highlight_idx + 1  # 현재 Highlighted 행 포함

        # 구간 데이터 추출
        segment = df.iloc[start_idx:end_idx]

        if len(segment) == 0:
            continue

        # 해당 노드 좌표
        x, y = node_positions[i]

        # 센서 데이터 평균 계산
        mag_x = segment['MagX'].mean()
        mag_y = segment['MagY'].mean()
        mag_z = segment['MagZ'].mean()
        pitch = segment['Pitch'].mean()
        roll = segment['Roll'].mean()
        yaw = segment['Yaw'].mean()

        # RightAngle 체크 (구간 내에 RightAngle=True가 있는지)
        right_angle = segment['RightAngle'].any()

        node_data.append({
            'node_index': i,
            'x': x,
            'y': y,
            'mag_x': mag_x,
            'mag_y': mag_y,
            'mag_z': mag_z,
            'pitch': pitch,
            'roll': roll,
            'yaw': yaw,
            'right_angle': right_angle,
            'segment_size': len(segment)
        })

    # DataFrame으로 변환 및 저장
    nodes_df = pd.DataFrame(node_data)
    nodes_df.to_csv(output_file, index=False)

    return len(nodes_df)

def process_all_files():
    """모든 law_data 파일을 변환"""
    print("=" * 70)
    print("RAW 데이터 → 노드 데이터 변환")
    print("=" * 70)

    # 마커 정보 로드
    markers_df = pd.read_csv('nodes_final.csv')

    # 디렉토리 설정
    input_dir = Path('law_data')
    output_dir = Path('node_data')
    output_dir.mkdir(exist_ok=True)

    # 모든 CSV 파일 찾기
    input_files = sorted(input_dir.glob('*.csv'))
    print(f"\n입력 파일 수: {len(input_files)}")
    print(f"출력 디렉토리: {output_dir}/\n")

    # 통계
    total_files = 0
    total_nodes = 0
    failed_files = []

    for input_file in input_files:
        output_file = output_dir / input_file.name

        try:
            num_nodes = convert_file(input_file, output_file, markers_df)

            if num_nodes is not None:
                total_files += 1
                total_nodes += num_nodes
                print(f"✓ {input_file.name:20s} → {num_nodes:3d} nodes")
            else:
                failed_files.append(input_file.name)

        except Exception as e:
            print(f"❌ {input_file.name:20s} - Error: {e}")
            failed_files.append(input_file.name)

    # 결과 출력
    print("\n" + "=" * 70)
    print("✅ 변환 완료!")
    print(f"   성공: {total_files} 파일")
    print(f"   실패: {len(failed_files)} 파일")
    print(f"   총 노드: {total_nodes} 개")
    print(f"   평균 노드/파일: {total_nodes/total_files:.1f}")
    print("=" * 70)

    if failed_files:
        print("\n⚠️  실패한 파일:")
        for f in failed_files:
            print(f"   - {f}")

if __name__ == '__main__':
    process_all_files()
