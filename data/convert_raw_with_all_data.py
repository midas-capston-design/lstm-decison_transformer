#!/usr/bin/env python3
"""
원본 센서 데이터의 모든 행을 보존하면서 (x, y) 좌표 부여
- 입력: law_data/{from}_{to}_{count}.csv (모든 센서 데이터)
- 출력: processed_data/{from}_{to}_{count}.csv (모든 행에 x, y 좌표 추가)
- Highlighted=True 행들을 기준으로 노드 좌표 계산
- 노드 사이 데이터는 균등하게 보간하여 좌표 부여
"""
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_node_positions(from_marker, to_marker, num_nodes, markers_df, right_angle_node=None):
    """
    마커 간 노드 좌표 계산 (직각 경로, 0.45m 간격)
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
        right_angle_node = num_nodes // 2

    positions = []

    # right_angle_node까지는 x 방향
    for i in range(right_angle_node + 1):
        distance = i * 0.45
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

def interpolate_position(pos1, pos2, ratio):
    """
    두 좌표 사이를 선형 보간

    Args:
        pos1: (x1, y1) 시작 좌표
        pos2: (x2, y2) 끝 좌표
        ratio: 0~1 사이 비율
    """
    x1, y1 = pos1
    x2, y2 = pos2
    x = x1 + ratio * (x2 - x1)
    y = y1 + ratio * (y2 - y1)
    return round(x, 2), round(y, 2)

def convert_file_with_all_data(input_file, output_file, markers_df):
    """
    모든 센서 데이터 행을 보존하면서 (x, y) 좌표 부여
    """
    # 데이터 로드
    df = pd.read_csv(input_file)

    # 파일명에서 마커 정보 추출
    filename = input_file.stem
    parts = filename.split('_')
    from_marker = int(parts[0])
    to_marker = int(parts[1])

    # Highlighted=True인 행 찾기 (노드 경계)
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

    # 모든 행에 좌표 부여 (균등 보간)
    x_coords = []
    y_coords = []

    for i in range(len(df)):
        # 현재 행이 어느 노드 구간에 속하는지 찾기
        node_idx = 0
        for j, highlight_idx in enumerate(highlighted_indices):
            if i <= highlight_idx:
                node_idx = j
                break

        # 구간의 시작과 끝 인덱스
        if node_idx == 0:
            segment_start = 0
        else:
            segment_start = highlighted_indices[node_idx - 1] + 1

        segment_end = highlighted_indices[node_idx]

        # 구간 내에서의 위치 비율 계산 (0 ~ 1)
        segment_length = segment_end - segment_start
        if segment_length > 0:
            position_in_segment = i - segment_start
            ratio = position_in_segment / segment_length
        else:
            ratio = 0

        # 좌표 보간 (이전 노드 → 현재 노드)
        if node_idx == 0:
            # 첫 번째 구간: 0번 노드에서 1번 노드로
            pos1 = node_positions[0]
            pos2 = node_positions[1] if len(node_positions) > 1 else node_positions[0]
            x, y = interpolate_position(pos1, pos2, ratio)
        else:
            # 이전 노드와 현재 노드 사이를 균등 보간
            pos1 = node_positions[node_idx - 1]
            pos2 = node_positions[node_idx]
            x, y = interpolate_position(pos1, pos2, ratio)

        x_coords.append(x)
        y_coords.append(y)

    # 좌표 추가
    df['x'] = x_coords
    df['y'] = y_coords

    # 필요한 열만 선택하여 저장
    output_df = df[['x', 'y', 'MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw', 'Highlighted', 'RightAngle']]
    output_df.to_csv(output_file, index=False)

    return len(output_df)

def process_all_files():
    """모든 law_data 파일을 변환"""
    print("=" * 70)
    print("RAW 데이터 → 전체 데이터 변환 (모든 행 보존)")
    print("=" * 70)

    # 마커 정보 로드
    markers_df = pd.read_csv('nodes_final.csv')

    # 디렉토리 설정
    input_dir = Path('law_data')
    output_dir = Path('processed_data')
    output_dir.mkdir(exist_ok=True)

    # 모든 CSV 파일 찾기
    input_files = sorted(input_dir.glob('*.csv'))
    print(f"\n입력 파일 수: {len(input_files)}")
    print(f"출력 디렉토리: {output_dir}/\n")

    # 통계
    total_files = 0
    total_rows = 0
    failed_files = []

    for input_file in input_files:
        output_file = output_dir / input_file.name

        try:
            num_rows = convert_file_with_all_data(input_file, output_file, markers_df)

            if num_rows is not None:
                total_files += 1
                total_rows += num_rows
                print(f"✓ {input_file.name:20s} → {num_rows:5d} rows")
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
    print(f"   총 데이터 행: {total_rows:,} 개")
    print(f"   평균 행/파일: {total_rows/total_files:.1f}")
    print("=" * 70)

    if failed_files:
        print("\n⚠️  실패한 파일:")
        for f in failed_files:
            print(f"   - {f}")

if __name__ == '__main__':
    process_all_files()
