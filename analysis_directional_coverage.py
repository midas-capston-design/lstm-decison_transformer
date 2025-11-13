#!/usr/bin/env python3
"""
방향을 고려한 마커 기반 커버리지 분석 스크립트.

law_data/*.csv의 Highlighted 마커를 노드 그래프 기반 좌표로 복원하고,
슬라이딩 윈도우(250샘플, stride 50) 기준 최종 좌표와 이동 방향을 분류해
(그리드, 방향) 조합별 샘플 수를 계산한다. (증강 전 원본 데이터만 대상)

출력: results/directional_marker_counts.csv
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

# 설정
GRID_SIZE = 0.9
GRID_OFFSET = 0.45  # 실제 마커는 0.45m 간격으로 배치되어 있어 0.45 오프로 정렬된 0.9m 그리드를 사용
CONNECTION_THRESHOLD = 5.0  # 노드 연결 기준
WRONG_CONNECTIONS = {(10, 28), (24, 25)}
MIN_TARGET = 10
GOAL_TARGET = 20
WINDOW_SIZE = 250
STRIDE = 50

DATA_DIR = Path("law_data")
NODES_PATH = Path("nodes_final.csv")
OUTPUT_CSV = Path("results/directional_marker_counts.csv")


def load_nodes() -> Tuple[Dict[int, Tuple[float, float]], nx.Graph]:
    nodes_df = pd.read_csv(NODES_PATH)
    node_positions = {
        int(row.id): (float(row.x_m), float(row.y_m)) for row in nodes_df.itertuples()
    }

    G = nx.Graph()
    for node_id in node_positions:
        G.add_node(node_id)

    ids = sorted(node_positions.keys())
    for i, node1 in enumerate(ids):
        for node2 in ids[i + 1 :]:
            pos1, pos2 = node_positions[node1], node_positions[node2]
            dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
            if dist <= CONNECTION_THRESHOLD:
                G.add_edge(node1, node2, weight=dist)

    for bad in WRONG_CONNECTIONS:
        if G.has_edge(*bad):
            G.remove_edge(*bad)

    return node_positions, G


def quantize_coord(value: float) -> float:
    """0.45 m offset 0.9 m grid에 스냅."""
    snapped = GRID_OFFSET + GRID_SIZE * round((value - GRID_OFFSET) / GRID_SIZE)
    return round(snapped, 2)


def direction_label(dx: float, dy: float) -> str:
    """이동 벡터를 4방향으로 분류."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "정지"

    if abs(dx) >= abs(dy):
        return "동행(→)" if dx > 0 else "서행(←)"
    else:
        return "북행(↑)" if dy > 0 else "남행(↓)"


def calculate_path_coordinates(
    node_path: Sequence[int],
    num_markers: int,
    node_positions: Dict[int, Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """노드 경로를 따라 num_markers개의 위치를 균등 배치."""
    if num_markers <= 0:
        return []

    path_positions: List[Tuple[float, float]] = [node_positions[node_path[0]]]
    path_distances = [0.0]
    cumulative = 0.0

    for idx in range(len(node_path) - 1):
        pos1 = node_positions[node_path[idx]]
        pos2 = node_positions[node_path[idx + 1]]
        dist = math.hypot(pos2[0] - pos1[0], pos2[1] - pos1[1])
        cumulative += dist
        path_positions.append(pos2)
        path_distances.append(cumulative)

    total_dist = path_distances[-1] if path_distances else 0.0
    marker_coords: List[Tuple[float, float]] = []

    for m_idx in range(num_markers):
        if num_markers == 1:
            marker_coords.append(path_positions[0])
            continue
        target = (m_idx / (num_markers - 1)) * total_dist
        for seg_idx in range(len(path_distances) - 1):
            start_d, end_d = path_distances[seg_idx], path_distances[seg_idx + 1]
            if start_d <= target <= end_d:
                ratio = 0.0 if end_d == start_d else (target - start_d) / (end_d - start_d)
                x1, y1 = path_positions[seg_idx]
                x2, y2 = path_positions[seg_idx + 1]
                marker_coords.append((x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio))
                break
        else:
            marker_coords.append(path_positions[-1])

    return marker_coords


def parse_route(filename: str) -> Tuple[int, int]:
    parts = Path(filename).stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    return int(parts[0]), int(parts[1])


def build_full_trajectory(
    highlight_indices: Sequence[int],
    marker_coords: Sequence[Tuple[float, float]],
    total_len: int,
) -> List[Tuple[float, float]]:
    """각 timestep을 마커 좌표 사이 선형 보간으로 복원."""
    trajectory: List[Tuple[float, float]] = []
    marker_idx = 0

    for t in range(total_len):
        if t <= highlight_indices[0]:
            trajectory.append(marker_coords[0])
            continue
        if t >= highlight_indices[-1]:
            trajectory.append(marker_coords[-1])
            continue

        while (
            marker_idx < len(highlight_indices) - 1
            and t > highlight_indices[marker_idx + 1]
        ):
            marker_idx += 1

        left_idx = highlight_indices[marker_idx]
        right_idx = highlight_indices[marker_idx + 1]
        p_left = marker_coords[marker_idx]
        p_right = marker_coords[marker_idx + 1]

        if right_idx == left_idx:
            trajectory.append(p_left)
            continue

        ratio = (t - left_idx) / (right_idx - left_idx)
        x = p_left[0] + (p_right[0] - p_left[0]) * ratio
        y = p_left[1] + (p_right[1] - p_left[1]) * ratio
        trajectory.append((x, y))

    return trajectory


def window_heading(traj: Sequence[Tuple[float, float]], start_idx: int, end_idx: int) -> Tuple[float, float]:
    """윈도우 마지막 구간 이동 벡터 계산."""
    tail = end_idx - 1
    lookback = min(5, tail - start_idx)
    if lookback <= 0:
        return 0.0, 0.0

    prev_idx = tail - lookback
    dx = traj[tail][0] - traj[prev_idx][0]
    dy = traj[tail][1] - traj[prev_idx][1]

    if abs(dx) < 1e-6 and abs(dy) < 1e-6 and prev_idx > start_idx:
        prev_idx = start_idx
        dx = traj[tail][0] - traj[prev_idx][0]
        dy = traj[tail][1] - traj[prev_idx][1]

    return dx, dy


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} 경로가 없습니다.")

    node_positions, graph = load_nodes()
    path_cache: Dict[Tuple[int, int], List[int]] = {}
    coverage: Dict[Tuple[float, float], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    files = sorted(DATA_DIR.glob("*.csv"))

    for csv_path in tqdm(files, desc="Analyzing markers"):
        try:
            start_node, end_node = parse_route(csv_path.name)
        except ValueError:
            continue

        if start_node not in node_positions or end_node not in node_positions:
            continue

        key = (start_node, end_node)
        if key not in path_cache:
            if not nx.has_path(graph, start_node, end_node):
                continue
            path_cache[key] = nx.shortest_path(graph, start_node, end_node, weight="weight")
        node_path = path_cache[key]

        df = pd.read_csv(csv_path)
        if "Highlighted" not in df.columns:
            continue

        highlighted_indices: List[int] = df.index[df["Highlighted"] == True].tolist()  # noqa: E712
        if len(highlighted_indices) < 2:
            continue

        marker_coords = calculate_path_coordinates(node_path, len(highlighted_indices), node_positions)

        full_trajectory = build_full_trajectory(highlighted_indices, marker_coords, len(df))
        if len(full_trajectory) < WINDOW_SIZE:
            continue

        for start_idx in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
            end_idx = start_idx + WINDOW_SIZE
            final_pos = full_trajectory[end_idx - 1]
            dx, dy = window_heading(full_trajectory, start_idx, end_idx)
            direction = direction_label(dx, dy)
            snapped_x = quantize_coord(final_pos[0])
            snapped_y = quantize_coord(final_pos[1])
            coverage[(snapped_x, snapped_y)][direction] += 1

    rows = []
    for (x_m, y_m), dir_counts in coverage.items():
        for direction, count in dir_counts.items():
            rows.append(
                {
                    "x_m": x_m,
                    "y_m": y_m,
                    "direction": direction,
                    "count": count,
                    "need_for_10": max(0, MIN_TARGET - count),
                    "need_for_20": max(0, GOAL_TARGET - count),
                }
            )

    if not rows:
        raise RuntimeError("집계된 마커가 없습니다. law_data 내용을 확인하세요.")

    df_out = pd.DataFrame(rows).sort_values(by=["count", "x_m", "y_m"]).reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    deficit_10 = (df_out["count"] < MIN_TARGET).sum()
    deficit_20 = (df_out["count"] < GOAL_TARGET).sum()
    print(f"총 (그리드, 방향) 조합: {len(df_out):,}")
    print(f"  <{MIN_TARGET}개 조합: {deficit_10:,}개")
    print(f"  <{GOAL_TARGET}개 조합: {deficit_20:,}개")
    print(f"결과 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
