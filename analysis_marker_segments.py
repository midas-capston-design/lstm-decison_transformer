#!/usr/bin/env python3
"""
Highlighted 마커 기반 from→to 세그먼트 카운트 분석.

각 law_data CSV에서 Highlighted=True 마커를 추출해 노드 그래프 경로 상 위치를 복원하고,
연속된 마커 쌍 (from→to)을 하나의 세그먼트로 본 뒤 동일 좌표/방향 조합별 샘플 수를 집계한다.
결과는 results/directional_marker_segments.csv에 저장된다.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

GRID_SIZE = 0.9
GRID_OFFSET = 0.45
CONNECTION_THRESHOLD = 5.0
WRONG_CONNECTIONS = {(10, 28), (24, 25)}
MIN_TARGET = 10
GOAL_TARGET = 20

DATA_DIR = Path("law_data")
NODES_PATH = Path("nodes_final.csv")
OUTPUT_CSV = Path("results/directional_marker_segments.csv")


def load_nodes() -> Tuple[Dict[int, Tuple[float, float]], nx.Graph]:
    nodes_df = pd.read_csv(NODES_PATH)
    node_positions = {
        int(row.id): (float(row.x_m), float(row.y_m)) for row in nodes_df.itertuples()
    }

    graph = nx.Graph()
    for node_id in node_positions:
        graph.add_node(node_id)

    ids = sorted(node_positions.keys())
    for i, node1 in enumerate(ids):
        for node2 in ids[i + 1 :]:
            pos1, pos2 = node_positions[node1], node_positions[node2]
            dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
            if dist <= CONNECTION_THRESHOLD:
                graph.add_edge(node1, node2, weight=dist)

    for bad in WRONG_CONNECTIONS:
        if graph.has_edge(*bad):
            graph.remove_edge(*bad)

    return node_positions, graph


def quantize_coord(value: float) -> float:
    snapped = GRID_OFFSET + GRID_SIZE * round((value - GRID_OFFSET) / GRID_SIZE)
    return round(snapped, 2)


def direction_label(dx: float, dy: float) -> str:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "정지"
    if abs(dx) >= abs(dy):
        return "동행(→)" if dx > 0 else "서행(←)"
    return "북행(↑)" if dy > 0 else "남행(↓)"


def calculate_path_coordinates(
    node_path: Sequence[int],
    num_markers: int,
    node_positions: Dict[int, Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    if num_markers <= 0:
        return [], []

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
    marker_edges: List[Tuple[int, int]] = []

    for m_idx in range(num_markers):
        if num_markers == 1:
            marker_coords.append(path_positions[0])
            marker_edges.append((node_path[0], node_path[0]))
            continue
        target = (m_idx / (num_markers - 1)) * total_dist
        for seg_idx in range(len(path_distances) - 1):
            start_d, end_d = path_distances[seg_idx], path_distances[seg_idx + 1]
            if start_d <= target <= end_d:
                ratio = 0.0 if end_d == start_d else (target - start_d) / (end_d - start_d)
                x1, y1 = path_positions[seg_idx]
                x2, y2 = path_positions[seg_idx + 1]
                marker_coords.append((x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio))
                marker_edges.append((node_path[seg_idx], node_path[seg_idx + 1]))
                break
        else:
            marker_coords.append(path_positions[-1])
            marker_edges.append((node_path[-2], node_path[-1]))

    return marker_coords, marker_edges


def parse_route(filename: str) -> Tuple[int, int]:
    parts = Path(filename).stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    return int(parts[0]), int(parts[1])


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} 경로가 없습니다.")

    node_positions, graph = load_nodes()
    path_cache: Dict[Tuple[int, int], List[int]] = {}
    segments = defaultdict(int)

    files = sorted(DATA_DIR.glob("*.csv"))
    for csv_path in tqdm(files, desc="Analyzing marker segments"):
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

        marker_coords, marker_edges = calculate_path_coordinates(node_path, len(highlighted_indices), node_positions)

        for idx in range(len(marker_coords) - 1):
            start = marker_coords[idx]
            end = marker_coords[idx + 1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            direction = direction_label(dx, dy)
            from_x = quantize_coord(start[0])
            from_y = quantize_coord(start[1])
            to_x = quantize_coord(end[0])
            to_y = quantize_coord(end[1])
            from_node, to_node = marker_edges[idx]
            key_seg = (from_node, to_node, from_x, from_y, to_x, to_y, direction)
            segments[key_seg] += 1

    if not segments:
        raise RuntimeError("세그먼트를 집계하지 못했습니다. law_data를 확인하세요.")

    rows = []
    for (from_node, to_node, from_x, from_y, to_x, to_y, direction), count in segments.items():
        rows.append(
            {
                "from_node": from_node,
                "to_node": to_node,
                "from_x": from_x,
                "from_y": from_y,
                "to_x": to_x,
                "to_y": to_y,
                "direction": direction,
                "count": count,
                "need_for_10": max(0, MIN_TARGET - count),
                "need_for_20": max(0, GOAL_TARGET - count),
            }
        )

    df_out = pd.DataFrame(rows).sort_values(by=["count", "from_x", "from_y"]).reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"총 마커 세그먼트 조합: {len(df_out):,}")
    print(f"  <{MIN_TARGET}개: {int((df_out['count'] < MIN_TARGET).sum()):,}개")
    print(f"  <{GOAL_TARGET}개: {int((df_out['count'] < GOAL_TARGET).sum()):,}개")
    print(f"결과 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
