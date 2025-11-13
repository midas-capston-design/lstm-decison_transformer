from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

BLOCKED_EDGES = {(10, 28), (28, 10), (24, 25), (25, 24)}


def load_node_graph(csv_path: str = "nodes_final.csv", connection_threshold: float = 5.0) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    df = pd.read_csv(csv_path)
    positions = {int(row["id"]): (float(row["x_m"]), float(row["y_m"])) for _, row in df.iterrows()}
    graph = nx.Graph()
    graph.add_nodes_from(positions.keys())

    for i, pi in positions.items():
        for j, pj in positions.items():
            if i >= j:
                continue
            dist = np.linalg.norm(np.array(pi) - np.array(pj))
            if dist <= connection_threshold and (i, j) not in BLOCKED_EDGES:
                graph.add_edge(i, j, weight=dist)
    return graph, positions


def nearest_node(coord: Sequence[float], positions: Dict[int, Tuple[float, float]]) -> int:
    coord_arr = np.array(coord)
    best_node = -1
    best_dist = float("inf")
    for node, pos in positions.items():
        dist = np.linalg.norm(coord_arr - np.array(pos))
        if dist < best_dist:
            best_dist = dist
            best_node = node
    return best_node


def _interpolate_node_path(node_path: List[int], positions: Dict[int, Tuple[float, float]], length: int) -> np.ndarray:
    coords = [np.array(positions[node]) for node in node_path]
    distances = [0.0]
    for i in range(1, len(coords)):
        distances.append(distances[-1] + np.linalg.norm(coords[i] - coords[i - 1]))
    total = distances[-1] if distances else 1.0
    samples = np.linspace(0.0, total, length)

    interp = []
    for s in samples:
        for i in range(1, len(distances)):
            if distances[i - 1] <= s <= distances[i]:
                t = (s - distances[i - 1]) / max(1e-6, distances[i] - distances[i - 1])
                point = coords[i - 1] + t * (coords[i] - coords[i - 1])
                interp.append(point)
                break
        else:
            interp.append(coords[-1])
    return np.stack(interp, axis=0)


def random_walk_paths(
    graph: nx.Graph,
    positions: Dict[int, Tuple[float, float]],
    start_coord: Sequence[float],
    num_paths: int,
    length: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    start_node = nearest_node(start_coord, positions)
    paths = []

    for _ in range(num_paths):
        node_path = [start_node]
        current = start_node
        steps = max(2, length // 25)
        for _ in range(steps):
            neighbors = list(graph.neighbors(current))
            if not neighbors:
                break
            current = int(rng.choice(neighbors))
            node_path.append(current)
        coords = _interpolate_node_path(node_path, positions, length)
        paths.append(coords)
    return np.stack(paths, axis=0)
