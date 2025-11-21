#!/usr/bin/env python3
"""
하나의 스크립트로 데이터 전처리 + Hyena 학습을 처리한다.

사용 예:
  - 데이터 준비:  python pipeline.py preprocess --law-dir law_data --nodes nodes_final.csv --output data
  - 학습:        python pipeline.py train --data-dir data --nodes nodes_final.csv --epochs 50
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pywt  # Wavelet transform
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# -----------------------------
# 전처리 관련 상수
# -----------------------------
WINDOW_SIZE = 250
STRIDE = 50
GRID_SIZE = 0.45  # 목표 정확도에 맞춤 (0.9 → 0.45)
STEP_DISTANCE = 0.45
BUTTON_COLUMNS = ("Highlighted", "RightAngle")
CONNECTION_THRESHOLD = 5.0
WRONG_CONNECTIONS = {frozenset((10, 28)), frozenset((24, 25))}
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
TARGET_THRESHOLD = 1.35
BASE_MAG = (-33.0, -15.0, -42.0)  # 지자기 평균값 (정규화 기준)

# 좌표 정규화 범위 (nodes_final.csv 기준)
COORD_BOUNDS = {
    "min_x": -85.5,
    "max_x": 0.0,
    "min_y": -9.0,
    "max_y": 9.0,
}

# -----------------------------
# 공통 유틸
# -----------------------------


def read_nodes(path: Path) -> Dict[int, Tuple[float, float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return {int(row["id"]): (float(row["x_m"]), float(row["y_m"])) for row in reader}


def build_graph(nodes: Dict[int, Tuple[float, float]]) -> Dict[int, List[Tuple[int, float]]]:
    adj = {node: [] for node in nodes}
    node_items = list(nodes.items())
    for i, (id1, (x1, y1)) in enumerate(node_items):
        for id2, (x2, y2) in node_items[i + 1 :]:
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist <= CONNECTION_THRESHOLD and frozenset((id1, id2)) not in WRONG_CONNECTIONS:
                adj[id1].append((id2, dist))
                adj[id2].append((id1, dist))
    return adj


def shortest_path(adj: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> List[int] | None:
    import heapq

    if start == end:
        return [start]
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    visited = set()
    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            break
        for neigh, w in adj[node]:
            nd = d + w
            if neigh not in dist or nd < dist[neigh]:
                dist[neigh] = nd
                prev[neigh] = node
                heapq.heappush(pq, (nd, neigh))
    if end not in dist:
        return None
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def build_segments(path_nodes: List[int], positions: Dict[int, Tuple[float, float]]):
    segments = []
    total = 0.0
    for n1, n2 in zip(path_nodes, path_nodes[1:]):
        p1 = positions[n1]
        p2 = positions[n2]
        dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        segments.append((n1, n2, p1, p2, dist))
        total += dist
    return segments, total


def coordinate_at_distance(segments, distance):
    traversed = 0.0
    for n1, n2, p1, p2, dist in segments:
        if distance <= traversed + dist or (n1, n2) == segments[-1][:2]:
            seg_pos = min(max(distance - traversed, 0.0), dist)
            frac = 0.0 if dist == 0 else seg_pos / dist
            x = p1[0] + frac * (p2[0] - p1[0])
            y = p1[1] + frac * (p2[1] - p1[1])
            sub_idx = int(round(seg_pos / GRID_SIZE))
            return (x, y), (n1, n2, sub_idx)
        traversed += dist
    n1, n2, _, p2, last_dist = segments[-1]
    sub_idx = int(round(last_dist / GRID_SIZE))
    return p2, (n1, n2, sub_idx)


def quantize_coord(coord: Tuple[float, float]) -> Tuple[float, float]:
    x = round(round(coord[0] / GRID_SIZE) * GRID_SIZE, 6)
    y = round(round(coord[1] / GRID_SIZE) * GRID_SIZE, 6)
    return (x, y)


def normalize_mag(value: float, mean: float, std: float = 5.0) -> float:
    """Z-score 정규화로 정보 보존 (기존 tanh는 미세한 변화 손실)"""
    return (value - mean) / std


def normalize_coord(x: float, y: float) -> Tuple[float, float]:
    """좌표를 [0, 1] 범위로 정규화"""
    norm_x = (x - COORD_BOUNDS["min_x"]) / (COORD_BOUNDS["max_x"] - COORD_BOUNDS["min_x"])
    norm_y = (y - COORD_BOUNDS["min_y"]) / (COORD_BOUNDS["max_y"] - COORD_BOUNDS["min_y"])
    return (norm_x, norm_y)


def denormalize_coord(norm_x: float, norm_y: float) -> Tuple[float, float]:
    """정규화된 좌표를 원래 미터 단위로 복원"""
    x = norm_x * (COORD_BOUNDS["max_x"] - COORD_BOUNDS["min_x"]) + COORD_BOUNDS["min_x"]
    y = norm_y * (COORD_BOUNDS["max_y"] - COORD_BOUNDS["min_y"]) + COORD_BOUNDS["min_y"]
    return (x, y)


def angle_to_feature(value: float) -> float:
    return math.sin(math.radians(value))


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    mode: str = "soft",
) -> np.ndarray:
    """
    웨이블렛 기반 신호 디노이징 (Wavelet Denoising)

    Args:
        signal: 1D 신호 (예: MagX 시계열)
        wavelet: 웨이블렛 종류 (db4=Daubechies 4, 부드러운 신호에 적합)
        level: 분해 레벨 (3-5 추천, 높을수록 더 많은 주파수 대역 분리)
        mode: threshold 모드 ('soft'=부드럽게, 'hard'=강하게)

    Returns:
        디노이징된 신호
    """
    # 신호가 너무 짧으면 그대로 반환
    if len(signal) < 2 ** (level + 1):
        return signal

    # 1. 웨이블렛 분해 (Decomposition)
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 2. 노이즈 추정 (MAD - Median Absolute Deviation)
    # 가장 고주파 성분(detail coefficients)에서 노이즈 레벨 추정
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # 3. Threshold 계산 (Universal threshold)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # 4. Thresholding (노이즈 제거)
    # 작은 계수는 노이즈로 간주하여 제거
    coeffs_thresh = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]

    # 5. 재구성 (Reconstruction)
    denoised = pywt.waverec(coeffs_thresh, wavelet)

    # 경계 효과로 길이가 약간 달라질 수 있음
    return denoised[:len(signal)]


def wavelet_denoise_multivariate(
    signals: List[np.ndarray],
    wavelet: str = "db4",
    level: int = 3,
) -> List[np.ndarray]:
    """
    여러 센서 신호를 동시에 디노이징 (MagX, MagY, MagZ 등)
    """
    return [wavelet_denoise(sig, wavelet, level) for sig in signals]


def extract_button_distances(rows: List[List[str]], col_idx: Dict[str, int]) -> List[float] | None:
    if not any(col in col_idx for col in BUTTON_COLUMNS):
        return None

    def is_true(val: str) -> bool:
        return str(val).strip().lower() in ("1", "true")

    distances = []
    last_state = False
    steps = 0
    for row in rows:
        state = False
        for col in BUTTON_COLUMNS:
            if col in col_idx and is_true(row[col_idx[col]]):
                state = True
                break
        if state and not last_state:
            steps += 1
        distances.append(steps * STEP_DISTANCE)
        last_state = state

    if steps == 0:
        return None
    return distances


def process_csv(
    file_path: Path,
    positions: Dict[int, Tuple[float, float]],
    graph: Dict[int, List[Tuple[int, float]]],
    feature_mode: str = "full",
) -> Tuple[List[List[float]], List[Tuple[float, float]], List[str]]:
    with file_path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) <= 1:
        return [], [], []
    header = rows[0]
    data = rows[1:]
    col_idx = {name: idx for idx, name in enumerate(header)}

    start, end, *_ = file_path.stem.split("_") + [None, None]
    start_node = int(start)
    end_node = int(end)
    path_nodes = shortest_path(graph, start_node, end_node)
    if path_nodes is None:
        print(f"⚠️  skip (길 없음): {file_path}")
        return [], [], []

    segments, total_dist = build_segments(path_nodes, positions)
    distances = extract_button_distances(data, col_idx)

    coords: List[Tuple[float, float]] = []
    tags: List[str] = []

    if len(segments) == 0 or total_dist == 0.0:
        base_coord = positions[path_nodes[0]]
        coords = [base_coord] * len(data)
        tags = [f"{start_node}->{end_node}"] * len(data)
    else:
        if distances is None or len(distances) != len(data):
            distances = [
                (i / max(1, len(data) - 1)) * total_dist for i in range(len(data))
            ]
        for dist in distances:
            coord, (n1, n2, _) = coordinate_at_distance(segments, dist)
            coords.append(coord)
            tags.append(f"{n1}->{n2}")

    # 1. 먼저 raw 센서 데이터 추출
    raw_magx = np.array([float(row[col_idx["MagX"]]) for row in data])
    raw_magy = np.array([float(row[col_idx["MagY"]]) for row in data])
    raw_magz = np.array([float(row[col_idx["MagZ"]]) for row in data])

    # 2. 웨이블렛 디노이징 적용 (노이즈 제거)
    clean_magx = wavelet_denoise(raw_magx, wavelet="db4", level=3)
    clean_magy = wavelet_denoise(raw_magy, wavelet="db4", level=3)
    clean_magz = wavelet_denoise(raw_magz, wavelet="db4", level=3)

    # 3. 정규화 및 특징 변환
    features = []

    if feature_mode == "mag4":
        # 지자기 4개: MagX, MagY, MagZ, Magnitude
        for i in range(len(data)):
            magx = normalize_mag(clean_magx[i], BASE_MAG[0])
            magy = normalize_mag(clean_magy[i], BASE_MAG[1])
            magz = normalize_mag(clean_magz[i], BASE_MAG[2])
            # Magnitude 계산 (원본 값 기준)
            mag_magnitude = math.sqrt(clean_magx[i]**2 + clean_magy[i]**2 + clean_magz[i]**2)
            # Magnitude도 정규화 (평균 50 기준)
            mag_magnitude_norm = (mag_magnitude - 50.0) / 10.0
            feat = [magx, magy, magz, mag_magnitude_norm]
            features.append(feat)

    elif feature_mode == "full":
        # 기존 6개: MagX, MagY, MagZ, Pitch, Roll, Yaw
        raw_pitch = np.array([float(row[col_idx["Pitch"]]) for row in data])
        raw_roll = np.array([float(row[col_idx["Roll"]]) for row in data])
        raw_yaw = np.array([float(row[col_idx["Yaw"]]) for row in data])

        clean_pitch = wavelet_denoise(raw_pitch, wavelet="db4", level=2)
        clean_roll = wavelet_denoise(raw_roll, wavelet="db4", level=2)
        clean_yaw = wavelet_denoise(raw_yaw, wavelet="db4", level=2)

        for i in range(len(data)):
            magx = normalize_mag(clean_magx[i], BASE_MAG[0])
            magy = normalize_mag(clean_magy[i], BASE_MAG[1])
            magz = normalize_mag(clean_magz[i], BASE_MAG[2])
            pitch = angle_to_feature(clean_pitch[i])
            roll = angle_to_feature(clean_roll[i])
            yaw = angle_to_feature(clean_yaw[i])
            feat = [magx, magy, magz, pitch, roll, yaw]
            features.append(feat)

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    return features, coords, tags


def create_full_sequence(
    features: List[List[float]],
    coords: List[Tuple[float, float]],
    tags: List[str],
    csv_path: Path,
) -> Dict | None:
    """전체 경로를 하나의 시퀀스로 생성 (seq2seq)"""
    if len(features) < 50:  # 최소 길이 체크
        return None

    # 전체 trajectory의 고유 edge path (방향성 포함)
    # 예: "1->2->3->...->11"
    start_end = csv_path.stem.split("_")[:2]
    edge_path = f"{start_end[0]}->{start_end[1]}"

    # 모든 좌표를 양자화 후 정규화
    quantized_coords = [quantize_coord(c) for c in coords]
    normalized_coords = [normalize_coord(x, y) for x, y in quantized_coords]

    return {
        "features": features,  # (seq_len, 6)
        "targets": normalized_coords,  # (seq_len, 2) - 정규화된 좌표 [0, 1]
        "edge_path": edge_path,  # "start->end"
        "seq_len": len(features),
    }


def generate_virtual_sequence(
    edge_path: str,
    positions: Dict[int, Tuple[float, float]],
    graph: Dict[int, List[Tuple[int, float]]],
    min_len: int = 500,
    max_len: int = 3000,
    feature_mode: str = "full",
) -> Dict | None:
    """가상의 전체 경로 시퀀스 생성 (seq2seq)"""
    start_node, end_node = map(int, edge_path.split("->"))
    if start_node not in positions or end_node not in positions:
        return None

    # 최단 경로 찾기
    path_nodes = shortest_path(graph, start_node, end_node)
    if path_nodes is None or len(path_nodes) < 2:
        return None

    # 경로 세그먼트 생성
    segments, total_dist = build_segments(path_nodes, positions)
    if total_dist == 0:
        return None

    # 랜덤 시퀀스 길이 (실제 측정 데이터와 유사하게)
    seq_len = random.randint(min_len, max_len)

    features = []
    coords = []

    # 전체 경로를 따라 시뮬레이션
    for step in range(seq_len):
        frac = step / max(1, seq_len - 1)
        dist = frac * total_dist
        coord, _ = coordinate_at_distance(segments, dist)
        coords.append(coord)

        # 현재 위치에서의 방향 계산 (이전 step과 비교)
        if step > 0:
            dx = coords[step][0] - coords[step - 1][0]
            dy = coords[step][1] - coords[step - 1][1]
            heading = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0
        else:
            # 첫 스텝은 경로 전체 방향 사용
            dx = positions[end_node][0] - positions[start_node][0]
            dy = positions[end_node][1] - positions[start_node][1]
            heading = math.atan2(dy, dx)

        heading_deg = math.degrees(heading)

        # 센서 시뮬레이션
        raw_magx = BASE_MAG[0] + 3 * math.cos(heading) + random.gauss(0, 1.5)
        raw_magy = BASE_MAG[1] + 3 * math.sin(heading) + random.gauss(0, 1.5)
        raw_magz = BASE_MAG[2] + random.gauss(0, 0.8)
        magx = normalize_mag(raw_magx, BASE_MAG[0])
        magy = normalize_mag(raw_magy, BASE_MAG[1])
        magz = normalize_mag(raw_magz, BASE_MAG[2])

        if feature_mode == "mag4":
            mag_magnitude = math.sqrt(raw_magx**2 + raw_magy**2 + raw_magz**2)
            mag_magnitude_norm = (mag_magnitude - 50.0) / 10.0
            features.append([magx, magy, magz, mag_magnitude_norm])
        elif feature_mode == "full":
            pitch = angle_to_feature(random.gauss(0, 3.0))
            roll = angle_to_feature(random.gauss(0, 3.0))
            yaw = angle_to_feature(heading_deg + random.gauss(0, 5.0))
            features.append([magx, magy, magz, pitch, roll, yaw])

    quantized_coords = [quantize_coord(c) for c in coords]
    normalized_coords = [normalize_coord(x, y) for x, y in quantized_coords]

    return {
        "features": features,
        "targets": normalized_coords,  # 정규화된 좌표 [0, 1]
        "edge_path": edge_path,
        "seq_len": seq_len,
    }


def balance_with_virtual(
    samples: List[Dict],
    positions: Dict[int, Tuple[float, float]],
    graph: Dict[int, List[Tuple[int, float]]],
    min_samples_per_path: int = 3,
    feature_mode: str = "full",
) -> List[Dict]:
    """부족한 경로에 가상 시퀀스 추가 (seq2seq 방식)"""
    counts = defaultdict(int)
    for sample in samples:
        counts[sample["edge_path"]] += 1

    # 모든 가능한 경로 수집
    all_paths = set(counts.keys())
    for node, neighbors in graph.items():
        for neigh, _ in neighbors:
            all_paths.add(f"{node}->{neigh}")

    # 부족한 경로에 가상 데이터 추가
    for path in all_paths:
        needed = max(0, min_samples_per_path - counts[path])
        for _ in range(needed):
            synthetic = generate_virtual_sequence(path, positions, graph, feature_mode=feature_mode)
            if synthetic is None:
                break
            samples.append(synthetic)
            counts[path] += 1

    return samples


def stratified_split(samples: List[Dict]):
    """경로별로 stratified split (seq2seq)"""
    buckets = defaultdict(list)
    for sample in samples:
        buckets[sample["edge_path"]].append(sample)

    train_set, val_set, test_set = [], [], []
    rng = random.Random(42)

    # 데이터 부족 경로 추적
    insufficient_paths = []

    for path, items in buckets.items():
        rng.shuffle(items)
        total = len(items)

        # 데이터 부족 체크 (5개 미만은 제대로 분할 불가)
        if total < 5:
            insufficient_paths.append((path, total))

        train_n = max(1, int(round(total * TRAIN_RATIO)))
        val_n = max(1, int(round(total * VAL_RATIO)))
        test_n = total - train_n - val_n

        # 최소 1개씩은 보장 (필수)
        if test_n <= 0:
            test_n = 1
            if val_n > 1:
                val_n -= 1
            else:
                train_n = max(1, train_n - 1)

        train_set.extend(items[:train_n])
        val_set.extend(items[train_n : train_n + val_n])
        test_set.extend(items[train_n + val_n : train_n + val_n + test_n])

    # 데이터 부족 경고
    if insufficient_paths:
        print(f"\n⚠️  데이터 부족 경로 발견: {len(insufficient_paths)}개")
        print("다음 경로들은 추가 측정이 필요합니다 (권장: 5개 이상):\n")
        for path, count in sorted(insufficient_paths, key=lambda x: x[1]):
            print(f"  - {path}: {count}개 (부족: {5-count}개)")
        print()

    return train_set, val_set, test_set


def save_jsonl(path: Path, samples: Iterable[Dict]):
    """Seq2seq 샘플 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for sample in samples:
            f.write(
                json.dumps(
                    {
                        "features": sample["features"],
                        "targets": sample["targets"],
                        "edge_path": sample["edge_path"],
                        "seq_len": sample["seq_len"],
                    }
                )
            )
            f.write("\n")


def preprocess(args):
    """Seq2seq 방식으로 데이터 전처리"""
    law_dir = Path(args.law_dir)
    output_dir = Path(args.output)
    nodes_path = Path(args.nodes)
    min_samples_per_path = getattr(args, "min_samples_per_path", 3)
    feature_mode = getattr(args, "feature_mode", "full")

    positions = read_nodes(nodes_path)
    graph = build_graph(positions)

    all_samples = []
    for csv_file in sorted(law_dir.glob("*.csv")):
        feats, coords, tags = process_csv(csv_file, positions, graph, feature_mode=feature_mode)
        if not feats:
            continue

        # 전체 경로를 하나의 시퀀스로 생성
        sample = create_full_sequence(feats, coords, tags, csv_file)
        if sample is not None:
            all_samples.append(sample)

    if not all_samples:
        raise RuntimeError("생성된 샘플이 없습니다. 입력 데이터를 확인하세요.")

    print(f"📊 실제 데이터: {len(all_samples)}개 경로 (feature_mode={feature_mode})")

    # 부족한 경로에 가상 데이터 추가
    if min_samples_per_path > 0:
        all_samples = balance_with_virtual(all_samples, positions, graph, min_samples_per_path, feature_mode=feature_mode)

    # 경로별 stratified split
    train_set, val_set, test_set = stratified_split(all_samples)

    # 저장
    save_jsonl(output_dir / "train.jsonl", train_set)
    save_jsonl(output_dir / "val.jsonl", val_set)
    save_jsonl(output_dir / "test.jsonl", test_set)

    # Summary
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w") as f:
        f.write("split,count,avg_seq_len\n")
        train_avg = sum(s["seq_len"] for s in train_set) / len(train_set) if train_set else 0
        val_avg = sum(s["seq_len"] for s in val_set) / len(val_set) if val_set else 0
        test_avg = sum(s["seq_len"] for s in test_set) / len(test_set) if test_set else 0
        f.write(f"train,{len(train_set)},{train_avg:.0f}\n")
        f.write(f"val,{len(val_set)},{val_avg:.0f}\n")
        f.write(f"test,{len(test_set)},{test_avg:.0f}\n")

    print(f"✅ 전처리 완료 (seq2seq)")
    print(f"   Train: {len(train_set)}개 경로 (평균 {train_avg:.0f} 타임스텝)")
    print(f"   Val:   {len(val_set)}개 경로 (평균 {val_avg:.0f} 타임스텝)")
    print(f"   Test:  {len(test_set)}개 경로 (평균 {test_avg:.0f} 타임스텝)")


# -----------------------------
# 학습 파이프라인 (Hyena)
# -----------------------------


def time_warp_tensor(x: torch.Tensor, scale: float = 0.2) -> torch.Tensor:
    if x.size(0) < 2 or scale <= 0.0:
        return x
    warp = random.uniform(1 - scale, 1 + scale)
    new_len = max(2, int(x.size(0) * warp))
    interp = F.interpolate(
        x.unsqueeze(0).transpose(1, 2),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2).squeeze(0)
    if new_len >= x.size(0):
        return interp[: x.size(0)]
    pad = x.size(0) - new_len
    return torch.cat([interp, interp[-1:].repeat(pad, 1)], dim=0)


def time_mask_tensor(x: torch.Tensor, ratio: float = 0.05) -> torch.Tensor:
    length = x.size(0)
    mask_len = max(1, int(length * ratio))
    start = random.randint(0, max(0, length - mask_len))
    x[start : start + mask_len] = 0
    return x


def apply_sequential_augments(x: torch.Tensor) -> torch.Tensor:
    x = x + torch.randn_like(x) * 0.01
    if random.random() < 0.5:
        x = time_warp_tensor(x)
    if random.random() < 0.5:
        x = time_mask_tensor(x)
    dropout_mask = torch.rand(x.shape[0], 1, device=x.device) < 0.02
    x = torch.where(dropout_mask, torch.zeros_like(x), x)
    return x


class Seq2SeqDataset(Dataset):
    """Seq2seq 방식 데이터셋: 전체 경로 로드"""

    def __init__(self, path: Path, augment: bool = False):
        self.samples = []
        self.edge_to_id = {}  # edge_path → ID 매핑

        with path.open() as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)

                    # edge_path를 ID로 변환
                    edge_path = sample["edge_path"]
                    if edge_path not in self.edge_to_id:
                        self.edge_to_id[edge_path] = len(self.edge_to_id)

        if not self.samples:
            raise RuntimeError(f"{path} 에 데이터가 없습니다.")

        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        x = torch.tensor(sample["features"], dtype=torch.float32)  # (seq_len, 6)
        y = torch.tensor(sample["targets"], dtype=torch.float32)  # (seq_len, 2)
        edge_id = self.edge_to_id[sample["edge_path"]]

        if self.augment:
            x = apply_sequential_augments(x)

        return x, y, edge_id


def collate_seq2seq(batch):
    """가변 길이 시퀀스를 배치로 묶기 (padding 사용)"""
    xs, ys, edge_ids = zip(*batch)

    # 최대 길이 찾기
    max_len = max(x.size(0) for x in xs)

    # Padding
    xs_padded = []
    ys_padded = []
    masks = []

    for x, y in zip(xs, ys):
        seq_len = x.size(0)
        pad_len = max_len - seq_len

        # Padding (0으로)
        x_pad = F.pad(x, (0, 0, 0, pad_len))  # (max_len, 6)
        y_pad = F.pad(y, (0, 0, 0, pad_len))  # (max_len, 2)

        # Mask (True = valid, False = padding)
        mask = torch.cat([torch.ones(seq_len, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)])

        xs_padded.append(x_pad)
        ys_padded.append(y_pad)
        masks.append(mask)

    xs_batch = torch.stack(xs_padded)  # (batch, max_len, 6)
    ys_batch = torch.stack(ys_padded)  # (batch, max_len, 2)
    masks_batch = torch.stack(masks)  # (batch, max_len)
    edge_ids_batch = torch.tensor(edge_ids, dtype=torch.long)  # (batch,)

    return xs_batch, ys_batch, edge_ids_batch, masks_batch


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]  # (seq_len, dim)


class ImplicitFilter(nn.Module):
    """작은 MLP로 긴 필터 생성 (Hyena의 핵심)"""

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        # 위치 [0, 1, ..., seq_len-1] 생성
        positions = torch.linspace(0, 1, seq_len, device=next(self.parameters()).device)
        positions = positions.unsqueeze(-1)  # (seq_len, 1)
        filter_weights = self.mlp(positions)  # (seq_len, dim)
        return filter_weights


class HyenaOperator(nn.Module):
    """진짜 Hyena: Implicit filter + Short conv + FFT long conv + Multiple gates"""

    def __init__(self, dim: int, order: int = 2):
        super().__init__()
        self.dim = dim
        self.order = order  # gating paths 개수

        # Implicit long filter
        self.implicit_filter = ImplicitFilter(dim)

        # Short convolution (data-controlled)
        self.short_conv = nn.Conv1d(
            dim, dim, kernel_size=3, padding=1, groups=dim  # depthwise conv
        )

        # Projections for multiple paths (v, u, z, ...)
        self.in_proj = nn.Linear(dim, dim * (order + 1))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # Multiple paths projection
        proj = self.in_proj(x)  # (batch, seq_len, dim * (order+1))
        paths = proj.chunk(self.order + 1, dim=-1)  # List of (batch, seq_len, dim)

        v = paths[0]  # 값 경로

        # Implicit filter 생성
        filt = self.implicit_filter(seq_len)  # (seq_len, dim)

        # Short convolution (data-controlled)
        u_input = paths[1].transpose(1, 2)  # (batch, dim, seq_len)
        u_short = self.short_conv(u_input).transpose(1, 2)  # (batch, seq_len, dim)

        # FFT long convolution
        # filt와 u_short를 element-wise 곱한 뒤 FFT conv
        U = torch.fft.rfft(u_short, dim=1)  # (batch, freq, dim)
        Filt = torch.fft.rfft(filt.unsqueeze(0), n=seq_len, dim=1)  # (1, freq, dim)
        filtered = torch.fft.irfft(U * Filt, n=seq_len, dim=1)  # (batch, seq_len, dim)

        # Multiple gating: v * filtered * z (if order >= 2)
        output = v * filtered

        if self.order >= 2:
            z = paths[2]
            output = output * torch.sigmoid(z)

        return self.out_proj(output)


class HyenaBlock(nn.Module):
    """Hyena Block with normalization and residual"""

    def __init__(self, dim: int, order: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.hyena = HyenaOperator(dim, order=order)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)
        """
        h = self.norm(x)
        out = self.hyena(h)
        out = self.dropout(out)
        return out + x  # residual connection


class HyenaSeq2SeqModel(nn.Module):
    """Seq2seq Hyena: 가변 길이 입력 → 전체 trajectory 출력"""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        depth: int = 4,
        order: int = 2,
        dropout: float = 0.1,
        num_edge_types: int = 100,  # 방향성 인코딩을 위한 edge 타입 수
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 입력 projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Edge path embedding (방향성 인코딩)
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_dim)

        # Hyena blocks
        self.blocks = nn.ModuleList(
            [HyenaBlock(hidden_dim, order=order, dropout=dropout) for _ in range(depth)]
        )

        # Output head (각 타임스텝마다 좌표 예측)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # (x, y)
        )

    def forward(
        self, x: torch.Tensor, edge_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        edge_ids: (batch,) - edge path의 ID (방향성 구분)
        returns: (batch, seq_len, 2)
        """
        batch, seq_len, _ = x.shape

        # Input projection
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Add positional encoding
        pos = self.pos_encoding(seq_len)  # (seq_len, hidden_dim)
        h = h + pos.unsqueeze(0)  # broadcast

        # Add edge path embedding (방향성 정보)
        if edge_ids is not None:
            edge_emb = self.edge_embedding(edge_ids)  # (batch, hidden_dim)
            h = h + edge_emb.unsqueeze(1)  # broadcast to all timesteps

        # Hyena blocks
        for block in self.blocks:
            h = block(h)

        # Output head (각 타임스텝마다 좌표 예측)
        h = self.norm(h)
        coords = self.head(h)  # (batch, seq_len, 2)

        return coords


def seq2seq_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    huber_delta: float = 1.0,
    l2_weight: float = 0.1,
    penalty_weight: float = 0.5,
):
    """
    Seq2seq 손실 함수 (mask 적용)
    pred: (batch, seq_len, 2)
    target: (batch, seq_len, 2)
    mask: (batch, seq_len) - True = valid, False = padding
    """
    # 유효한 타임스텝만 선택
    pred_valid = pred[mask]  # (num_valid, 2)
    target_valid = target[mask]  # (num_valid, 2)

    if pred_valid.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Huber loss
    huber = F.huber_loss(pred_valid, target_valid, delta=huber_delta)

    # L2 loss
    l2 = F.mse_loss(pred_valid, target_valid)

    return huber + l2_weight * l2


def load_nodes_tensor(nodes_path: Path, device: torch.device) -> torch.Tensor:
    positions = read_nodes(nodes_path)
    coords = torch.tensor(list(positions.values()), dtype=torch.float32, device=device)
    return coords


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_positions: torch.Tensor,
    thresholds=(0.9, 1.35, 2.0),
    topk: int = 5,
):
    diff = torch.norm(pred - target, dim=1)
    rmse = torch.sqrt(F.mse_loss(pred, target))

    metrics = {}
    for thr in thresholds:
        metrics[f"top1_{thr}m"] = (diff <= thr).float().mean().item()

    if node_positions is not None:
        # 후보 노드 상위 K개의 거리
        pred_exp = pred.unsqueeze(1) - node_positions.unsqueeze(0)
        pred_nodes = torch.norm(pred_exp, dim=2)
        _, pred_idx = torch.topk(
            pred_nodes, k=min(topk, node_positions.shape[0]), largest=False
        )

        target_exp = target.unsqueeze(1) - node_positions.unsqueeze(0)
        target_dist, target_idx = torch.min(target_exp.norm(dim=2), dim=1, keepdim=True)

        for thr in thresholds:
            within_thr = target_dist.squeeze(1) <= thr
            match = (pred_idx == target_idx).any(dim=1)
            metrics[f"top5_{thr}m"] = (within_thr & match).float().mean().item()
    else:
        for thr in thresholds:
            metrics[f"top5_{thr}m"] = float("nan")

    metrics["rmse"] = rmse.item()
    metrics["avg_dist"] = diff.mean().item()
    return metrics


def run_epoch_seq2seq(model, loader, optimizer, device, train=True):
    """Seq2seq 학습/평가 루프"""
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    # 평가 메트릭 누적
    all_distances = []

    for batch in loader:
        x, y, edge_ids, mask = batch
        x = x.to(device)
        y = y.to(device)
        edge_ids = edge_ids.to(device)
        mask = mask.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x, edge_ids)  # (batch, seq_len, 2)
            loss = seq2seq_loss(pred, y, mask)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        # 거리 메트릭 계산 (유효한 타임스텝만) - 역정규화 후 미터 단위로
        pred_valid = pred[mask]
        target_valid = y[mask]

        # 역정규화하여 실제 미터 단위 거리 계산
        pred_denorm = torch.zeros_like(pred_valid)
        target_denorm = torch.zeros_like(target_valid)
        for i in range(pred_valid.size(0)):
            x_pred, y_pred = denormalize_coord(pred_valid[i, 0].item(), pred_valid[i, 1].item())
            x_tgt, y_tgt = denormalize_coord(target_valid[i, 0].item(), target_valid[i, 1].item())
            pred_denorm[i] = torch.tensor([x_pred, y_pred])
            target_denorm[i] = torch.tensor([x_tgt, y_tgt])

        distances = torch.norm(pred_denorm - target_denorm, dim=1)
        all_distances.extend(distances.detach().cpu().tolist())

    avg_loss = total_loss / total_samples
    all_distances = torch.tensor(all_distances)

    metrics = {
        "loss": avg_loss,
        "rmse": torch.sqrt(torch.mean(all_distances ** 2)).item(),
        "mae": torch.mean(all_distances).item(),
        "median": torch.median(all_distances).item(),
        "p90": torch.quantile(all_distances, 0.9).item(),
    }

    return metrics


def evaluate_with_sliding_window(model, dataset, device, window_size=250, stride=50):
    """슬라이딩 윈도우 방식으로 평가 (validation/test 모두 사용)

    Args:
        stride: 슬라이딩 간격 (작을수록 정확하지만 느림)
                - Validation: 100-150 권장 (속도)
                - Test: 50 권장 (정확도)
    """
    model.eval()
    all_distances = []

    with torch.no_grad():
        for sample in dataset.samples:
            feats = sample["features"]
            targets = sample["targets"]  # 정규화된 좌표
            edge_path = sample["edge_path"]
            edge_id = dataset.edge_to_id.get(edge_path, 0)

            if len(feats) < window_size:
                continue

            # 슬라이딩 윈도우
            for i in range(0, len(feats) - window_size + 1, stride):
                window_feat = feats[i : i + window_size]
                window_target = targets[i : i + window_size]

                # 텐서 변환
                x = torch.tensor(window_feat, dtype=torch.float32).unsqueeze(0).to(device)
                edge_tensor = torch.tensor([edge_id], dtype=torch.long).to(device)

                # 예측 (마지막 타임스텝만 사용)
                pred = model(x, edge_tensor)  # (1, window_size, 2)
                pred_last_norm = pred[0, -1, :].cpu().numpy()

                # 역정규화
                pred_last = denormalize_coord(pred_last_norm[0], pred_last_norm[1])
                target_last_norm = window_target[-1]
                target_last = denormalize_coord(target_last_norm[0], target_last_norm[1])

                # 거리 계산
                dist = math.hypot(pred_last[0] - target_last[0], pred_last[1] - target_last[1])
                all_distances.append(dist)

    if not all_distances:
        return {"rmse": float("inf"), "mae": float("inf"), "median": float("inf"), "p90": float("inf")}

    all_distances = torch.tensor(all_distances)
    metrics = {
        "rmse": torch.sqrt(torch.mean(all_distances ** 2)).item(),
        "mae": torch.mean(all_distances).item(),
        "median": torch.median(all_distances).item(),
        "p90": torch.quantile(all_distances, 0.9).item(),
    }

    return metrics


def train(args):
    """Seq2seq Hyena 모델 학습"""
    data_dir = Path(args.data_dir)
    nodes_path = Path(args.nodes)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} 가 없습니다. 먼저 preprocess를 실행하세요.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Seq2seq 데이터셋 로드 (augment 제거)
    train_ds = Seq2SeqDataset(train_path, augment=False)
    val_ds = Seq2SeqDataset(val_path, augment=False)
    test_ds = Seq2SeqDataset(test_path, augment=False)

    # Edge 타입 수 계산 (모든 split의 edge 합치기)
    all_edges = set()
    all_edges.update(train_ds.edge_to_id.keys())
    all_edges.update(val_ds.edge_to_id.keys())
    all_edges.update(test_ds.edge_to_id.keys())
    num_edge_types = len(all_edges)

    # Input dimension 자동 감지
    sample_features = train_ds.samples[0]["features"]
    input_dim = len(sample_features[0])

    print(f"📊 데이터 로드 완료:")
    print(f"   Train: {len(train_ds)}개 경로")
    print(f"   Val:   {len(val_ds)}개 경로")
    print(f"   Test:  {len(test_ds)}개 경로")
    print(f"   Edge types: {num_edge_types}개 (방향성 포함)")
    print(f"   Input dim: {input_dim}개 특징")

    # DataLoader (collate_fn 사용)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_seq2seq,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_seq2seq)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_seq2seq)

    # Seq2seq Hyena 모델
    model = HyenaSeq2SeqModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        order=getattr(args, "hyena_order", 2),
        dropout=args.dropout,
        num_edge_types=num_edge_types,
    ).to(device)

    print(f"🧠 모델: Hyena Seq2seq")
    print(f"   Input dim: {input_dim}")
    print(f"   Hidden dim: {args.hidden_dim}")
    print(f"   Depth: {args.depth}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    no_improve = 0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    print(f"\n🚀 학습 시작 (Epochs: {args.epochs})\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch_seq2seq(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch_seq2seq(model, val_loader, optimizer, device, train=False)
        scheduler.step()

        print(
            f"[Epoch {epoch:03d}] "
            f"TrainLoss={train_metrics['loss']:.4f} "
            f"ValLoss={val_metrics['loss']:.4f} | "
            f"RMSE={val_metrics['rmse']:.3f}m "
            f"MAE={val_metrics['mae']:.3f}m "
            f"Median={val_metrics['median']:.3f}m "
            f"P90={val_metrics['p90']:.3f}m"
        )

        if val_metrics["loss"] + 1e-4 < best_val:
            best_val = val_metrics["loss"]
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "edge_to_id": train_ds.edge_to_id,  # 저장
                },
                best_path,
            )
            print(f"   💾 Best model saved (loss={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n⏹️  Early stopping at epoch {epoch} (patience {args.patience})")
                break

    print(f"\n✅ 학습 완료. 베스트 체크포인트: {best_path}")

    # Best model 로드
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    # Test 평가 (슬라이딩 윈도우 방식)
    print(f"\n📈 Test 평가 중 (슬라이딩 윈도우: window=250, stride=50)...")
    test_metrics = evaluate_with_sliding_window(model, test_ds, device, window_size=250, stride=50)

    print(
        f"\n[Test Results - Sliding Window]\n"
        f"  RMSE:   {test_metrics['rmse']:.3f}m\n"
        f"  MAE:    {test_metrics['mae']:.3f}m\n"
        f"  Median: {test_metrics['median']:.3f}m\n"
        f"  P90:    {test_metrics['p90']:.3f}m\n"
    )


def inference(args):
    """250 윈도우 슬라이딩 추론"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 한국어 폰트 설정 (macOS/Linux/Windows 자동 감지)
    try:
        import platform
        system = platform.system()
        if system == "Darwin":  # macOS
            plt.rc("font", family="AppleGothic")
        elif system == "Windows":
            plt.rc("font", family="Malgun Gothic")
        else:  # Linux
            plt.rc("font", family="NanumGothic")
        plt.rc("axes", unicode_minus=False)  # 마이너스 기호 깨짐 방지
    except Exception:
        print("⚠️  한국어 폰트 설정 실패. 영문으로 표시됩니다.")

    checkpoint_path = Path(args.checkpoint)
    csv_path = Path(args.csv)
    nodes_path = Path(args.nodes)
    window_size = args.window_size

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {checkpoint_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일 없음: {csv_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 체크포인트 로드
    print(f"📦 체크포인트 로드: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 노드 및 그래프
    positions = read_nodes(nodes_path)
    graph = build_graph(positions)

    # Edge to ID 매핑 로드
    edge_to_id = checkpoint.get("edge_to_id", {})
    num_edge_types = len(edge_to_id)

    # 모델 로드
    model = HyenaSeq2SeqModel(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        order=2,
        dropout=0.0,  # 추론 시 dropout 없음
        num_edge_types=num_edge_types,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"✅ 모델 로드 완료")

    # CSV 처리
    print(f"📊 CSV 처리: {csv_path}")
    feats, coords, tags = process_csv(csv_path, positions, graph)

    if len(feats) < window_size:
        raise ValueError(f"데이터 길이({len(feats)})가 윈도우 크기({window_size})보다 작습니다.")

    # Edge path 추론 (파일명에서)
    start_end = csv_path.stem.split("_")[:2]
    edge_path = f"{start_end[0]}->{start_end[1]}"
    edge_id = edge_to_id.get(edge_path, 0)

    print(f"🔍 추론 시작 (윈도우={window_size}, 전체 길이={len(feats)})")

    # 슬라이딩 윈도우 추론
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for i in range(len(feats) - window_size + 1):
            window_feat = feats[i : i + window_size]
            window_coords = coords[i : i + window_size]

            # 텐서 변환
            x = torch.tensor(window_feat, dtype=torch.float32).unsqueeze(0).to(device)  # (1, window_size, 6)
            edge_tensor = torch.tensor([edge_id], dtype=torch.long).to(device)

            # 예측 (정규화된 좌표)
            pred = model(x, edge_tensor)  # (1, window_size, 2)
            pred_last_norm = pred[0, -1, :].cpu().numpy()  # 마지막 타임스텝만 (정규화됨)

            # 역정규화하여 미터 단위로 변환
            pred_last = denormalize_coord(pred_last_norm[0], pred_last_norm[1])

            predictions.append(pred_last)
            ground_truths.append(window_coords[-1])

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # 평가
    distances = np.linalg.norm(predictions - ground_truths, axis=1)
    rmse = np.sqrt(np.mean(distances ** 2))
    mae = np.mean(distances)
    median = np.median(distances)
    p90 = np.percentile(distances, 90)

    print(f"\n📈 추론 결과:")
    print(f"  샘플 수: {len(predictions)}")
    print(f"  RMSE:    {rmse:.3f}m")
    print(f"  MAE:     {mae:.3f}m")
    print(f"  Median:  {median:.3f}m")
    print(f"  P90:     {p90:.3f}m")

    # 시각화
    if not args.no_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Trajectory 비교
        ax = axes[0, 0]
        ax.plot(ground_truths[:, 0], ground_truths[:, 1], "b-", label="Ground Truth", alpha=0.7)
        ax.plot(predictions[:, 0], predictions[:, 1], "r--", label="Prediction", alpha=0.7)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Trajectory Comparison")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

        # 2. 시간에 따른 오차
        ax = axes[0, 1]
        ax.plot(distances, label="Distance Error")
        ax.axhline(y=rmse, color="r", linestyle="--", label=f"RMSE={rmse:.2f}m")
        ax.axhline(y=1.35, color="orange", linestyle=":", label="Target=1.35m")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Error (m)")
        ax.set_title("Error over Time")
        ax.legend()
        ax.grid(True)

        # 3. 오차 히스토그램
        ax = axes[1, 0]
        ax.hist(distances, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(x=mae, color="r", linestyle="--", label=f"MAE={mae:.2f}m")
        ax.axvline(x=median, color="g", linestyle="--", label=f"Median={median:.2f}m")
        ax.set_xlabel("Error (m)")
        ax.set_ylabel("Frequency")
        ax.set_title("Error Distribution")
        ax.legend()
        ax.grid(True)

        # 4. CDF
        ax = axes[1, 1]
        sorted_distances = np.sort(distances)
        cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
        ax.plot(sorted_distances, cdf * 100)
        ax.axvline(x=1.35, color="orange", linestyle=":", label="Target=1.35m")
        ax.axhline(y=90, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Error (m)")
        ax.set_ylabel("Cumulative Percentage (%)")
        ax.set_title("Cumulative Error Distribution")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        output_path = checkpoint_path.parent / f"inference_{csv_path.stem}.png"
        plt.savefig(output_path, dpi=150)
        print(f"\n💾 그래프 저장: {output_path}")

        if not args.no_show:
            plt.show()


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="지자기 기반 실내측위 파이프라인")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_parser = subparsers.add_parser("preprocess", help="CSV -> 윈도우 데이터 생성")
    prep_parser.add_argument("--law-dir", default="law_data")
    prep_parser.add_argument("--nodes", default="nodes_final.csv")
    prep_parser.add_argument("--output", default="data")
    prep_parser.add_argument("--min-samples-per-path", type=int, default=3)
    prep_parser.add_argument("--feature-mode", default="full", choices=["full", "mag4"],
                             help="Feature mode: full (6 features) or mag4 (4 features)")

    train_parser = subparsers.add_parser("train", help="Hyena 모델 학습")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--nodes", default="nodes_final.csv")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--depth", type=int, default=8)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--checkpoint-dir", default="checkpoints")
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--cpu", action="store_true")

    infer_parser = subparsers.add_parser("infer", help="250 윈도우 슬라이딩 추론")
    infer_parser.add_argument("--checkpoint", required=True, help="체크포인트 파일 경로")
    infer_parser.add_argument("--csv", required=True, help="추론할 CSV 파일 경로")
    infer_parser.add_argument("--nodes", default="nodes_final.csv")
    infer_parser.add_argument("--window-size", type=int, default=250)
    infer_parser.add_argument("--hidden-dim", type=int, default=256)
    infer_parser.add_argument("--depth", type=int, default=8)
    infer_parser.add_argument("--no-plot", action="store_true", help="그래프 생성 안 함")
    infer_parser.add_argument("--no-show", action="store_true", help="그래프 표시 안 함")
    infer_parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)

    if args.command == "preprocess":
        preprocess(args)
    elif args.command == "train":
        train(args)
    elif args.command == "infer":
        inference(args)


if __name__ == "__main__":
    main()
