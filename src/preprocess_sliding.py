#!/usr/bin/env python3
"""Sliding Window 방식 전처리 - Causal Training용"""
import json
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pywt

# 정규화 기준값
BASE_MAG = (-33.0, -15.0, -42.0)
COORD_CENTER = (-41.0, 0.0)
COORD_SCALE = 50.0

def normalize_mag(val: float, base: float) -> float:
    return (val - base) / 10.0

def normalize_coord(x: float, y: float) -> Tuple[float, float]:
    x_norm = (x - COORD_CENTER[0]) / COORD_SCALE
    y_norm = (y - COORD_CENTER[1]) / COORD_SCALE
    return (x_norm, y_norm)

def wavelet_denoise(signal: List[float], wavelet='db4', level=3) -> List[float]:
    """Wavelet denoising"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet).tolist()

def read_nodes(path: Path) -> Dict[int, Tuple[float, float]]:
    """노드 위치 읽기"""
    positions = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row["id"])
            x = float(row["x_m"])
            y = float(row["y_m"])
            positions[node_id] = (x, y)
    return positions

def build_graph(positions: Dict[int, Tuple[float, float]]) -> Dict[int, List[Tuple[int, float]]]:
    """그래프 구축"""
    graph = {node: [] for node in positions}
    nodes = sorted(positions.keys())

    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            xa, ya = positions[a]
            xb, yb = positions[b]
            dist = math.hypot(xb - xa, yb - ya)
            if dist <= 20.0:
                graph[a].append((b, dist))
                graph[b].append((a, dist))

    return graph

def process_csv_sliding(
    file_path: Path,
    positions: Dict[int, Tuple[float, float]],
    feature_mode: str = "mag3",
    window_size: int = 250,
    stride: int = 50,
) -> List[Dict]:
    """CSV를 sliding window로 처리

    Returns:
        List of samples, each: {"features": [250, n_features], "target": [x, y]}
    """
    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < window_size:
        return []

    # 센서 데이터 추출
    try:
        magx = [float(row["MagX"]) for row in rows]
        magy = [float(row["MagY"]) for row in rows]
        magz = [float(row["MagZ"]) for row in rows]
    except (KeyError, ValueError):
        return []

    # Wavelet denoising
    clean_magx = wavelet_denoise(magx)
    clean_magy = wavelet_denoise(magy)
    clean_magz = wavelet_denoise(magz)

    # 경로 정보로 위치 얻기
    parts = file_path.stem.split("_")
    if len(parts) < 2:
        return []

    try:
        start_node = int(parts[0])
        end_node = int(parts[1])
    except ValueError:
        return []

    if start_node not in positions or end_node not in positions:
        return []

    start_pos = positions[start_node]
    end_pos = positions[end_node]

    # 선형 보간으로 각 타임스텝 위치 계산
    num_steps = len(rows)
    positions_list = []
    for i in range(num_steps):
        t = i / (num_steps - 1) if num_steps > 1 else 0.5
        x = start_pos[0] + t * (end_pos[0] - start_pos[0])
        y = start_pos[1] + t * (end_pos[1] - start_pos[1])
        positions_list.append((x, y))

    # Feature 모드에 따라 특징 생성
    features_list = []
    for i in range(num_steps):
        magx_norm = normalize_mag(clean_magx[i], BASE_MAG[0])
        magy_norm = normalize_mag(clean_magy[i], BASE_MAG[1])
        magz_norm = normalize_mag(clean_magz[i], BASE_MAG[2])

        if feature_mode == "mag3":
            feat = [magx_norm, magy_norm, magz_norm]
        elif feature_mode == "mag4":
            mag_magnitude = math.sqrt(clean_magx[i]**2 + clean_magy[i]**2 + clean_magz[i]**2)
            mag_magnitude_norm = (mag_magnitude - 50.0) / 10.0
            feat = [magx_norm, magy_norm, magz_norm, mag_magnitude_norm]
        elif feature_mode == "full":
            pitch = float(rows[i]["Pitch"])
            roll = float(rows[i]["Roll"])
            yaw = float(rows[i]["Yaw"])
            pitch_norm = pitch / 180.0
            roll_norm = roll / 180.0
            yaw_norm = yaw / 180.0
            feat = [magx_norm, magy_norm, magz_norm, pitch_norm, roll_norm, yaw_norm]
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        features_list.append(feat)

    # Sliding window 생성
    samples = []
    for i in range(0, num_steps - window_size + 1, stride):
        window_features = features_list[i:i + window_size]  # [250, n_features]

        # 마지막 타임스텝 위치가 label
        last_idx = i + window_size - 1
        last_pos = positions_list[last_idx]
        target = normalize_coord(last_pos[0], last_pos[1])  # (x_norm, y_norm)

        sample = {
            "features": window_features,
            "target": list(target)
        }
        samples.append(sample)

    return samples

def preprocess_sliding(
    raw_dir: Path,
    nodes_path: Path,
    output_dir: Path,
    feature_mode: str = "mag3",
    window_size: int = 250,
    stride: int = 50,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """Sliding window 방식 전처리"""
    print("=" * 80)
    print("🔄 Sliding Window 전처리 시작")
    print("=" * 80)
    print(f"  Feature mode: {feature_mode}")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride}")
    print()

    # 노드 및 그래프
    positions = read_nodes(nodes_path)
    graph = build_graph(positions)

    # 모든 CSV 파일 처리
    all_samples = []
    csv_files = list(raw_dir.glob("*.csv"))

    print(f"📂 총 {len(csv_files)}개 파일 처리 중...")

    for csv_file in csv_files:
        samples = process_csv_sliding(
            csv_file, positions, feature_mode, window_size, stride
        )
        all_samples.extend(samples)

    print(f"✅ 총 {len(all_samples)}개 샘플 생성")
    print()

    # Train/Val/Test 분할
    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    print(f"📊 데이터 분할:")
    print(f"  Train: {len(train_samples)}개 샘플")
    print(f"  Val:   {len(val_samples)}개 샘플")
    print(f"  Test:  {len(test_samples)}개 샘플")
    print()

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"💾 {output_path} 저장 완료")

    # 메타데이터 저장
    n_features = len(train_samples[0]["features"][0])
    meta = {
        "feature_mode": feature_mode,
        "n_features": n_features,
        "window_size": window_size,
        "stride": stride,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
    }

    meta_path = output_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"💾 {meta_path} 저장 완료")

    print()
    print("=" * 80)
    print("✅ 전처리 완료!")
    print("=" * 80)

    return meta

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", help="원본 CSV 디렉토리")
    parser.add_argument("--nodes", default="data/nodes_final.csv")
    parser.add_argument("--output", default="data/sliding", help="출력 디렉토리")
    parser.add_argument("--feature-mode", default="mag3", choices=["mag3", "mag4", "full"])
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    args = parser.parse_args()

    preprocess_sliding(
        raw_dir=Path(args.raw_dir),
        nodes_path=Path(args.nodes),
        output_dir=Path(args.output),
        feature_mode=args.feature_mode,
        window_size=args.window_size,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
