#!/usr/bin/env python3
"""
Synthetic dataset generator for Flow Matching.

요구사항:
  - data/processed_data 기반 (실제 경로만 사용)
  - 경로별 데이터 불균형 해소 (모든 경로 동일 샘플 수)
  - 기존 대비 최소 30배 이상의 샘플 생성
  - 최종 출력: flow_matching/processed_data_flow_matching_synth/*

생성 전략:
  1) 각 CSV에서 윈도우(250샘플)를 stride 25로 추출
  2) 경로별 기본 윈도우를 충분히 확보
  3) 경로마다 목표 샘플 수 (약 2,640개) 만큼 랜덤 윈도우를 뽑아
     - 시간 워프(속도/시작점 변형)
     - 센서 drift / smooth noise
     - 채널별 스케일 변화
     등을 적용해 합성
  4) 모든 경로 동일 샘플 개수 → 데이터 균형 보장
  5) 센서/좌표 정규화 후 train/val/test로 분할
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
WINDOW_SIZE = 250
BASE_STRIDE = 25
SENSOR_COLS = ['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']
POSITION_COLS = ['x', 'y']
GRID_SIZE_M = 0.45
ORIGINAL_WINDOW_COUNT = 6057     # 기존 전처리 결과 (flow_matching에서 사용)
TARGET_MULTIPLIER = 30           # 최소 30배 이상
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15                 # TEST = 나머지
RNG = np.random.default_rng(2025)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'processed_data'
NODES_CSV = PROJECT_ROOT / 'data' / 'nodes_final.csv'
OUTPUT_DIR = Path(__file__).parent / 'processed_data_flow_matching_synth'
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Helpers
# ============================================================================
@dataclass
class Window:
    sensors: np.ndarray  # (250, 6)
    trajectory: np.ndarray  # (250, 2)


def parse_route_key(csv_path: Path) -> str:
    """csv 파일명에서 start_end 추출"""
    name = csv_path.stem
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename: {csv_path.name}")
    return f"{parts[0]}_{parts[1]}"


def extract_windows(df: pd.DataFrame, stride: int) -> list[Window]:
    """하나의 CSV에서 윈도우 추출"""
    if len(df) < WINDOW_SIZE:
        return []

    sensors = df[SENSOR_COLS].values.astype(np.float32)
    positions = df[POSITION_COLS].values.astype(np.float32)

    windows: list[Window] = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, stride):
        end = start + WINDOW_SIZE
        seq = sensors[start:end]
        traj = positions[start:end]

        if np.isnan(seq).any() or np.isnan(traj).any():
            continue

        windows.append(Window(seq.copy(), traj.copy()))

    return windows


def random_time_warp(window: Window) -> Window:
    """속도/시작점 변형 (nearest neighbor resampling)"""
    seq = window.sensors
    traj = window.trajectory
    T = seq.shape[0]
    idx = np.arange(T)

    scale = 1.0 + RNG.uniform(-0.15, 0.15)
    shift = RNG.uniform(-12, 12)

    warped_idx = np.clip(np.round(idx * scale + shift).astype(int), 0, T - 1)

    return Window(seq[warped_idx], traj[warped_idx])


def apply_sensor_noise(seq: np.ndarray) -> np.ndarray:
    """Drift + smooth noise + channel scaling"""
    seq = seq.copy()

    # Drift
    seq[:, 0:3] += RNG.normal(0.0, 1.2, size=3)  # Mag drift
    seq[:, 3:6] += RNG.normal(0.0, 2.5, size=3)  # Orientation drift

    # Smooth noise
    noise = RNG.normal(0.0, 0.6, size=seq.shape).astype(np.float32)
    for c in range(seq.shape[1]):
        sigma = RNG.uniform(1.5, 4.0)
        noise[:, c] = gaussian_filter1d(noise[:, c], sigma=sigma, mode='reflect')
    seq += noise

    # Channel scaling (slight calibration errors)
    mag_scale = 1.0 + RNG.uniform(-0.08, 0.08, size=3)
    orient_scale = 1.0 + RNG.uniform(-0.04, 0.04, size=3)
    seq[:, 0:3] *= mag_scale
    seq[:, 3:6] *= orient_scale

    return seq


def blend_windows(win_a: Window, win_b: Window) -> Window:
    """같은 경로 내 다른 샘플과 혼합하여 새로운 시퀀스 생성"""
    alpha = RNG.uniform(0.3, 0.7)
    sensors = alpha * win_a.sensors + (1 - alpha) * win_b.sensors
    trajectory = win_a.trajectory  # 위치는 기준 윈도우 유지
    return Window(sensors.astype(np.float32), trajectory.copy())


def build_balanced_dataset(route_windows: dict[str, list[Window]],
                           target_per_route: int):
    """경로별 동일 개수의 합성 샘플 생성"""
    route_keys = sorted(route_windows.keys())
    total_samples = target_per_route * len(route_keys)

    states = np.empty((total_samples, WINDOW_SIZE, len(SENSOR_COLS)), dtype=np.float32)
    trajectories = np.empty((total_samples, WINDOW_SIZE, 2), dtype=np.float32)
    coords = np.empty((total_samples, 2), dtype=np.float32)

    route_to_indices: dict[str, list[int]] = {}
    write_idx = 0

    for route in tqdm(route_keys, desc="Generating synthetic samples"):
        windows = route_windows[route]
        if not windows:
            raise RuntimeError(f"No base windows for route {route}")

        indices = []
        for _ in range(target_per_route):
            base = RNG.choice(windows)
            if RNG.random() < 0.6 and len(windows) > 1:
                other = base
                while other is base:
                    other = RNG.choice(windows)
                base = blend_windows(base, other)

            warped = random_time_warp(base) if RNG.random() < 0.7 else base
            augmented_seq = apply_sensor_noise(warped.sensors)

            states[write_idx] = augmented_seq
            trajectories[write_idx] = warped.trajectory
            coords[write_idx] = warped.trajectory[-1]

            indices.append(write_idx)
            write_idx += 1

        route_to_indices[route] = indices

    return states, trajectories, coords, route_to_indices


def normalize_sensors(states: np.ndarray):
    mean = states.mean(axis=(0, 1))
    std = states.std(axis=(0, 1))
    std[std < 1e-6] = 1.0
    states_norm = (states - mean) / std
    return states_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def normalize_positions(coords: np.ndarray, traj: np.ndarray,
                        bounds: dict[str, float]):
    x_min, x_max = bounds['x_min'], bounds['x_max']
    y_min, y_max = bounds['y_min'], bounds['y_max']
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    coords_norm = np.empty_like(coords)
    traj_norm = np.empty_like(traj)

    coords_norm[:, 0] = (coords[:, 0] - x_min) / x_range * 2 - 1
    coords_norm[:, 1] = (coords[:, 1] - y_min) / y_range * 2 - 1

    traj_norm[:, :, 0] = (traj[:, :, 0] - x_min) / x_range * 2 - 1
    traj_norm[:, :, 1] = (traj[:, :, 1] - y_min) / y_range * 2 - 1

    return coords_norm.astype(np.float32), traj_norm.astype(np.float32)


def coord_to_grid(coords: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    x_min, x_max = bounds['x_min'], bounds['x_max']
    y_min, y_max = bounds['y_min'], bounds['y_max']

    num_x = int(np.ceil((x_max - x_min) / GRID_SIZE_M)) + 1
    num_y = int(np.ceil((y_max - y_min) / GRID_SIZE_M)) + 1

    rel_x = np.round((coords[:, 0] - x_min) / GRID_SIZE_M).astype(int)
    rel_y = np.round((coords[:, 1] - y_min) / GRID_SIZE_M).astype(int)

    rel_x = np.clip(rel_x, 0, num_x - 1)
    rel_y = np.clip(rel_y, 0, num_y - 1)

    return rel_y * num_x + rel_x


def split_by_route(route_indices: dict[str, list[int]]):
    routes = list(route_indices.keys())
    RNG.shuffle(routes)

    n_routes = len(routes)
    n_train = int(n_routes * TRAIN_RATIO)
    n_val = int(n_routes * VAL_RATIO)

    train_routes = routes[:n_train]
    val_routes = routes[n_train:n_train + n_val]
    test_routes = routes[n_train + n_val:]

    def gather(keys):
        idx = []
        for key in keys:
            idx.extend(route_indices[key])
        return np.array(idx, dtype=np.int64)

    return (
        gather(train_routes),
        gather(val_routes),
        gather(test_routes),
        train_routes,
        val_routes,
        test_routes,
    )


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("🧪 Synthetic Flow Matching Dataset Generator")
    print("=" * 70)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    base_files = sorted(DATA_DIR.glob('*.csv'))
    print(f"  CSV files: {len(base_files)}")

    # 1) Load base windows
    print("\n[1/6] 원본 윈도우 추출...")
    route_windows: dict[str, list[Window]] = defaultdict(list)
    total_base = 0
    for csv_path in tqdm(base_files, desc="Scanning CSV"):
        df = pd.read_csv(csv_path)
        windows = extract_windows(df, stride=BASE_STRIDE)
        if not windows:
            continue
        route_key = parse_route_key(csv_path)
        route_windows[route_key].extend(windows)
        total_base += len(windows)

    n_routes = len(route_windows)
    if n_routes == 0:
        raise RuntimeError("No valid routes found in processed_data")

    print(f"  경로 수: {n_routes}")
    print(f"  기본 윈도우 수: {total_base:,}")

    # 2) Determine target per route
    target_per_route = math.ceil((ORIGINAL_WINDOW_COUNT * TARGET_MULTIPLIER) / n_routes)
    target_total = target_per_route * n_routes
    print(f"\n[2/6] 목표 샘플 수 설정...")
    print(f"  기존 윈도우 수: {ORIGINAL_WINDOW_COUNT}")
    print(f"  목표 배수: {TARGET_MULTIPLIER}x")
    print(f"  경로당 목표 샘플: {target_per_route}")
    print(f"  총 합성 샘플: {target_total:,}")

    # 3) Generate balanced synthetic dataset
    print("\n[3/6] 합성 샘플 생성...")
    states, trajectories, coords, route_indices = build_balanced_dataset(
        route_windows, target_per_route
    )

    # 4) Normalization + labels
    print("\n[4/6] 정규화 및 라벨 계산...")
    nodes_df = pd.read_csv(NODES_CSV)
    x_min, x_max = nodes_df['x_m'].min(), nodes_df['x_m'].max()
    y_min, y_max = nodes_df['y_m'].min(), nodes_df['y_m'].max()
    bounds = {'x_min': float(x_min), 'x_max': float(x_max),
              'y_min': float(y_min), 'y_max': float(y_max)}

    states_norm, sensor_mean, sensor_std = normalize_sensors(states)
    coords_norm, traj_norm = normalize_positions(coords, trajectories, bounds)
    labels = coord_to_grid(coords, bounds)

    # 5) Split
    print("\n[5/6] Train/Val/Test 분할 (경로 기준)...")
    idx_train, idx_val, idx_test, train_routes, val_routes, test_routes = split_by_route(route_indices)

    def subset(arr, indices):
        return arr[indices]

    states_train = subset(states_norm, idx_train)
    states_val = subset(states_norm, idx_val)
    states_test = subset(states_norm, idx_test)

    coords_train = subset(coords_norm, idx_train)
    coords_val = subset(coords_norm, idx_val)
    coords_test = subset(coords_norm, idx_test)

    traj_train = subset(traj_norm, idx_train)
    traj_val = subset(traj_norm, idx_val)
    traj_test = subset(traj_norm, idx_test)

    labels_train = subset(labels, idx_train)
    labels_val = subset(labels, idx_val)
    labels_test = subset(labels, idx_test)

    print(f"  Train: {len(states_train):,} | Val: {len(states_val):,} | Test: {len(states_test):,}")

    # 6) Save
    print("\n[6/6] 저장...")
    np.save(OUTPUT_DIR / 'states_train.npy', states_train)
    np.save(OUTPUT_DIR / 'coords_train.npy', coords_train)
    np.save(OUTPUT_DIR / 'trajectories_train.npy', traj_train)
    np.save(OUTPUT_DIR / 'labels_train.npy', labels_train)

    np.save(OUTPUT_DIR / 'states_val.npy', states_val)
    np.save(OUTPUT_DIR / 'coords_val.npy', coords_val)
    np.save(OUTPUT_DIR / 'trajectories_val.npy', traj_val)
    np.save(OUTPUT_DIR / 'labels_val.npy', labels_val)

    np.save(OUTPUT_DIR / 'states_test.npy', states_test)
    np.save(OUTPUT_DIR / 'coords_test.npy', coords_test)
    np.save(OUTPUT_DIR / 'trajectories_test.npy', traj_test)
    np.save(OUTPUT_DIR / 'labels_test.npy', labels_test)

    metadata = {
        'window_size': WINDOW_SIZE,
        'stride': BASE_STRIDE,
        'grid_size': GRID_SIZE_M,
        'sensor_cols': SENSOR_COLS,
        'position_cols': POSITION_COLS,
        'original_windows': ORIGINAL_WINDOW_COUNT,
        'target_multiplier': TARGET_MULTIPLIER,
        'target_per_route': target_per_route,
        'num_routes': n_routes,
        'sensor_normalization': {
            'mean': sensor_mean.tolist(),
            'std': sensor_std.tolist(),
        },
        'position_bounds': bounds,
        'splits': {
            'train_routes': train_routes,
            'val_routes': val_routes,
            'test_routes': test_routes,
        },
    }

    import pickle
    with open(OUTPUT_DIR / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print("\n✅ 합성 데이터셋 생성 완료!")
    print(f"  저장 위치: {OUTPUT_DIR}")
    print(f"  총 샘플: {target_total:,} (경로당 {target_per_route})")


if __name__ == '__main__':
    main()
