#!/usr/bin/env python3
"""
Flow Matching preprocessing using data/processed_data CSVs.

This script converts the pre-aligned dataset (with x/y coordinates already
assigned) into the numpy tensors consumed by the Flow Matching training code.

Input  : data/processed_data/*.csv (columns: x, y, Mag*, Pitch/Roll/Yaw, etc.)
Output : flow_matching/processed_data_flow_matching/*.npy + metadata.pkl
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
WINDOW_SIZE = 250          # 시퀀스 길이 (샘플당 약 5초)
STRIDE = 50                # 윈도우 이동 간격 (샘플 중첩)
GRID_SIZE_M = 0.45         # 0.45m 간격 (노드 간 거리)
SENSOR_COLS = ['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']
POSITION_COLS = ['x', 'y']
RNG_SEED = 42

print("=" * 70)
print("🔧 Flow Matching 전처리 (data/processed_data → numpy tensors)")
print("=" * 70)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data' / 'processed_data'
OUTPUT_DIR = Path(__file__).parent / 'processed_data_flow_matching'

if not DATA_DIR.exists():
    raise FileNotFoundError(f"데이터 디렉토리 없음: {DATA_DIR}")

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Helpers
# ============================================================================
def parse_filename(path: Path) -> tuple[int, int, str]:
    """파일명 → (start, end, trial)"""
    name = path.stem  # e.g. 1_23_6
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {name}")
    start_node = int(parts[0])
    end_node = int(parts[1])
    trial = parts[2]
    return start_node, end_node, trial


def slide_windows(df: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """윈도우 단위로 (센서, 궤적, 최종좌표) 생성"""
    if len(df) < WINDOW_SIZE:
        return None

    seq_list, coord_list, traj_list = [], [], []

    sensor_values = df[SENSOR_COLS].values.astype(np.float32)
    pos_values = df[POSITION_COLS].values.astype(np.float32)

    for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE

        seq = sensor_values[start:end]
        traj = pos_values[start:end]

        if np.isnan(seq).any() or np.isnan(traj).any():
            continue  # 결측값 샘플은 제거

        seq_list.append(seq)
        traj_list.append(traj)
        coord_list.append(traj[-1])  # 윈도우 마지막 위치

    if not seq_list:
        return None

    return {
        'sequences': np.stack(seq_list),
        'trajectories': np.stack(traj_list),
        'coords': np.stack(coord_list),
    }


def normalize_sequences(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """센서 데이터를 표준화하고 (정규화된 시퀀스, mean, std) 반환"""
    mean = sequences.mean(axis=(0, 1), keepdims=False)
    std = sequences.std(axis=(0, 1), keepdims=False)
    std[std < 1e-6] = 1.0  # 분산 0 방지
    normalized = (sequences - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def normalize_positions(coords: np.ndarray, x_min: float, x_max: float,
                        y_min: float, y_max: float) -> np.ndarray:
    """절대 좌표(x/y)를 -1~1 범위로 정규화"""
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    coords_norm = np.empty_like(coords, dtype=np.float32)
    coords_norm[:, 0] = (coords[:, 0] - x_min) / x_range * 2.0 - 1.0
    coords_norm[:, 1] = (coords[:, 1] - y_min) / y_range * 2.0 - 1.0
    return coords_norm


def normalize_trajectories(traj: np.ndarray, x_min: float, x_max: float,
                           y_min: float, y_max: float) -> np.ndarray:
    """전체 궤적을 -1~1로 정규화"""
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    traj_norm = np.empty_like(traj, dtype=np.float32)
    traj_norm[..., 0] = (traj[..., 0] - x_min) / x_range * 2.0 - 1.0
    traj_norm[..., 1] = (traj[..., 1] - y_min) / y_range * 2.0 - 1.0
    return traj_norm


def coord_to_grid_id(coords: np.ndarray, x_min: float, y_min: float,
                     x_max: float, y_max: float) -> np.ndarray:
    """좌표를 GRID_SIZE_M 단위의 grid id로 변환"""
    num_x_grids = int(np.ceil((x_max - x_min) / GRID_SIZE_M)) + 1
    num_y_grids = int(np.ceil((y_max - y_min) / GRID_SIZE_M)) + 1

    rel_x = np.round((coords[:, 0] - x_min) / GRID_SIZE_M).astype(int)
    rel_y = np.round((coords[:, 1] - y_min) / GRID_SIZE_M).astype(int)

    rel_x = np.clip(rel_x, 0, num_x_grids - 1)
    rel_y = np.clip(rel_y, 0, num_y_grids - 1)

    return (rel_y * num_x_grids + rel_x).astype(np.int32)


# ============================================================================
# 1) 파일별 윈도우 생성
# ============================================================================
print("\n[1/5] CSV 스캔 및 윈도우 생성...")
files = sorted(DATA_DIR.glob('*.csv'))
if not files:
    raise RuntimeError(f"CSV 파일을 찾을 수 없습니다: {DATA_DIR}")

all_sequences = []
all_coords = []
all_trajectories = []
file_to_indices = defaultdict(list)  # 파일 → 전체 인덱스
global_index = 0

for csv_path in tqdm(files, desc="Processing CSV"):
    try:
        start_node, end_node, trial = parse_filename(csv_path)
    except ValueError:
        print(f"  ⚠️  파일명 형식 무시: {csv_path.name}")
        continue

    df = pd.read_csv(csv_path)
    required_cols = set(SENSOR_COLS + POSITION_COLS)
    if not required_cols.issubset(df.columns):
        print(f"  ⚠️  누락 컬럼으로 스킵: {csv_path.name}")
        continue

    window_data = slide_windows(df)
    if window_data is None:
        continue

    seqs = window_data['sequences']
    coords = window_data['coords']
    trajs = window_data['trajectories']

    all_sequences.append(seqs)
    all_coords.append(coords)
    all_trajectories.append(trajs)

    file_key = f"{start_node}_{end_node}_{trial}"
    file_to_indices[file_key].extend(range(global_index, global_index + len(seqs)))
    global_index += len(seqs)

total_samples = global_index
if total_samples == 0:
    raise RuntimeError("생성된 윈도우가 없습니다. 데이터/윈도우 설정을 확인하세요.")

print(f"  총 윈도우 샘플: {total_samples:,}개")

# 스택
all_sequences = np.vstack(all_sequences)
all_coords = np.vstack(all_coords)
all_trajectories = np.vstack(all_trajectories)

# 건물 범위
x_min, x_max = float(all_coords[:, 0].min()), float(all_coords[:, 0].max())
y_min, y_max = float(all_coords[:, 1].min()), float(all_coords[:, 1].max())
print(f"  좌표 범위: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

# ============================================================================
# 2) 정규화
# ============================================================================
print("\n[2/5] 센서/좌표 정규화...")
states_norm, sensor_mean, sensor_std = normalize_sequences(all_sequences)
coords_norm = normalize_positions(all_coords, x_min, x_max, y_min, y_max)
traj_norm = normalize_trajectories(all_trajectories, x_min, x_max, y_min, y_max)
labels = coord_to_grid_id(all_coords, x_min, y_min, x_max, y_max)

# ============================================================================
# 3) Train/Val/Test split (파일 단위)
# ============================================================================
print("\n[3/5] Train/Val/Test 분할 (파일 단위)...")
file_keys = list(file_to_indices.keys())
rng = np.random.default_rng(RNG_SEED)
rng.shuffle(file_keys)

n_files = len(file_keys)
n_train = int(n_files * 0.7)
n_val = int(n_files * 0.15)

train_files = file_keys[:n_train]
val_files = file_keys[n_train:n_train + n_val]
test_files = file_keys[n_train + n_val:]

def gather_indices(keys):
    idx = []
    for key in keys:
        idx.extend(file_to_indices[key])
    return np.array(idx, dtype=np.int32)

train_idx = gather_indices(train_files)
val_idx = gather_indices(val_files)
test_idx = gather_indices(test_files)

print(f"  파일 기준 분할: Train {len(train_files)}, Val {len(val_files)}, Test {len(test_files)}")
print(f"  샘플 수: Train {len(train_idx):,}, Val {len(val_idx):,}, Test {len(test_idx):,}")


def subset(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return arr[indices]


states_train = subset(states_norm, train_idx)
coords_train = subset(coords_norm, train_idx)
traj_train = subset(traj_norm, train_idx)
labels_train = subset(labels, train_idx)

states_val = subset(states_norm, val_idx)
coords_val = subset(coords_norm, val_idx)
traj_val = subset(traj_norm, val_idx)
labels_val = subset(labels, val_idx)

states_test = subset(states_norm, test_idx)
coords_test = subset(coords_norm, test_idx)
traj_test = subset(traj_norm, test_idx)
labels_test = subset(labels, test_idx)

# ============================================================================
# 4) 저장
# ============================================================================
print("\n[4/5] numpy 저장...")
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
    'stride': STRIDE,
    'grid_size': GRID_SIZE_M,
    'sensor_cols': SENSOR_COLS,
    'position_cols': POSITION_COLS,
    'num_samples': {
        'train': int(len(train_idx)),
        'val': int(len(val_idx)),
        'test': int(len(test_idx)),
        'total': int(total_samples),
    },
    'sensor_normalization': {
        'mean': sensor_mean.tolist(),
        'std': sensor_std.tolist(),
    },
    'normalization': {  # 기존 스크립트와 호환
        'sensor_mean': sensor_mean.tolist(),
        'sensor_std': sensor_std.tolist(),
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
    },
    'position_bounds': {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
    },
    'file_splits': {
        'train': train_files,
        'val': val_files,
        'test': test_files,
    },
}

with open(OUTPUT_DIR / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n[5/5] 완료 요약")
print(f"  states_train.npy shape: {states_train.shape}")
print(f"  states_val.npy   shape: {states_val.shape}")
print(f"  states_test.npy  shape: {states_test.shape}")
print(f"  metadata: {OUTPUT_DIR / 'metadata.pkl'}")
print("\n✅ data/processed_data → flow_matching/processed_data_flow_matching 변환 완료!")
