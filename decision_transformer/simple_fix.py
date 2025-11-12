#!/usr/bin/env python3
"""
Decision Transformer for Indoor Localization - 좌표 정규화 추가

문제: 좌표가 -85 ~ 9 범위라 Loss가 너무 높음
해결: 좌표를 [-1, 1]로 정규화
"""
import numpy as np
from pathlib import Path

# 데이터 로드
script_dir = Path(__file__).parent.parent
data_dir = script_dir / 'v3' / 'processed_data_v3'

train_positions = np.load(data_dir / 'coords_train.npy')
val_positions = np.load(data_dir / 'coords_val.npy')
test_positions = np.load(data_dir / 'coords_test.npy')

print("원본 좌표 범위:")
print(f"  Min: {train_positions.min(axis=0)}")
print(f"  Max: {train_positions.max(axis=0)}")

# 정규화
coords_min = train_positions.min(axis=0, keepdims=True)
coords_max = train_positions.max(axis=0, keepdims=True)
coords_range = coords_max - coords_min

train_norm = (train_positions - coords_min) / coords_range * 2 - 1
val_norm = (val_positions - coords_min) / coords_range * 2 - 1
test_norm = (test_positions - coords_min) / coords_range * 2 - 1

print("\n정규화 후 좌표 범위:")
print(f"  Min: {train_norm.min(axis=0)}")
print(f"  Max: {train_norm.max(axis=0)}")

# 저장
np.save(data_dir / 'coords_train_norm.npy', train_norm)
np.save(data_dir / 'coords_val_norm.npy', val_norm)
np.save(data_dir / 'coords_test_norm.npy', test_norm)

# 정규화 파라미터 저장 (나중에 역변환용)
np.save(data_dir / 'coords_norm_params.npy', {
    'min': coords_min,
    'range': coords_range
})

print("\n정규화된 좌표 저장 완료!")
print("  coords_train_norm.npy")
print("  coords_val_norm.npy")
print("  coords_test_norm.npy")
print("  coords_norm_params.npy")
