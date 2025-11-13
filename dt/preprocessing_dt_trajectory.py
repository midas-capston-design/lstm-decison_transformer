#!/usr/bin/env python3
"""
Decision Transformer용 데이터 전처리
각 timestep마다의 위치를 저장
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

print("="*70)
print("🔧 Decision Transformer용 Trajectory 데이터 생성")
print("="*70)

# 기존 v3 데이터 로드
data_dir = Path('v3/processed_data_v3')

X_train = np.load(data_dir / 'X_train.npy')
X_val = np.load(data_dir / 'X_val.npy')
X_test = np.load(data_dir / 'X_test.npy')

coords_train = np.load(data_dir / 'coords_train.npy')
coords_val = np.load(data_dir / 'coords_val.npy')
coords_test = np.load(data_dir / 'coords_test.npy')

print(f"\n기존 데이터:")
print(f"  X_train: {X_train.shape}")
print(f"  coords_train: {coords_train.shape}")

# ============================================================================
# 문제: coords는 마지막 timestep만!
# 해결: raw 데이터에서 다시 추출해야 함
# ============================================================================

print("\n⚠️  현재 coords는 각 샘플의 마지막 위치만 저장됨")
print("⚠️  Decision Transformer는 각 timestep의 위치가 필요")
print("\n옵션:")
print("  1. Raw 데이터에서 전처리 다시 실행 (느림)")
print("  2. 마지막 위치로 전체 trajectory를 근사 (빠름, 부정확)")
print("  3. 현재 데이터로 억지로 학습 (마지막 위치만 예측)")

print("\n현재는 Option 3로 진행 중입니다.")
print("각 timestep의 실제 위치가 필요하면 preprocessing_v3.py를 수정해야 합니다.")
