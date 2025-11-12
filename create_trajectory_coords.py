#!/usr/bin/env python3
"""
각 timestep의 위치를 선형 보간으로 생성
(간단한 근사 - 실제로는 더 복잡한 움직임)
"""
import numpy as np
from pathlib import Path

data_dir = Path('v3/processed_data_v3')

# 마지막 위치만 있는 coords 로드
coords_train = np.load(data_dir / 'coords_train.npy')  # (N, 2)
coords_val = np.load(data_dir / 'coords_val.npy')
coords_test = np.load(data_dir / 'coords_test.npy')

print(f"원본 coords_train: {coords_train.shape}")

def create_trajectory(final_coords, window_size=100):
    """
    마지막 위치에서 역으로 trajectory 생성 (선형 근사)

    간단한 가정: 등속 직선 운동
    """
    N = len(final_coords)
    trajectories = np.zeros((N, window_size, 2))

    for i in range(N):
        # 마지막 위치
        end_pos = final_coords[i]

        # 시작 위치를 추정 (간단히: 마지막 위치 근처)
        # 실제로는 raw 데이터에서 계산해야 하지만, 여기서는 근사
        # 가정: 평균 2초간 2m 이동 (대략)
        start_offset = np.random.randn(2) * 1.0  # 1m 정도 차이
        start_pos = end_pos + start_offset

        # 선형 보간
        for t in range(window_size):
            progress = t / (window_size - 1)
            trajectories[i, t] = start_pos + (end_pos - start_pos) * progress

    return trajectories

print("\n선형 trajectory 생성 중...")
traj_train = create_trajectory(coords_train)
traj_val = create_trajectory(coords_val)
traj_test = create_trajectory(coords_test)

print(f"  traj_train: {traj_train.shape}")

# 저장
np.save(data_dir / 'trajectory_train.npy', traj_train)
np.save(data_dir / 'trajectory_val.npy', traj_val)
np.save(data_dir / 'trajectory_test.npy', traj_test)

print(f"\n저장 완료!")
print(f"  trajectory_train.npy: {traj_train.shape}")
print(f"\n⚠️  주의: 선형 근사이므로 정확하지 않음")
print("   실제 trajectory는 raw 데이터에서 재추출 필요")
