#!/usr/bin/env python3
"""
데이터 검증 스크립트
입력: 센서 데이터 (지자기값 등)
출력: x,y 위치값
"""
import numpy as np
from pathlib import Path

print("=" * 70)
print("🔍 데이터 검증")
print("=" * 70)

data_dir = Path(__file__).resolve().parent.parent / 'dt' / 'processed_data_dt'

# 데이터 로드
print("\n[1/3] 데이터 로드...")
states_train = np.load(data_dir / 'states_train.npy', allow_pickle=True)
traj_train = np.load(data_dir / 'trajectories_train.npy', allow_pickle=True)

print(f"  states_train shape: {states_train.shape}")
print(f"  trajectories_train shape: {traj_train.shape}")

# 입력 데이터 (센서) 확인
print("\n[2/3] 입력 데이터 (센서) 분석...")
print(f"\n  첫 번째 샘플의 센서 데이터 (처음 5개 timestep):")
print(states_train[0, :5, :])

print(f"\n  센서 데이터 통계 (전체 데이터):")
for i in range(states_train.shape[2]):
    col_data = states_train[:, :, i]
    print(f"    Dim {i}: min={col_data.min():.4f}, max={col_data.max():.4f}, "
          f"mean={col_data.mean():.4f}, std={col_data.std():.4f}")

print(f"\n  💡 센서 데이터 해석:")
print(f"     - Shape: (N, 100, 6)")
print(f"     - 100 timesteps, 6차원 센서 데이터")
print(f"     - 예상: 지자기 3축(x,y,z) + 기타 센서 3개")

# 출력 데이터 (위치) 확인
print("\n[3/3] 출력 데이터 (위치) 분석...")
print(f"\n  첫 10개 샘플의 마지막 위치 (x, y):")
for i in range(10):
    x, y = traj_train[i, -1, :]
    print(f"    Sample {i}: x={x:.4f}, y={y:.4f}")

print(f"\n  위치 데이터 통계:")
x_coords = traj_train[:, -1, 0]
y_coords = traj_train[:, -1, 1]
print(f"    X: min={x_coords.min():.4f}, max={x_coords.max():.4f}, "
      f"mean={x_coords.mean():.4f}, std={x_coords.std():.4f}")
print(f"    Y: min={y_coords.min():.4f}, max={y_coords.max():.4f}, "
      f"mean={y_coords.mean():.4f}, std={y_coords.std():.4f}")

print(f"\n  💡 위치 데이터 해석:")
print(f"     - Shape: (N, 100, 2)")
print(f"     - 마지막 위치 = 타겟 위치")
print(f"     - 2차원 좌표 (x, y)")
print(f"     - 단위: {'미터(m)' if x_coords.max() < 1000 else '알 수 없음'}")

# 전체 궤적 확인
print(f"\n[보너스] 전체 궤적 분석...")
print(f"  첫 번째 샘플의 궤적 (처음 5개 timestep):")
for t in range(5):
    x, y = traj_train[0, t, :]
    print(f"    t={t}: x={x:.4f}, y={y:.4f}")

print("\n" + "=" * 70)
print("✅ 데이터 검증 완료!")
print("=" * 70)
print("""
📋 요약:
  ✅ 입력: (N, 100, 6) - 센서 시퀀스 (지자기 등)
  ✅ 출력: (N, 100, 2) - 위치 궤적 (x, y)
  ✅ 타겟: 마지막 위치 (x, y)

🎯 Flow Matching 모델:
  입력: 센서 데이터 (100, 6) → 출력: 위치 (x, y)
""")
