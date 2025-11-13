#!/usr/bin/env python3
"""
걸음 수 분석 스크립트
- 100 timesteps = 몇 걸음?
- 최소 몇 걸음부터 위치 추정 가능?
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 70)
print("👣 걸음 수 분석")
print("=" * 70)

data_dir = Path(__file__).resolve().parent.parent / 'dt' / 'processed_data_dt'

# 데이터 로드
print("\n[1/4] 데이터 로드...")
states_train = np.load(data_dir / 'states_train.npy', allow_pickle=True)
traj_train = np.load(data_dir / 'trajectories_train.npy', allow_pickle=True)

# 궤적 변화 분석
print("\n[2/4] 궤적 변화 분석...")
sample_indices = [0, 100, 1000, 5000]

for idx in sample_indices:
    traj = traj_train[idx]  # (100, 2)

    # 각 timestep의 이동 거리 계산
    distances = []
    for t in range(1, len(traj)):
        dist = np.linalg.norm(traj[t] - traj[t-1])
        distances.append(dist)

    distances = np.array(distances)
    total_distance = np.sum(distances)

    # 걸음 감지: 이동 거리가 임계값 이상인 경우
    threshold = np.mean(distances) + np.std(distances)
    steps = np.sum(distances > threshold)

    print(f"\n  Sample {idx}:")
    print(f"    시작 위치: ({traj[0, 0]:.4f}, {traj[0, 1]:.4f})")
    print(f"    끝 위치: ({traj[-1, 0]:.4f}, {traj[-1, 1]:.4f})")
    print(f"    총 이동 거리: {total_distance:.4f}m")
    print(f"    평균 timestep 이동: {np.mean(distances):.6f}m")
    print(f"    추정 걸음 수: {steps}개")
    print(f"    timesteps/걸음: {100/max(steps, 1):.1f}")

# 전체 데이터 분석
print("\n[3/4] 전체 데이터 통계...")
all_total_distances = []
all_movements = []

for i in range(min(10000, len(traj_train))):
    traj = traj_train[i]
    distances = []
    for t in range(1, len(traj)):
        dist = np.linalg.norm(traj[t] - traj[t-1])
        distances.append(dist)
        all_movements.append(dist)

    total_distance = np.sum(distances)
    all_total_distances.append(total_distance)

all_total_distances = np.array(all_total_distances)
all_movements = np.array(all_movements)

print(f"\n  총 이동 거리 통계 (샘플 10,000개):")
print(f"    평균: {np.mean(all_total_distances):.4f}m")
print(f"    중간값: {np.median(all_total_distances):.4f}m")
print(f"    최소: {np.min(all_total_distances):.4f}m")
print(f"    최대: {np.max(all_total_distances):.4f}m")

print(f"\n  timestep당 이동 거리 통계:")
print(f"    평균: {np.mean(all_movements):.6f}m")
print(f"    중간값: {np.median(all_movements):.6f}m")
print(f"    최대: {np.max(all_movements):.6f}m")

# 걸음 수 추정
print("\n[4/4] 걸음 수 추정...")

# 방법 1: 일반적인 보폭 기준 (0.7m)
avg_stride = 0.7  # 일반적인 보폭
estimated_steps_from_distance = np.mean(all_total_distances) / avg_stride

print(f"\n  방법 1 - 보폭 기준 (0.7m/걸음):")
print(f"    평균 이동 거리: {np.mean(all_total_distances):.4f}m")
print(f"    추정 걸음 수: {estimated_steps_from_distance:.1f}걸음")
print(f"    timesteps/걸음: {100/estimated_steps_from_distance:.1f}")

# 방법 2: 움직임이 거의 없는 timestep 제외
moving_threshold = 0.001  # 1mm 이상 움직이면 이동으로 간주
moving_ratio = np.mean(all_movements > moving_threshold)
non_moving_ratio = 1 - moving_ratio

print(f"\n  방법 2 - 움직임 분석:")
print(f"    움직이는 timestep 비율: {moving_ratio*100:.1f}%")
print(f"    정지 timestep 비율: {non_moving_ratio*100:.1f}%")

# 샘플링 레이트 추정
print(f"\n  💡 추정 결과:")
if estimated_steps_from_distance < 10:
    print(f"    100 timesteps ≈ {estimated_steps_from_distance:.0f}걸음")
    print(f"    센서 샘플링: 약 {100/estimated_steps_from_distance:.0f} samples/걸음")
elif estimated_steps_from_distance < 50:
    print(f"    100 timesteps ≈ {estimated_steps_from_distance:.0f}걸음")
    print(f"    센서 샘플링: 약 {100/estimated_steps_from_distance:.1f} samples/걸음")
else:
    print(f"    100 timesteps ≈ {estimated_steps_from_distance:.0f}걸음")
    print(f"    거의 timestep마다 걸음 (고빈도 샘플링)")

print("\n" + "=" * 70)
print("📊 최소 걸음 수 테스트")
print("=" * 70)

# 시퀀스 길이별 테스트
sequence_lengths = [10, 20, 30, 50, 75, 100]
print("\n다양한 시퀀스 길이에서 사용 가능한 정보:")

for seq_len in sequence_lengths:
    estimated_steps = seq_len / (100 / estimated_steps_from_distance)
    estimated_distance = (seq_len / 100) * np.mean(all_total_distances)

    print(f"\n  {seq_len} timesteps:")
    print(f"    추정 걸음 수: {estimated_steps:.1f}걸음")
    print(f"    추정 이동 거리: {estimated_distance:.4f}m")
    print(f"    예측 가능성: {'✅ 충분' if seq_len >= 30 else '⚠️ 부족할 수 있음'}")

print("\n" + "=" * 70)
print("✅ 분석 완료!")
print("=" * 70)
print(f"""
📋 요약:
  🚶 100 timesteps ≈ {estimated_steps_from_distance:.0f}걸음
  📏 평균 총 이동 거리: {np.mean(all_total_distances):.4f}m
  ⏱️  샘플링 레이트: ~{100/estimated_steps_from_distance:.1f} samples/걸음

🎯 권장사항:
  ✅ 30+ timesteps (≈ {30/(100/estimated_steps_from_distance):.1f}걸음) - 기본 예측 가능
  ✅ 50+ timesteps (≈ {50/(100/estimated_steps_from_distance):.1f}걸음) - 안정적 예측
  ✅ 100 timesteps (≈ {estimated_steps_from_distance:.0f}걸음) - 최적 예측
""")
