#!/usr/bin/env python3
"""
더 자세한 걸음 수 분석
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("👣 상세 걸음 수 분석")
print("=" * 70)

data_dir = Path(__file__).resolve().parent.parent / 'dt' / 'processed_data_dt'
traj_train = np.load(data_dir / 'trajectories_train.npy', allow_pickle=True)

print(f"\n전체 데이터: {traj_train.shape}")

# 전체 데이터에서 이동 거리 통계
all_total_distances = []
all_start_to_end = []

for i in range(len(traj_train)):
    traj = traj_train[i]  # (100, 2)

    # 누적 이동 거리
    distances = []
    for t in range(1, len(traj)):
        dist = np.linalg.norm(traj[t] - traj[t-1])
        distances.append(dist)

    total_dist = np.sum(distances)
    all_total_distances.append(total_dist)

    # 시작->끝 직선 거리
    start_to_end = np.linalg.norm(traj[-1] - traj[0])
    all_start_to_end.append(start_to_end)

all_total_distances = np.array(all_total_distances)
all_start_to_end = np.array(all_start_to_end)

print(f"\n📊 누적 이동 거리 통계:")
print(f"  평균: {np.mean(all_total_distances):.4f}m")
print(f"  중간값: {np.median(all_total_distances):.4f}m")
print(f"  최소: {np.min(all_total_distances):.4f}m")
print(f"  최대: {np.max(all_total_distances):.4f}m")
print(f"  표준편차: {np.std(all_total_distances):.4f}m")

print(f"\n📊 시작→끝 직선 거리 통계:")
print(f"  평균: {np.mean(all_start_to_end):.4f}m")
print(f"  중간값: {np.median(all_start_to_end):.4f}m")
print(f"  최소: {np.min(all_start_to_end):.4f}m")
print(f"  최대: {np.max(all_start_to_end):.4f}m")

# 이동이 큰 샘플들 분석
print(f"\n🔍 이동 거리가 큰 샘플들 (상위 10개):")
top_indices = np.argsort(all_total_distances)[-10:][::-1]

for rank, idx in enumerate(top_indices, 1):
    total_dist = all_total_distances[idx]
    straight_dist = all_start_to_end[idx]
    traj = traj_train[idx]

    # 추정 걸음 수 (보폭 0.7m)
    estimated_steps = total_dist / 0.7 if total_dist > 0 else 0

    print(f"\n  {rank}. 샘플 #{idx}:")
    print(f"     누적 이동: {total_dist:.4f}m")
    print(f"     직선 거리: {straight_dist:.4f}m")
    print(f"     추정 걸음: {estimated_steps:.1f}걸음")
    print(f"     시작 위치: ({traj[0, 0]:.4f}, {traj[0, 1]:.4f})")
    print(f"     끝 위치: ({traj[-1, 0]:.4f}, {traj[-1, 1]:.4f})")

# 히스토그램
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].hist(all_total_distances, bins=100, edgecolor='black')
axes[0].set_xlabel('누적 이동 거리 (m)')
axes[0].set_ylabel('빈도')
axes[0].set_title('누적 이동 거리 분포')
axes[0].axvline(np.mean(all_total_distances), color='r', linestyle='--', label=f'평균: {np.mean(all_total_distances):.4f}m')
axes[0].legend()

axes[1].hist(all_start_to_end, bins=100, edgecolor='black')
axes[1].set_xlabel('시작→끝 직선 거리 (m)')
axes[1].set_ylabel('빈도')
axes[1].set_title('시작→끝 직선 거리 분포')
axes[1].axvline(np.mean(all_start_to_end), color='r', linestyle='--', label=f'평균: {np.mean(all_start_to_end):.4f}m')
axes[1].legend()

plt.tight_layout()
plt.savefig('flow_matching/distance_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n📊 히스토그램 저장: flow_matching/distance_distribution.png")

# 걸음 수 추정
avg_total_dist = np.mean(all_total_distances)
avg_straight_dist = np.mean(all_start_to_end)

# 보폭 0.7m 기준
estimated_steps_total = avg_total_dist / 0.7
estimated_steps_straight = avg_straight_dist / 0.7

print("\n" + "=" * 70)
print("📋 최종 추정 (보폭 0.7m 기준):")
print("=" * 70)
print(f"  평균 누적 이동 거리: {avg_total_dist:.4f}m")
print(f"  평균 직선 거리: {avg_straight_dist:.4f}m")
print(f"  추정 걸음 수: {estimated_steps_total:.1f}걸음 (누적 이동 기준)")
print(f"  추정 걸음 수: {estimated_steps_straight:.1f}걸음 (직선 거리 기준)")
print(f"\n💡 100 timesteps ≈ {estimated_steps_total:.1f}걸음")
print(f"   샘플링 레이트: 약 {100/estimated_steps_total:.1f} samples/걸음" if estimated_steps_total > 0 else "")

# 실제 적용 시나리오
print(f"""
🎯 실제 적용 시나리오:

  1️⃣ 최소 필요 걸음 수:
     - 10 timesteps: {10/100*estimated_steps_total:.1f}걸음 (너무 짧음 ⚠️)
     - 30 timesteps: {30/100*estimated_steps_total:.1f}걸음 (기본 예측)
     - 50 timesteps: {50/100*estimated_steps_total:.1f}걸음 (안정적 ✅)
     - 100 timesteps: {estimated_steps_total:.1f}걸음 (최적 ✅)

  2️⃣ 실시간 위치 추정:
     - 사용자가 걸으면서 센서 데이터 수집
     - 50-100 timesteps 축적 후 예측 시작
     - Top-k sampling으로 안정적인 위치 선택
""")
