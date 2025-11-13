#!/usr/bin/env python3
"""
데이터 구성 및 걸음 수 분석
"""
import numpy as np
from pathlib import Path

print("=" * 70)
print("📊 데이터 구성 설명")
print("=" * 70)

print("""
🔍 Raw 데이터 (law_data/*.csv):
  ├─ Timestamp
  ├─ AccX, AccY, AccZ (가속도계)
  ├─ GyroX, GyroY, GyroZ (자이로스코프)
  ├─ MagX, MagY, MagZ (지자기 3축) ✅
  ├─ Pitch, Roll, Yaw (자세 각도 3개) ✅
  ├─ Highlighted (발 접촉 센서 - 왼발) 👣
  └─ RightAngle (발 접촉 센서 - 오른발) 👣

📦 전처리 후 (dt/processed_data_dt/*.npy):
  센서 데이터 (6차원):
    [0:3] = MagX, MagY, MagZ (지자기 3축)
    [3:6] = Pitch, Roll, Yaw (자세 각도)

  💡 발 센서는 어디로?
    → Highlighted=True인 시점 = 마커(발 디딤) 위치
    → 마커 사이의 위치를 선형 보간해서 전체 궤적 생성
    → 발 센서 자체는 학습 데이터에 포함 안 됨!
""")

print("\n" + "=" * 70)
print("👣 걸음 수 분석")
print("=" * 70)

data_dir = Path(__file__).resolve().parent.parent / 'dt' / 'processed_data_dt'
states_train = np.load(data_dir / 'states_train.npy', allow_pickle=True)
traj_train = np.load(data_dir / 'trajectories_train.npy', allow_pickle=True)

print(f"\n데이터 Shape:")
print(f"  센서: {states_train.shape}")
print(f"  궤적: {traj_train.shape}")

# 궤적 분석으로 걸음 수 추정
print(f"\n[분석] 샘플 10개의 이동 패턴:")

step_estimates = []
for i in range(10):
    traj = traj_train[i]  # (100, 2)

    # 각 timestep 사이의 이동 거리
    distances = []
    for t in range(1, len(traj)):
        dist = np.linalg.norm(traj[t] - traj[t-1])
        distances.append(dist)

    distances = np.array(distances)
    total_dist = np.sum(distances)

    # 실제로 움직인 timestep 개수
    moving_timesteps = np.sum(distances > 0.001)  # 1mm 이상 움직임

    # 평균 보폭 0.7m로 걸음 수 추정
    estimated_steps = total_dist / 0.7 if total_dist > 0 else 0
    step_estimates.append(estimated_steps)

    print(f"\n  샘플 {i}:")
    print(f"    총 이동 거리: {total_dist:.4f}m")
    print(f"    움직인 timesteps: {moving_timesteps}/100")
    print(f"    추정 걸음 수: {estimated_steps:.1f}걸음")
    print(f"    samples/걸음: {100/estimated_steps:.1f}" if estimated_steps > 0 else "    samples/걸음: N/A")

avg_steps = np.mean(step_estimates)
print("\n" + "=" * 70)
print(f"📊 전체 평균 추정:")
print(f"  100 timesteps ≈ {avg_steps:.1f}걸음")
print(f"  샘플링 레이트: 약 {100/avg_steps:.1f} samples/걸음" if avg_steps > 0 else "")
print("=" * 70)

print(f"""
🎯 최소 걸음 수별 예측 가능성:

  📍 10 timesteps (≈ {10/avg_steps:.1f}걸음)
    → ⚠️ 너무 짧음, 예측 어려움

  📍 30 timesteps (≈ {30/avg_steps:.1f}걸음)
    → ⚠️ 기본 예측 가능하지만 부정확할 수 있음

  📍 50 timesteps (≈ {50/avg_steps:.1f}걸음)
    → ✅ 안정적인 예측 가능

  📍 100 timesteps (≈ {100/avg_steps:.1f}걸음)
    → ✅ 최적 예측 (학습 데이터 기준)

💡 실제 적용 시:
  - 실시간 예측: 최소 30-50 timesteps 축적 후 예측
  - 높은 정확도: 100 timesteps (현재 학습 설정)
  - Flow Matching: 더 짧은 시퀀스에서도 동작 가능 (실험 필요)
""")
