#!/usr/bin/env python3
"""
데이터 품질 분석 - 1.5m @ 90% 달성 가능성 검증

핵심 질문:
1. 같은 위치에서 센서값 variance가 얼마나 되는가?
2. SNR은 얼마나 되는가?
3. 방향성 문제가 얼마나 심각한가?
4. 이 데이터로 1.5m 정확도가 가능한가?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pickle

print("="*70)
print("데이터 품질 분석")
print("="*70)

# 데이터 로드
data_dir = Path('hyena/processed_data_hyena')

if not data_dir.exists():
    print("❌ 전처리 데이터가 없습니다. preprocessing_hyena.py를 먼저 실행하세요.")
    exit(1)

states_train = np.load(data_dir / 'states_train.npy')
positions_train = np.load(data_dir / 'positions_train.npy')

states_test = np.load(data_dir / 'states_test.npy')
positions_test = np.load(data_dir / 'positions_test.npy')

with open(data_dir / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

coords_min = np.array(metadata['normalization']['coords_min'])
coords_max = np.array(metadata['normalization']['coords_max'])

print(f"\n데이터 크기:")
print(f"  Train: {len(states_train):,}개")
print(f"  Test: {len(states_test):,}개")

# Denormalize
def denormalize_coords(coords_norm, coords_min, coords_max):
    coords_range = coords_max - coords_min
    return (coords_norm + 1) / 2 * coords_range + coords_min

positions_train_real = denormalize_coords(positions_train, coords_min, coords_max)
positions_test_real = denormalize_coords(positions_test, coords_min, coords_max)

# ============================================================================
# 1. 같은 위치(그리드)에서 센서값 variance
# ============================================================================
print("\n" + "="*70)
print("1. 위치별 센서값 Variance 분석")
print("="*70)

GRID_SIZE = 0.9  # m

def coord_to_grid(x, y):
    return (int(x / GRID_SIZE), int(y / GRID_SIZE))

# 그리드별로 샘플 그룹화
grid_samples = defaultdict(list)
for i, (x, y) in enumerate(positions_train_real):
    grid_id = coord_to_grid(x, y)
    grid_samples[grid_id].append(i)

# 그리드별 variance 계산
grid_variances = []
for grid_id, indices in grid_samples.items():
    if len(indices) < 2:
        continue

    # 이 그리드의 모든 샘플
    samples = states_train[indices]  # (N, 250, 6)

    # 각 센서별 variance (시간축 평균 후)
    mean_values = samples.mean(axis=1)  # (N, 6) - 시퀀스 평균
    variance = mean_values.var(axis=0)  # (6,) - 샘플간 variance

    grid_variances.append({
        'grid': grid_id,
        'n_samples': len(indices),
        'mag_var': variance[:3].mean(),  # MagX, MagY, MagZ
        'orient_var': variance[3:].mean(),  # Pitch, Roll, Yaw
    })

df_var = pd.DataFrame(grid_variances)

print(f"\n분석한 그리드: {len(df_var)}개")
print(f"\n지자기 Variance:")
print(f"  평균: {df_var['mag_var'].mean():.4f}")
print(f"  중앙값: {df_var['mag_var'].median():.4f}")
print(f"  최대: {df_var['mag_var'].max():.4f}")
print(f"\n방향 Variance:")
print(f"  평균: {df_var['orient_var'].mean():.4f}")
print(f"  중앙값: {df_var['orient_var'].median():.4f}")
print(f"  최대: {df_var['orient_var'].max():.4f}")

# ============================================================================
# 2. 위치 분포 - 커버리지 확인
# ============================================================================
print("\n" + "="*70)
print("2. 위치 분포 분석")
print("="*70)

x_range = coords_max[0] - coords_min[0]
y_range = coords_max[1] - coords_min[1]
area = x_range * y_range

print(f"\n커버리지:")
print(f"  X 범위: {coords_min[0]:.2f} ~ {coords_max[0]:.2f} ({x_range:.2f}m)")
print(f"  Y 범위: {coords_min[1]:.2f} ~ {coords_max[1]:.2f} ({y_range:.2f}m)")
print(f"  총 면적: {area:.2f} m²")
print(f"  샘플 밀도: {len(states_train)/area:.2f} 샘플/m²")

# ============================================================================
# 3. 그리드별 샘플 수 분포
# ============================================================================
print("\n" + "="*70)
print("3. 그리드별 샘플 분포")
print("="*70)

samples_per_grid = [len(indices) for indices in grid_samples.values()]
print(f"\n그리드 수: {len(grid_samples)}")
print(f"그리드당 샘플 수:")
print(f"  평균: {np.mean(samples_per_grid):.1f}")
print(f"  중앙값: {np.median(samples_per_grid):.0f}")
print(f"  최소: {np.min(samples_per_grid)}")
print(f"  최대: {np.max(samples_per_grid)}")

# 샘플이 적은 그리드
few_samples = sum(1 for n in samples_per_grid if n < 5)
print(f"\n샘플 < 5개인 그리드: {few_samples}개 ({few_samples/len(grid_samples)*100:.1f}%)")

# ============================================================================
# 4. 센서값 범위 및 SNR 추정
# ============================================================================
print("\n" + "="*70)
print("4. 센서값 범위 및 SNR")
print("="*70)

# 전체 데이터의 센서값 통계
all_mag = states_train[:, :, :3].reshape(-1, 3)  # MagX, Y, Z
all_orient = states_train[:, :, 3:].reshape(-1, 3)  # Pitch, Roll, Yaw

print(f"\n지자기 (μT):")
print(f"  범위: [{all_mag.min():.2f}, {all_mag.max():.2f}]")
print(f"  평균: {all_mag.mean():.2f}")
print(f"  표준편차: {all_mag.std():.2f}")

print(f"\n방향 (도):")
print(f"  범위: [{all_orient.min():.2f}, {all_orient.max():.2f}]")
print(f"  평균: {all_orient.mean():.2f}")
print(f"  표준편차: {all_orient.std():.2f}")

# SNR 추정 (신호 범위 / 노이즈)
signal_range_mag = all_mag.max() - all_mag.min()
noise_mag = df_var['mag_var'].mean() ** 0.5  # std
snr_mag = signal_range_mag / noise_mag if noise_mag > 0 else float('inf')

print(f"\n추정 SNR:")
print(f"  지자기: {snr_mag:.2f}")

# ============================================================================
# 5. 달성 가능성 추정
# ============================================================================
print("\n" + "="*70)
print("5. 1.5m @ 90% 달성 가능성 분석")
print("="*70)

# 그리드 크기가 0.9m이고, 각 그리드 내 variance를 봤을 때
# 이론적 최소 오차 추정
theoretical_min_error = GRID_SIZE / 2  # 그리드 중심에서 최대 거리

print(f"\n그리드 기반 이론적 최소 오차: {theoretical_min_error:.2f}m")

# 샘플 밀도로 추정
avg_samples_per_grid = np.mean(samples_per_grid)
if avg_samples_per_grid < 3:
    print("\n⚠️  경고: 그리드당 평균 샘플 수가 매우 적습니다.")
    print(f"   평균 {avg_samples_per_grid:.1f}개 - 일반화 어려울 수 있음")

# Variance가 큰 그리드
high_var_grids = len(df_var[df_var['mag_var'] > df_var['mag_var'].quantile(0.75)])
print(f"\nVariance 상위 25% 그리드: {high_var_grids}개")
print("  → 이 영역들은 학습이 어려울 수 있음")

# 결론
print("\n" + "="*70)
print("📊 종합 평가")
print("="*70)

feasibility_score = 0
max_score = 5

# 1. 샘플 밀도
if len(states_train) / area > 50:
    print("✅ 샘플 밀도: 충분함")
    feasibility_score += 1
else:
    print("⚠️  샘플 밀도: 부족할 수 있음")

# 2. 그리드당 샘플
if avg_samples_per_grid >= 5:
    print("✅ 그리드당 샘플: 충분함")
    feasibility_score += 1
else:
    print("⚠️  그리드당 샘플: 부족함")

# 3. SNR
if snr_mag > 10:
    print("✅ SNR: 양호")
    feasibility_score += 1
elif snr_mag > 5:
    print("⚠️  SNR: 보통")
    feasibility_score += 0.5
else:
    print("❌ SNR: 낮음")

# 4. Variance
if df_var['mag_var'].mean() < 1.0:
    print("✅ Variance: 낮음 (좋음)")
    feasibility_score += 1
else:
    print("⚠️  Variance: 높음")

# 5. 그리드 크기 vs 목표
if theoretical_min_error < 1.5:
    print("✅ 그리드 크기: 목표 달성 가능")
    feasibility_score += 1
else:
    print("❌ 그리드 크기: 목표 달성 어려움")

print(f"\n종합 점수: {feasibility_score:.1f}/{max_score}")

if feasibility_score >= 4:
    print("✅ 1.5m @ 90% 달성 가능성: 높음")
elif feasibility_score >= 2.5:
    print("⚠️  1.5m @ 90% 달성 가능성: 보통 - 튜닝 필요")
else:
    print("❌ 1.5m @ 90% 달성 가능성: 낮음 - 데이터 재수집 권장")

# ============================================================================
# 시각화
# ============================================================================
print("\n시각화 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 위치 분포
ax = axes[0, 0]
ax.scatter(positions_train_real[:, 0], positions_train_real[:, 1],
           alpha=0.1, s=1, c='blue')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Train 데이터 위치 분포')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 2. 그리드별 샘플 수
ax = axes[0, 1]
ax.hist(samples_per_grid, bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('그리드당 샘플 수')
ax.set_ylabel('그리드 수')
ax.set_title('그리드별 샘플 분포')
ax.grid(True, alpha=0.3)

# 3. Variance 분포
ax = axes[0, 2]
ax.hist(df_var['mag_var'], bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('지자기 Variance')
ax.set_ylabel('그리드 수')
ax.set_title('그리드별 지자기 Variance')
ax.grid(True, alpha=0.3)

# 4. 지자기 값 분포
ax = axes[1, 0]
for i, name in enumerate(['MagX', 'MagY', 'MagZ']):
    ax.hist(all_mag[:, i], bins=50, alpha=0.5, label=name)
ax.set_xlabel('값 (μT)')
ax.set_ylabel('빈도')
ax.set_title('지자기 값 분포')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 방향 값 분포
ax = axes[1, 1]
for i, name in enumerate(['Pitch', 'Roll', 'Yaw']):
    ax.hist(all_orient[:, i], bins=50, alpha=0.5, label=name)
ax.set_xlabel('각도 (도)')
ax.set_ylabel('빈도')
ax.set_title('방향 값 분포')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Variance vs 샘플 수
ax = axes[1, 2]
ax.scatter(df_var['n_samples'], df_var['mag_var'], alpha=0.5)
ax.set_xlabel('그리드당 샘플 수')
ax.set_ylabel('지자기 Variance')
ax.set_title('샘플 수 vs Variance')
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path = Path('results/data_analysis.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"저장: {output_path}")

print("\n" + "="*70)
print("분석 완료")
print("="*70)
