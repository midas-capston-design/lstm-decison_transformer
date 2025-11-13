#!/usr/bin/env python3
"""
데이터 판별력 체크: 위치별로 센서 패턴이 고유한가?
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For saving plots without display

print("="*70)
print("🔍 데이터 판별력 체크")
print("="*70)

# 데이터 로드 (원본만 사용 - 증강 제외)
data_dir = Path(__file__).resolve().parent / 'processed_data_flow_matching'
states_train_all = np.load(data_dir / 'states_train.npy')
coords_train_all = np.load(data_dir / 'coords_train.npy')

# 원본만 사용 (Train: 27,243개 = 18,162 원본 + 9,081 증강)
# 증강은 인위적 패턴이므로 실제 센서 데이터의 분리도만 테스트
N_ORIGINAL = 18162
states_train = states_train_all[:N_ORIGINAL]
coords_train = coords_train_all[:N_ORIGINAL]

print(f"\nTrain (전체): {len(states_train_all):,}개")
print(f"Train (원본만): {len(states_train):,}개 - 증강 제외하고 테스트")

# Grid 0.9m 기준으로 위치 그룹화
def coord_to_grid(x, y, grid_size=0.9):
    """좌표를 grid ID로"""
    # 역정규화 (-1~1 → 실제 좌표)
    # 건물 범위 약 85.5m x 18m
    x_real = (x + 1) / 2 * 85.5
    y_real = (y + 1) / 2 * 18

    grid_x = int(x_real / grid_size)
    grid_y = int(y_real / grid_size)
    return (grid_x, grid_y)

# 위치별 샘플 그룹화
location_samples = defaultdict(list)

print("\n[1/3] 위치별 샘플 그룹화...")
for i in tqdm(range(len(coords_train))):
    grid = coord_to_grid(coords_train[i, 0], coords_train[i, 1])
    location_samples[grid].append(i)

# 샘플 2개 이상인 위치만
locations_with_multiple = {k: v for k, v in location_samples.items() if len(v) >= 2}

print(f"\n통계:")
print(f"  총 위치: {len(location_samples)}개")
print(f"  샘플 2개 이상 위치: {len(locations_with_multiple)}개")
print(f"  평균 샘플/위치: {len(states_train)/len(location_samples):.1f}개")

# 샘플 많은 상위 10개 위치
top_locations = sorted(location_samples.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print(f"\n샘플 많은 위치 TOP 10:")
for i, (loc, samples) in enumerate(top_locations, 1):
    print(f"  {i}. Grid {loc}: {len(samples)}개 샘플")

print("\n[2/3] 같은 위치 vs 다른 위치 거리 계산...")

# 센서 거리 함수
def sensor_distance(s1, s2):
    """두 센서 시퀀스의 거리"""
    return np.linalg.norm(s1 - s2)

# 샘플 많은 위치 5개로 테스트
test_locations = top_locations[:5]

same_location_distances = []
diff_location_distances = []

for loc, sample_indices in tqdm(test_locations[:5], desc="Computing"):
    # 같은 위치 내 거리
    if len(sample_indices) >= 2:
        for i in range(min(10, len(sample_indices))):
            for j in range(i+1, min(10, len(sample_indices))):
                idx1, idx2 = sample_indices[i], sample_indices[j]
                dist = sensor_distance(states_train[idx1], states_train[idx2])
                same_location_distances.append(dist)

    # 다른 위치와의 거리
    other_loc, other_indices = test_locations[1] if loc == test_locations[0][0] else test_locations[0]
    for i in range(min(10, len(sample_indices))):
        for j in range(min(10, len(other_indices))):
            idx1 = sample_indices[i]
            idx2 = other_indices[j]
            dist = sensor_distance(states_train[idx1], states_train[idx2])
            diff_location_distances.append(dist)

same_location_distances = np.array(same_location_distances)
diff_location_distances = np.array(diff_location_distances)

print("\n[3/3] 결과 분석:")
print(f"\n📊 센서 패턴 거리:")
print(f"  같은 위치끼리: 평균 {same_location_distances.mean():.4f} (std {same_location_distances.std():.4f})")
print(f"  다른 위치끼리: 평균 {diff_location_distances.mean():.4f} (std {diff_location_distances.std():.4f})")

# 판별력 지표
ratio = diff_location_distances.mean() / same_location_distances.mean()
overlap = np.percentile(diff_location_distances, 25) < np.percentile(same_location_distances, 75)

print(f"\n💡 판별력 분석:")
print(f"  거리 비율 (다른 위치 / 같은 위치): {ratio:.2f}x")

if ratio > 2.0:
    print(f"  ✅ 비율 {ratio:.1f}x → 위치별 패턴 구분 가능!")
elif ratio > 1.3:
    print(f"  ⚠️ 비율 {ratio:.1f}x → 어느정도 구분 가능, 어려움")
else:
    print(f"  ❌ 비율 {ratio:.1f}x → 위치 구분 거의 불가능!")

if overlap:
    print(f"  ⚠️ 분포 겹침 큼 → 모델 학습 어려움")
else:
    print(f"  ✅ 분포 분리됨 → 모델 학습 가능")

print("\n" + "="*70)
if ratio > 1.5:
    print("✅ 데이터에 위치 정보 있음 → 모델 개선 가능")
else:
    print("❌ 데이터 자체 문제 → 센서가 위치 구분 못함")
print("="*70)

# 시각화
print("\n[4/4] 시각화 생성...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distance Distribution Histogram
ax1 = axes[0, 0]
bins = np.linspace(
    min(same_location_distances.min(), diff_location_distances.min()),
    max(same_location_distances.max(), diff_location_distances.max()),
    50
)
ax1.hist(same_location_distances, bins=bins, alpha=0.6, color='blue', label='Same Location', density=True)
ax1.hist(diff_location_distances, bins=bins, alpha=0.6, color='red', label='Different Location', density=True)
ax1.axvline(same_location_distances.mean(), color='blue', linestyle='--', linewidth=2, label=f'Same Mean: {same_location_distances.mean():.2f}')
ax1.axvline(diff_location_distances.mean(), color='red', linestyle='--', linewidth=2, label=f'Diff Mean: {diff_location_distances.mean():.2f}')
ax1.set_xlabel('Sensor Pattern Distance', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Sensor Pattern Distance Distribution (Original Data)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Box Plot
ax2 = axes[0, 1]
data_to_plot = [same_location_distances, diff_location_distances]
bp = ax2.boxplot(data_to_plot, tick_labels=['Same Location', 'Different Location'], patch_artist=True)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor('red')
bp['boxes'][1].set_alpha(0.6)
ax2.set_ylabel('Sensor Pattern Distance', fontsize=12)
ax2.set_title('Distance Comparison (Box Plot)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.text(0.5, 0.95, f'Ratio: {ratio:.2f}x', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# 3. Samples per Location Distribution
ax3 = axes[1, 0]
samples_per_location = [len(v) for v in location_samples.values()]
ax3.hist(samples_per_location, bins=30, color='green', alpha=0.7, edgecolor='black')
ax3.axvline(np.mean(samples_per_location), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(samples_per_location):.1f}')
ax3.set_xlabel('Samples per Location', fontsize=12)
ax3.set_ylabel('Number of Locations', fontsize=12)
ax3.set_title('Sample Distribution by Location', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Spatial Location Distribution (2D) - ACTUAL COORDINATES
ax4 = axes[1, 1]
# De-normalize original coordinates to real meters
x_real = (coords_train[:, 0] + 1) / 2 * 85.5
y_real = (coords_train[:, 1] + 1) / 2 * 18

scatter = ax4.scatter(x_real, y_real, s=1, alpha=0.3, c='blue')
ax4.set_xlabel('X Coordinate (m)', fontsize=12)
ax4.set_ylabel('Y Coordinate (m)', fontsize=12)
ax4.set_title('Actual Data Collection Path (Original Coordinates)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 90)
ax4.set_ylim(-2, 20)
ax4.set_aspect('equal', adjustable='box')

# Add range info
ax4.text(0.02, 0.98, f'X: [{x_real.min():.1f}, {x_real.max():.1f}]m\nY: [{y_real.min():.1f}, {y_real.max():.1f}]m',
         transform=ax4.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('separability_analysis.png', dpi=150, bbox_inches='tight')
print(f"✅ 시각화 저장: separability_analysis.png")

print("\n" + "="*70)
print("🎯 핵심 요약:")
print(f"  - 총 위치: {len(location_samples)}개")
print(f"  - 평균 샘플/위치: {len(states_train)/len(location_samples):.1f}개")
print(f"  - 같은 위치 거리: {same_location_distances.mean():.2f} ± {same_location_distances.std():.2f}")
print(f"  - 다른 위치 거리: {diff_location_distances.mean():.2f} ± {diff_location_distances.std():.2f}")
print(f"  - 분리도 비율: {ratio:.2f}x (2.0x 이상이면 우수)")
print("="*70)
