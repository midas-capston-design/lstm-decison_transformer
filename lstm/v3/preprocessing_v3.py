#!/usr/bin/env python3
"""
지자기 기반 실내 위치 추정 - 데이터 전처리 v3
- Stride 축소 (10 → 5)
- 보수적 데이터 증강 (센서 노이즈 수준)
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import defaultdict

# 설정
WINDOW_SIZE = 100  # 샘플
STRIDE = 5         # Sliding window stride (5샘플 = 0.1초 간격)
GRID_SIZE = 0.45   # m
SENSOR_COLS = ['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']

# 데이터 증강 설정
AUGMENT_PROB = 0.3  # 30% 샘플만 증강
MAG_NOISE_STD = 0.8  # 자기장 센서 노이즈 (μT)
ORIENTATION_NOISE_STD = 1.5  # 방향 센서 노이즈 (도)

print("="*70)
print("🔧 데이터 전처리 v3 시작 (Stride=5 + 데이터 증강)")
print("="*70)
print(f"  Window: {WINDOW_SIZE} 샘플 (2초 @ 50Hz)")
print(f"  Stride: {STRIDE} 샘플 (0.1초)")
print(f"  Grid: {GRID_SIZE}m")
print(f"  증강 확률: {AUGMENT_PROB*100:.0f}%")
print(f"  자기장 노이즈: std={MAG_NOISE_STD}μT")
print(f"  방향 노이즈: std={ORIENTATION_NOISE_STD}°")

# ============================================================================
# 데이터 증강 함수
# ============================================================================

def augment_sequence(seq):
    """
    센서 노이즈 수준의 보수적 증강

    Args:
        seq: (100, 6) numpy array [MagX, MagY, MagZ, Pitch, Roll, Yaw]

    Returns:
        augmented seq: (100, 6)
    """
    seq_aug = seq.copy()

    # 자기장 센서 노이즈 추가 (MagX, MagY, MagZ)
    mag_noise = np.random.normal(0, MAG_NOISE_STD, (seq.shape[0], 3))
    seq_aug[:, 0:3] += mag_noise

    # 방향 센서 노이즈 추가 (Pitch, Roll, Yaw)
    orientation_noise = np.random.normal(0, ORIENTATION_NOISE_STD, (seq.shape[0], 3))
    seq_aug[:, 3:6] += orientation_noise

    return seq_aug

# ============================================================================
# 1단계: 노드 정보 로드
# ============================================================================
print("\n[1/7] 노드 정보 로드...")
nodes_df = pd.read_csv('nodes_final.csv')
node_positions = {row['id']: (row['x_m'], row['y_m'])
                  for _, row in nodes_df.iterrows()}

# 건물 범위 계산
x_coords = [pos[0] for pos in node_positions.values()]
y_coords = [pos[1] for pos in node_positions.values()]
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

print(f"  건물 범위: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

# ============================================================================
# 2단계: 그리드 매핑 함수
# ============================================================================

def coord_to_grid(x, y):
    """절대 좌표를 그리드 ID로 변환"""
    grid_x = int(round((x - x_min) / GRID_SIZE))
    grid_y = int(round((y - y_min) / GRID_SIZE))

    num_x_grids = int(np.ceil((x_max - x_min) / GRID_SIZE)) + 1
    num_y_grids = int(np.ceil((y_max - y_min) / GRID_SIZE)) + 1

    grid_x = max(0, min(grid_x, num_x_grids - 1))
    grid_y = max(0, min(grid_y, num_y_grids - 1))

    grid_id = grid_y * num_x_grids + grid_x
    return grid_id


def calculate_marker_coordinates(start_pos, end_pos, num_markers):
    """시작점에서 끝점까지 num_markers 개의 좌표 생성"""
    coords = []
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]

    for i in range(num_markers):
        progress = i / (num_markers - 1) if num_markers > 1 else 0
        x = start_pos[0] + dx * progress
        y = start_pos[1] + dy * progress
        coords.append((x, y))

    return coords


def interpolate_position(pos1, pos2, t1, t2, t_current):
    """
    두 마커 사이에서 현재 샘플의 위치를 선형 보간

    Args:
        pos1, pos2: 마커 A, B의 좌표
        t1, t2: 마커 A, B의 인덱스
        t_current: 현재 샘플 인덱스
    """
    if t2 == t1:
        return pos1

    progress = (t_current - t1) / (t2 - t1)
    progress = max(0, min(1, progress))  # 0-1로 클리핑

    x = pos1[0] + (pos2[0] - pos1[0]) * progress
    y = pos1[1] + (pos2[1] - pos1[1]) * progress

    return (x, y)


# ============================================================================
# 3단계: 파일별 처리 (마커 사이 데이터 포함 + 증강)
# ============================================================================

def process_file_v3(filepath):
    """
    하나의 경로 파일 처리 (Stride=5 + 데이터 증강)

    Returns:
        sequences: (N, 100, 6)
        labels: (N,)
        coords: (N, 2)
    """
    filename = os.path.basename(filepath)
    parts = filename.replace('.csv', '').split('_')

    if len(parts) != 3:
        return None

    start_node = int(parts[0])
    end_node = int(parts[1])

    if start_node not in node_positions or end_node not in node_positions:
        return None

    # 데이터 로드
    df = pd.read_csv(filepath)

    # 센서 컬럼 확인
    if not all(col in df.columns for col in SENSOR_COLS):
        return None

    # Highlighted 마커 인덱스
    highlighted_indices = df[df['Highlighted'] == True].index.tolist()
    num_markers = len(highlighted_indices)

    if num_markers < 2:  # 최소 2개 마커 필요
        return None

    # 마커 좌표 계산
    start_pos = node_positions[start_node]
    end_pos = node_positions[end_node]
    marker_coords = calculate_marker_coordinates(start_pos, end_pos, num_markers)

    sequences = []
    labels = []
    coords_list = []

    # 연속된 마커 쌍 순회
    for i in range(len(highlighted_indices) - 1):
        marker_idx_A = highlighted_indices[i]
        marker_idx_B = highlighted_indices[i + 1]
        coord_A = marker_coords[i]
        coord_B = marker_coords[i + 1]

        # 마커 A와 B 사이의 모든 샘플 순회 (Sliding Window with STRIDE=5)
        for center_idx in range(marker_idx_A, marker_idx_B, STRIDE):
            # Window 범위
            start_idx = center_idx - WINDOW_SIZE
            end_idx = center_idx

            # 범위 체크
            if start_idx < 0:
                continue
            if end_idx > len(df):
                break

            # 센서 데이터 추출
            seq = df.iloc[start_idx:end_idx][SENSOR_COLS].values

            # 길이 체크
            if len(seq) != WINDOW_SIZE:
                continue

            # 현재 샘플의 위치 보간
            x, y = interpolate_position(coord_A, coord_B,
                                        marker_idx_A, marker_idx_B,
                                        center_idx)

            # 그리드 ID
            grid_id = coord_to_grid(x, y)

            # 원본 데이터 추가
            sequences.append(seq)
            labels.append(grid_id)
            coords_list.append((x, y))

            # 데이터 증강 (확률적)
            if np.random.random() < AUGMENT_PROB:
                seq_aug = augment_sequence(seq)
                sequences.append(seq_aug)
                labels.append(grid_id)
                coords_list.append((x, y))

    if len(sequences) == 0:
        return None

    return {
        'sequences': np.array(sequences),
        'labels': np.array(labels),
        'coords': np.array(coords_list),
        'route': f"{start_node}→{end_node}",
    }


# ============================================================================
# 4단계: 전체 데이터셋 생성
# ============================================================================

print("\n[2/7] 데이터 파일 처리 (Stride=5 + 증강)...")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
data_dir = PROJECT_ROOT / 'law_data'
files = sorted(list(data_dir.glob('*.csv')))

all_data = []
grid_stats = defaultdict(int)

np.random.seed(42)  # 재현성

for filepath in tqdm(files, desc="Processing files"):
    result = process_file_v3(filepath)
    if result is not None:
        all_data.append(result)

        for label in result['labels']:
            grid_stats[label] += 1

total_samples = sum(len(d['sequences']) for d in all_data)

print(f"\n  처리된 파일: {len(all_data)}/{len(files)}")
print(f"  고유 그리드 셀: {len(grid_stats)}")
print(f"  총 샘플 수: {total_samples:,}")
print(f"  증가율: {total_samples / 12179:.1f}배 (v1 대비)")

# ============================================================================
# 5단계: 데이터 통합 및 정규화
# ============================================================================

print("\n[3/7] 데이터 통합 및 정규화...")

all_sequences = np.vstack([d['sequences'] for d in all_data])
all_labels = np.concatenate([d['labels'] for d in all_data])
all_coords = np.vstack([d['coords'] for d in all_data])

print(f"  통합 데이터 shape: {all_sequences.shape}")
print(f"  라벨 shape: {all_labels.shape}")

# 정규화
mean = all_sequences.mean(axis=(0, 1))
std = all_sequences.std(axis=(0, 1))

print(f"\n  정규화 파라미터:")
for i, col in enumerate(SENSOR_COLS):
    print(f"    {col}: mean={mean[i]:.4f}, std={std[i]:.4f}")

all_sequences_norm = (all_sequences - mean) / (std + 1e-8)

# ============================================================================
# 6단계: Train/Val/Test 분할
# ============================================================================

print("\n[4/7] Train/Val/Test 분할...")

# 경로별 그룹화
route_groups = defaultdict(list)
for i, data in enumerate(all_data):
    route_groups[data['route']].append(i)

routes = list(route_groups.keys())
np.random.seed(42)
np.random.shuffle(routes)

n_routes = len(routes)
n_train = int(0.7 * n_routes)
n_val = int(0.15 * n_routes)

train_routes = routes[:n_train]
val_routes = routes[n_train:n_train+n_val]
test_routes = routes[n_train+n_val:]

print(f"  Train 경로: {len(train_routes)}")
print(f"  Val 경로: {len(val_routes)}")
print(f"  Test 경로: {len(test_routes)}")

# 인덱스 수집
train_indices = []
val_indices = []
test_indices = []

idx_offset = 0
for data in all_data:
    route = data['route']
    n_samples = len(data['sequences'])
    indices = list(range(idx_offset, idx_offset + n_samples))

    if route in train_routes:
        train_indices.extend(indices)
    elif route in val_routes:
        val_indices.extend(indices)
    else:
        test_indices.extend(indices)

    idx_offset += n_samples

print(f"\n  Train 샘플: {len(train_indices):,}")
print(f"  Val 샘플: {len(val_indices):,}")
print(f"  Test 샘플: {len(test_indices):,}")

# ============================================================================
# 7단계: 클래스 분포 분석
# ============================================================================

print("\n[5/7] 클래스 분포 분석...")

y_train = all_labels[train_indices]
unique, counts = np.unique(y_train, return_counts=True)

print(f"  Train 클래스 수: {len(unique):,}")
print(f"  클래스당 평균 샘플: {counts.mean():.1f}개")
print(f"  클래스당 최소: {counts.min()}개")
print(f"  클래스당 최대: {counts.max()}개")

# 샘플 수별 분포
bins = [1, 5, 10, 20, 50, 100, 200]
print(f"\n  샘플 수별 클래스 분포:")
for i in range(len(bins)-1):
    count = np.sum((counts >= bins[i]) & (counts < bins[i+1]))
    print(f"    {bins[i]:3d}-{bins[i+1]:3d}개: {count:4d} 클래스")
count = np.sum(counts >= bins[-1])
print(f"    {bins[-1]:3d}+개:   {count:4d} 클래스")

# ============================================================================
# 8단계: 클래스 필터링 (선택적)
# ============================================================================

print("\n[6/7] 클래스 필터링...")

MIN_SAMPLES = 10  # 최소 10개 샘플 이상 클래스만 유지

# 충분한 샘플이 있는 클래스만 선택
valid_classes = unique[counts >= MIN_SAMPLES]
print(f"  필터링 전 클래스: {len(unique)}")
print(f"  필터링 후 클래스: {len(valid_classes)} (>= {MIN_SAMPLES} 샘플)")

# 필터링된 인덱스
filtered_train_indices = [i for i in train_indices if all_labels[i] in valid_classes]
filtered_val_indices = [i for i in val_indices if all_labels[i] in valid_classes]
filtered_test_indices = [i for i in test_indices if all_labels[i] in valid_classes]

print(f"\n  필터링 후 샘플:")
print(f"    Train: {len(filtered_train_indices):,} ({len(filtered_train_indices)/len(train_indices)*100:.1f}% 유지)")
print(f"    Val: {len(filtered_val_indices):,} ({len(filtered_val_indices)/len(val_indices)*100:.1f}% 유지)")
print(f"    Test: {len(filtered_test_indices):,} ({len(filtered_test_indices)/len(test_indices)*100:.1f}% 유지)")

# 클래스 ID 재매핑
class_mapping = {old_id: new_id for new_id, old_id in enumerate(valid_classes)}
all_labels_remapped = np.array([class_mapping.get(label, -1) for label in all_labels])

# 최종 클래스 분포
y_train_final = all_labels_remapped[filtered_train_indices]
unique_final, counts_final = np.unique(y_train_final, return_counts=True)

print(f"\n  최종 클래스 분포:")
print(f"    클래스 수: {len(unique_final)}")
print(f"    평균 샘플: {counts_final.mean():.1f}개")
print(f"    최소 샘플: {counts_final.min()}개")
print(f"    최대 샘플: {counts_final.max()}개")

# ============================================================================
# 9단계: 데이터셋 저장
# ============================================================================

print("\n[7/7] 데이터셋 저장...")

output_dir = Path('processed_data_v3')
output_dir.mkdir(exist_ok=True)

np.save(output_dir / 'X_train.npy', all_sequences_norm[filtered_train_indices])
np.save(output_dir / 'y_train.npy', all_labels_remapped[filtered_train_indices])
np.save(output_dir / 'coords_train.npy', all_coords[filtered_train_indices])

np.save(output_dir / 'X_val.npy', all_sequences_norm[filtered_val_indices])
np.save(output_dir / 'y_val.npy', all_labels_remapped[filtered_val_indices])
np.save(output_dir / 'coords_val.npy', all_coords[filtered_val_indices])

np.save(output_dir / 'X_test.npy', all_sequences_norm[filtered_test_indices])
np.save(output_dir / 'y_test.npy', all_labels_remapped[filtered_test_indices])
np.save(output_dir / 'coords_test.npy', all_coords[filtered_test_indices])

metadata = {
    'window_size': WINDOW_SIZE,
    'stride': STRIDE,
    'grid_size': GRID_SIZE,
    'sensor_cols': SENSOR_COLS,
    'mean': mean,
    'std': std,
    'num_classes': len(valid_classes),
    'class_mapping': class_mapping,
    'valid_classes': valid_classes.tolist(),
    'grid_stats': dict(grid_stats),
    'x_range': (x_min, x_max),
    'y_range': (y_min, y_max),
    'augmentation': {
        'augment_prob': AUGMENT_PROB,
        'mag_noise_std': MAG_NOISE_STD,
        'orientation_noise_std': ORIENTATION_NOISE_STD,
    }
}

with open(output_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"  저장 완료: {output_dir}")

# ============================================================================
# 10단계: 검증
# ============================================================================

X_train = np.load(output_dir / 'X_train.npy')
y_train = np.load(output_dir / 'y_train.npy')

print("\n" + "="*70)
print("✅ 전처리 v3 완료!")
print("="*70)
print(f"""
📊 최종 데이터셋:
  Train: {len(filtered_train_indices):,} 샘플
  Val:   {len(filtered_val_indices):,} 샘플
  Test:  {len(filtered_test_indices):,} 샘플
  총합:  {len(filtered_train_indices) + len(filtered_val_indices) + len(filtered_test_indices):,} 샘플

  증가율: {total_samples / 12179:.1f}배 (v1 대비)
  필터링 후: {(len(filtered_train_indices) + len(filtered_val_indices) + len(filtered_test_indices)) / 12179:.1f}배

  클래스: {len(valid_classes):,}개
  클래스당 평균: {counts_final.mean():.1f}개 샘플

  입력 shape: {X_train.shape}

저장 위치: {output_dir}/

🎯 v3 개선 효과:
  ✅ Stride 축소 (10 → 5): 샘플 2배 증가
  ✅ 데이터 증강 (30%): 추가 1.3배 증가
  ✅ 클래스 필터링: 학습 가능한 클래스만 유지
  ✅ 최종 클래스당 평균 {counts_final.mean():.1f}개 샘플
  ✅ 전체 데이터 퀄리티 향상

이제 LSTM 모델 학습을 시작할 수 있습니다! 🚀
""")
