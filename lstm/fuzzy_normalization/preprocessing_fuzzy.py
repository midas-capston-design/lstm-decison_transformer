#!/usr/bin/env python3
"""
지자기 기반 실내 위치 추정 - Fuzzy 정규화 전처리
Fuzzy membership functions를 사용한 정규화
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import defaultdict

print("="*70)
print("🔧 데이터 전처리 v3 + Fuzzy 정규화")
print("="*70)

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'law_data'
NODES_FILE = Path('../nodes_final.csv')
OUTPUT_DIR = Path('processed_data_v3_fuzzy')
OUTPUT_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 100  # 시퀀스 길이
STRIDE = 5         # 슬라이딩 윈도우 stride
GRID_SIZE = 0.45   # 그리드 크기 (m)

# 데이터 증강 설정
AUGMENT_RATIO = 0.3
MAG_NOISE_STD = 0.8
ORIENTATION_NOISE_STD = 1.5

# ============================================================================
# Fuzzy Normalization Functions
# ============================================================================

def triangular_membership(x, a, b, c):
    """
    삼각형 멤버십 함수
    a: 왼쪽 끝
    b: 중심 (피크)
    c: 오른쪽 끝
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)

def trapezoidal_membership(x, a, b, c, d):
    """
    사다리꼴 멤버십 함수
    a: 왼쪽 시작
    b: 왼쪽 평탄 시작
    c: 오른쪽 평탄 끝
    d: 오른쪽 끝
    """
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1.0
    else:  # c < x < d
        return (d - x) / (d - a)

def fuzzy_normalize(data, feature_name):
    """
    Fuzzy 정규화: 각 특징에 대해 Low/Medium/High 멤버십 함수 적용
    출력: 각 샘플마다 (low, medium, high) membership 값
    """
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val

    # 퍼지 구간 정의
    q25 = np.percentile(data, 25)
    q50 = np.percentile(data, 50)
    q75 = np.percentile(data, 75)

    # Low: 삼각형 (min, min, q50)
    # Medium: 삼각형 (q25, q50, q75)
    # High: 삼각형 (q50, max, max)

    n_samples = len(data)
    fuzzy_features = np.zeros((n_samples, 3))

    for i, val in enumerate(data):
        # Low membership
        if val <= min_val:
            fuzzy_features[i, 0] = 1.0
        elif val <= q50:
            fuzzy_features[i, 0] = (q50 - val) / (q50 - min_val)
        else:
            fuzzy_features[i, 0] = 0.0

        # Medium membership (삼각형)
        fuzzy_features[i, 1] = triangular_membership(val, q25, q50, q75)

        # High membership
        if val >= max_val:
            fuzzy_features[i, 2] = 1.0
        elif val >= q50:
            fuzzy_features[i, 2] = (val - q50) / (max_val - q50)
        else:
            fuzzy_features[i, 2] = 0.0

    return fuzzy_features

def standard_normalize(data, mean, std):
    """표준 정규화 (비교용)"""
    return (data - mean) / (std + 1e-8)

# ============================================================================
# 1. 노드 정보 로드
# ============================================================================
print("\n[1/6] 노드 정보 로드...")
nodes_df = pd.read_csv(NODES_FILE)
print(f"  총 {len(nodes_df)}개 노드")

# ============================================================================
# 2. 모든 CSV 파일 로드 및 경로 분석
# ============================================================================
print("\n[2/6] CSV 파일 로드 및 분석...")

all_trajectories = []
route_info = defaultdict(list)

csv_files = sorted(DATA_DIR.glob('*.csv'))
print(f"  총 {len(csv_files)}개 파일")

for csv_file in tqdm(csv_files, desc="  파일 로드"):
    try:
        df = pd.read_csv(csv_file)

        # 파일명 파싱: start_end_trial.csv
        parts = csv_file.stem.split('_')
        start_node = int(parts[0])
        end_node = int(parts[1])
        trial = int(parts[2])

        all_trajectories.append({
            'file': csv_file.name,
            'start_node': start_node,
            'end_node': end_node,
            'trial': trial,
            'df': df
        })

        route_info[(start_node, end_node)].append(csv_file.name)

    except Exception as e:
        print(f"    오류 ({csv_file.name}): {e}")

print(f"  로드 완료: {len(all_trajectories)}개 궤적")
print(f"  고유 경로: {len(route_info)}개")

# ============================================================================
# 3. 마커 기반 절대 좌표 계산 및 시퀀스 생성
# ============================================================================
print("\n[3/6] 마커 기반 시퀀스 생성...")

sequences = []
all_labels = []

for traj in tqdm(all_trajectories, desc="  시퀀스 생성"):
    df = traj['df']
    start_node = traj['start_node']
    end_node = traj['end_node']

    # Highlighted 마커 추출
    marker_indices = df[df['Highlighted'] == True].index.tolist()

    if len(marker_indices) < 2:
        continue

    # 시작/끝 노드 좌표
    start_coord = nodes_df[nodes_df['Node'] == start_node][['X', 'Y']].values[0]
    end_coord = nodes_df[nodes_df['Node'] == end_node][['X', 'Y']].values[0]

    # 경로 벡터
    path_vector = end_coord - start_coord
    path_length = np.linalg.norm(path_vector)

    # 마커 간격 0.45m
    for i in range(len(marker_indices) - 1):
        marker_idx_A = marker_indices[i]
        marker_idx_B = marker_indices[i + 1]

        # 마커 A의 절대 좌표 계산
        progress = (i * GRID_SIZE) / path_length  # 경로 진행률
        marker_pos = start_coord + progress * path_vector

        # 마커 A와 B 사이의 모든 샘플에 대해 슬라이딩 윈도우
        for center_idx in range(marker_idx_A, marker_idx_B, STRIDE):
            start_idx = center_idx - WINDOW_SIZE
            end_idx = center_idx

            if start_idx < 0:
                continue

            # 윈도우 추출
            window = df.iloc[start_idx:end_idx]

            if len(window) != WINDOW_SIZE:
                continue

            # 특징 추출: MagX, MagY, MagZ, Pitch, Roll, Yaw
            features = window[['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']].values

            sequences.append(features)
            all_labels.append(marker_pos)

sequences = np.array(sequences)  # (N, 100, 6)
all_labels = np.array(all_labels)  # (N, 2)

print(f"  생성된 시퀀스: {len(sequences):,}개")

# ============================================================================
# 4. Fuzzy 정규화 적용
# ============================================================================
print("\n[4/6] Fuzzy 정규화 적용...")

# 각 특징에 대한 통계
feature_names = ['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']
fuzzy_sequences = []

for i, fname in enumerate(feature_names):
    print(f"  {fname} 정규화 중...")

    # 모든 샘플의 해당 특징 추출
    feature_data = sequences[:, :, i].flatten()

    # Fuzzy 정규화 (Low, Medium, High)
    fuzzy_feat = fuzzy_normalize(feature_data, fname)

    # 원래 shape으로 복원
    fuzzy_feat = fuzzy_feat.reshape(len(sequences), WINDOW_SIZE, 3)
    fuzzy_sequences.append(fuzzy_feat)

# 모든 특징 합치기: (N, 100, 6*3) = (N, 100, 18)
fuzzy_sequences = np.concatenate(fuzzy_sequences, axis=2)

print(f"  Fuzzy 정규화 완료: {fuzzy_sequences.shape}")

# ============================================================================
# 5. 그리드 매핑 및 라벨 생성
# ============================================================================
print("\n[5/6] 그리드 매핑...")

# 좌표를 그리드 ID로 변환
def coord_to_grid_id(x, y, grid_size=GRID_SIZE):
    grid_x = int(np.round(x / grid_size))
    grid_y = int(np.round(y / grid_size))
    return f"{grid_x}_{grid_y}"

grid_ids = [coord_to_grid_id(x, y) for x, y in all_labels]
unique_grids = sorted(set(grid_ids))
grid_to_idx = {grid: idx for idx, grid in enumerate(unique_grids)}
labels = np.array([grid_to_idx[gid] for gid in grid_ids])

print(f"  고유 그리드: {len(unique_grids)}개")

# 클래스 필터링 (10개 미만 샘플)
from collections import Counter
class_counts = Counter(labels)
valid_classes = [cls for cls, count in class_counts.items() if count >= 10]

print(f"  필터링 전 클래스: {len(unique_grids)}개")
print(f"  필터링 후 클래스: {len(valid_classes)}개")

# 필터링
valid_indices = [i for i, lbl in enumerate(labels) if lbl in valid_classes]
fuzzy_sequences = fuzzy_sequences[valid_indices]
labels = labels[valid_indices]

# 클래스 ID 재매핑
old_to_new = {old: new for new, old in enumerate(sorted(valid_classes))}
labels = np.array([old_to_new[lbl] for lbl in labels])

# ============================================================================
# 6. Train/Val/Test 분할 및 저장
# ============================================================================
print("\n[6/6] 데이터 분할 및 저장...")

# 경로 기반 분할
route_files = defaultdict(list)
for i, traj in enumerate(all_trajectories):
    route_key = (traj['start_node'], traj['end_node'])
    route_files[route_key].append(traj['file'])

unique_routes = list(route_files.keys())
np.random.shuffle(unique_routes)

n_routes = len(unique_routes)
n_train = int(0.7 * n_routes)
n_val = int(0.15 * n_routes)

train_routes = set(unique_routes[:n_train])
val_routes = set(unique_routes[n_train:n_train + n_val])
test_routes = set(unique_routes[n_train + n_val:])

# 샘플 분할
train_indices = []
val_indices = []
test_indices = []

for i, traj in enumerate(all_trajectories):
    if i >= len(valid_indices):
        continue
    route_key = (traj['start_node'], traj['end_node'])

    if route_key in train_routes:
        train_indices.append(valid_indices[i])
    elif route_key in val_routes:
        val_indices.append(valid_indices[i])
    else:
        test_indices.append(valid_indices[i])

# 실제 분할
X_train = fuzzy_sequences[train_indices]
y_train = labels[train_indices]

X_val = fuzzy_sequences[val_indices]
y_val = labels[val_indices]

X_test = fuzzy_sequences[test_indices]
y_test = labels[test_indices]

print(f"  Train: {len(X_train):,} 샘플 ({len(train_routes)} 경로)")
print(f"  Val:   {len(X_val):,} 샘플 ({len(val_routes)} 경로)")
print(f"  Test:  {len(X_test):,} 샘플 ({len(test_routes)} 경로)")

# 저장
np.save(OUTPUT_DIR / 'X_train.npy', X_train)
np.save(OUTPUT_DIR / 'y_train.npy', y_train)
np.save(OUTPUT_DIR / 'X_val.npy', X_val)
np.save(OUTPUT_DIR / 'y_val.npy', y_val)
np.save(OUTPUT_DIR / 'X_test.npy', X_test)
np.save(OUTPUT_DIR / 'y_test.npy', y_test)

# 메타데이터 저장
metadata = {
    'num_classes': len(valid_classes),
    'num_features': 18,  # 6 features * 3 fuzzy values
    'window_size': WINDOW_SIZE,
    'grid_size': GRID_SIZE,
    'fuzzy_normalization': True,
    'grid_to_idx': grid_to_idx,
    'feature_names': feature_names,
}

with open(OUTPUT_DIR / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"\n✅ Fuzzy 정규화 전처리 완료!")
print(f"  출력 디렉토리: {OUTPUT_DIR}")
print(f"  입력 shape: (batch, {WINDOW_SIZE}, 18)")
print(f"  출력 클래스: {len(valid_classes)}개")
