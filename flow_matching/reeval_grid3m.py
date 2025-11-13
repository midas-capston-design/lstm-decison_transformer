#!/usr/bin/env python3
"""
Grid 3m로 재평가
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_flow_matching import FlowMatchingDataset, CONFIG
from model import FlowMatchingLocalization

print("="*70)
print("📊 Grid 3m로 재평가")
print("="*70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 로드
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / 'processed_data_flow_matching'
states_test = np.load(data_dir / 'states_test.npy', allow_pickle=True)
traj_test = np.load(data_dir / 'trajectories_test.npy', allow_pickle=True)

test_dataset = FlowMatchingDataset(states_test, traj_test, augment=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"\nTest 샘플: {len(test_dataset):,}개")

# 모델 로드
model = FlowMatchingLocalization(
    sensor_dim=6, position_dim=2, d_model=256,
    encoder_layers=4, velocity_layers=4, n_heads=8, dropout=0.1
).to(DEVICE)

checkpoint = torch.load(BASE_DIR.parent / 'models' / 'flow_matching_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Grid 크기들 테스트
grid_sizes_meters = [0.9, 1.5, 2.0, 3.0, 5.0]
print(f"\n다양한 Grid 크기로 평가:")

for grid_m in grid_sizes_meters:
    # 정규화된 grid 크기 (건물 범위 85.5m, 정규화 -1~1)
    grid_normalized = grid_m / 85.5 * 2

    correct_normal = 0
    correct_topk = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Grid {grid_m}m", leave=False):
            sensor_data = batch['sensor_data'].to(DEVICE)
            positions = batch['position'].to(DEVICE)

            # 일반 샘플링
            pred_pos = model.sample(sensor_data, n_steps=10)
            error = torch.norm(pred_pos - positions, dim=1)
            correct_normal += (error <= grid_normalized).sum().item()

            # Top-5 샘플링
            best_pos, topk_positions, topk_scores = model.sample_topk(
                sensor_data, n_samples=10, k=5, n_steps=10
            )

            # 5개 중 하나라도 맞으면 정답
            for b in range(len(positions)):
                target_pos = positions[b]
                candidates = topk_positions[b]  # (5, 2)
                errors = torch.norm(candidates - target_pos, dim=1)
                if (errors <= grid_normalized).any():
                    correct_topk += 1

            total += len(positions)

    acc_normal = correct_normal / total * 100
    acc_topk = correct_topk / total * 100

    print(f"\n  📍 Grid {grid_m}m (정규화: {grid_normalized:.4f}):")
    print(f"     일반 샘플링: {acc_normal:.2f}% ({correct_normal}/{total})")
    print(f"     Top-5 후보: {acc_topk:.2f}% ({correct_topk}/{total})")

print("\n" + "="*70)
print("✅ 재평가 완료!")
print("="*70)
