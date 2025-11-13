#!/usr/bin/env python3
"""
Flow Matching 진단 스크립트
"""
import torch
import numpy as np
from pathlib import Path
from train_flow_matching import FlowMatchingDataset
from model import FlowMatchingLocalization

print("="*70)
print("🔍 Flow Matching 진단")
print("="*70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parent

# 데이터 로드
data_dir = BASE_DIR / 'processed_data_flow_matching'
states_test = np.load(data_dir / 'states_test.npy')
traj_test = np.load(data_dir / 'trajectories_test.npy')

print(f"\n데이터:")
print(f"  Test: {states_test.shape}")
print(f"  센서 범위: [{states_test.min():.2f}, {states_test.max():.2f}]")
print(f"  위치 범위: [{traj_test[:, -1, :].min():.2f}, {traj_test[:, -1, :].max():.2f}]")

# 모델 로드
model = FlowMatchingLocalization(
    sensor_dim=6, position_dim=2, d_model=256,
    encoder_layers=4, velocity_layers=4, n_heads=8, dropout=0.1
).to(DEVICE)

checkpoint = torch.load(BASE_DIR.parent / 'models' / 'flow_matching_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 샘플 10개로 테스트
test_dataset = FlowMatchingDataset(states_test[:100], traj_test[:100], augment=False)

sensor_data = torch.stack([test_dataset[i]['sensor_data'] for i in range(10)]).to(DEVICE)
true_pos = torch.stack([test_dataset[i]['position'] for i in range(10)]).to(DEVICE)

with torch.no_grad():
    pred_pos = model.sample(sensor_data, n_steps=10)

print(f"\n예측 결과 (10개 샘플):")
print(f"{'idx':<5} {'True X':<10} {'True Y':<10} {'Pred X':<10} {'Pred Y':<10} {'Error':<10}")
print("-"*70)
for i in range(10):
    tx, ty = true_pos[i].cpu().numpy()
    px, py = pred_pos[i].cpu().numpy()
    error = np.linalg.norm([tx-px, ty-py])
    print(f"{i:<5} {tx:<10.4f} {ty:<10.4f} {px:<10.4f} {py:<10.4f} {error:<10.4f}")

print(f"\n통계:")
errors = torch.norm(pred_pos - true_pos, dim=1).cpu().numpy()
print(f"  평균 오차: {errors.mean():.4f}")
print(f"  최소 오차: {errors.min():.4f}")
print(f"  최대 오차: {errors.max():.4f}")

# Grid 크기
grid_size = 0.9 / 85.5 * 2  # 정규화된 grid
print(f"\n1 Grid 크기 (정규화): {grid_size:.4f}")
print(f"1 Grid 이내: {(errors <= grid_size).sum()}/10")

print("\n💡 분석:")
if errors.mean() > 1.0:
    print("  ⚠️ 평균 오차가 매우 큼 → 모델이 위치 학습 실패")
if errors.std() < 0.1:
    print("  ⚠️ 분산이 너무 작음 → 모델이 항상 비슷한 위치 예측")
if np.abs(pred_pos.cpu().numpy()).mean() > 2.0:
    print("  ⚠️ 예측 위치가 정규화 범위 벗어남 → 좌표계 문제")
