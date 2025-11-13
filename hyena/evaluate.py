#!/usr/bin/env python3
"""
Hyena for Indoor Positioning - Evaluation Script

평가 항목:
1. Test set position error
2. Inference speed
3. Visualization
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from model import HyenaLocalization
import pickle

print("=" * 70)
print("📊 Hyena Evaluation")
print("=" * 70)

# ============================================================================
# Configuration
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {DEVICE}")

# ============================================================================
# Denormalization
# ============================================================================
def denormalize_coords(coords_norm, coords_min, coords_max):
    """
    정규화된 좌표를 실제 좌표로 변환

    Args:
        coords_norm: (-1, 1) 범위의 정규화된 좌표
        coords_min: 원본 최소값
        coords_max: 원본 최대값

    Returns:
        실제 좌표 (미터)
    """
    coords_range = coords_max - coords_min
    coords_real = (coords_norm + 1) / 2 * coords_range + coords_min
    return coords_real

# ============================================================================
# Dataset
# ============================================================================
class HyenaDataset(Dataset):
    def __init__(self, states, positions):
        self.states = torch.FloatTensor(states)
        self.positions = torch.FloatTensor(positions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'sensor_data': self.states[idx],
            'position': self.positions[idx],
        }

# ============================================================================
# Evaluation
# ============================================================================
def evaluate():
    print("\n[1/5] 모델 로드...")

    model_path = Path(__file__).parent.parent / 'models' / 'hyena_best.pt'

    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 train.py를 실행하여 모델을 학습하세요.")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint['config']

    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Position Error: {checkpoint.get('val_position_error', 'N/A')}")

    # 모델 초기화
    model = HyenaLocalization(
        input_dim=config['input_dim'],
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        order=config['order'],
        filter_order=config['filter_order'],
        dropout=config['dropout']
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\n[2/5] Test 데이터 로드...")

    data_dir = Path(__file__).parent / 'processed_data_hyena'

    states_test = np.load(data_dir / 'states_test.npy')
    positions_test = np.load(data_dir / 'positions_test.npy')

    # Metadata 로드 (denormalization용)
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    coords_min = np.array(metadata['normalization']['coords_min'])
    coords_max = np.array(metadata['normalization']['coords_max'])

    print(f"  Test: {states_test.shape}")
    print(f"  좌표 범위: x=[{coords_min[0]:.2f}, {coords_max[0]:.2f}], y=[{coords_min[1]:.2f}, {coords_max[1]:.2f}]")

    test_dataset = HyenaDataset(states_test, positions_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    print(f"  Test batches: {len(test_loader)}")

    # ============================================================================
    # 성능 평가
    # ============================================================================
    print("\n[3/5] Test set 평가...")

    position_errors = []
    all_predictions_norm = []
    all_targets_norm = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  평가 중"):
            sensor_data = batch['sensor_data'].to(DEVICE)
            positions_norm = batch['position'].to(DEVICE)

            # Predict (normalized)
            pred_positions_norm = model(sensor_data)

            all_predictions_norm.append(pred_positions_norm.cpu().numpy())
            all_targets_norm.append(positions_norm.cpu().numpy())

    all_predictions_norm = np.concatenate(all_predictions_norm, axis=0)
    all_targets_norm = np.concatenate(all_targets_norm, axis=0)

    # Denormalize to real coordinates (meters)
    all_predictions_real = denormalize_coords(all_predictions_norm, coords_min, coords_max)
    all_targets_real = denormalize_coords(all_targets_norm, coords_min, coords_max)

    # Calculate error in meters
    position_errors = np.linalg.norm(all_predictions_real - all_targets_real, axis=1)

    print(f"  Mean Error: {position_errors.mean():.4f} m")
    print(f"  Std Error: {position_errors.std():.4f} m")
    print(f"  Median Error: {np.median(position_errors):.4f} m")
    print(f"  90th percentile: {np.percentile(position_errors, 90):.4f} m")
    print(f"  95th percentile: {np.percentile(position_errors, 95):.4f} m")

    # ============================================================================
    # Inference Speed 측정
    # ============================================================================
    print("\n[4/5] Inference Speed 측정...")

    test_batch = next(iter(test_loader))
    sensor_data = test_batch['sensor_data'][:32].to(DEVICE)

    times = []

    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(sensor_data)

        # Measure
        for _ in range(100):
            start = time.time()
            _ = model(sensor_data)
            end = time.time()
            times.append(end - start)

    avg_time = np.mean(times) * 1000  # ms
    print(f"  평균 추론 시간: {avg_time:.2f} ms/batch (32 samples)")
    print(f"  샘플당 추론 시간: {avg_time/32:.2f} ms/sample")

    # ============================================================================
    # 시각화
    # ============================================================================
    print("\n[5/5] 결과 시각화...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error distribution
    ax = axes[0]
    ax.hist(position_errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(position_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {position_errors.mean():.4f}m')
    ax.axvline(np.median(position_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(position_errors):.4f}m')
    ax.set_xlabel('Position Error (m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Position Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Predicted vs Actual (실제 좌표)
    ax = axes[1]
    sample_size = min(500, len(all_targets_real))
    ax.scatter(all_targets_real[:sample_size, 0], all_targets_real[:sample_size, 1],
               alpha=0.3, s=10, c='blue', label='Actual')
    ax.scatter(all_predictions_real[:sample_size, 0], all_predictions_real[:sample_size, 1],
               alpha=0.3, s=10, c='red', label='Predicted')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Predicted vs Actual Positions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    plt.savefig(results_dir / 'hyena_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"  저장: {results_dir / 'hyena_evaluation.png'}")

    # ============================================================================
    # 결과 저장
    # ============================================================================
    print("\n결과 저장 중...")

    summary = {
        'test_samples': len(test_dataset),
        'mean_error': float(position_errors.mean()),
        'std_error': float(position_errors.std()),
        'median_error': float(np.median(position_errors)),
        'p90_error': float(np.percentile(position_errors, 90)),
        'p95_error': float(np.percentile(position_errors, 95)),
        'inference_speed_ms': float(avg_time),
    }

    with open(results_dir / 'hyena_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Hyena Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Test Samples: {summary['test_samples']}\n\n")

        f.write("Position Error:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean:   {summary['mean_error']:.4f}\n")
        f.write(f"  Std:    {summary['std_error']:.4f}\n")
        f.write(f"  Median: {summary['median_error']:.4f}\n")
        f.write(f"  90th:   {summary['p90_error']:.4f}\n")
        f.write(f"  95th:   {summary['p95_error']:.4f}\n\n")

        f.write("Inference Speed:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {summary['inference_speed_ms']:.2f} ms/batch (32 samples)\n")
        f.write(f"  {summary['inference_speed_ms']/32:.2f} ms/sample\n")

    print(f"  저장: {results_dir / 'hyena_summary.txt'}")

    print("\n" + "=" * 70)
    print("✅ 평가 완료!")
    print("=" * 70)
    print(f"\n결과 파일:")
    print(f"  - {results_dir / 'hyena_evaluation.png'}")
    print(f"  - {results_dir / 'hyena_summary.txt'}")

    print(f"\n📊 최종 성능:")
    print(f"  Mean Error: {summary['mean_error']:.4f}")
    print(f"  Inference Speed: {summary['inference_speed_ms']:.2f} ms/batch")

if __name__ == '__main__':
    evaluate()
