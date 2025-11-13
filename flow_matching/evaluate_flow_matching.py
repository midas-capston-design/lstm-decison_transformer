#!/usr/bin/env python3
"""
Flow Matching for Indoor Positioning - Evaluation Script

평가 항목:
1. Test set position error (다양한 inference steps)
2. Inference speed comparison
3. Visualization (predicted vs actual positions)
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from model import FlowMatchingLocalization

print("=" * 70)
print("📊 Flow Matching Evaluation")
print("=" * 70)

# ============================================================================
# Configuration
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {DEVICE}")

# ============================================================================
# Dataset
# ============================================================================
class FlowMatchingDataset(Dataset):
    def __init__(self, states, trajectories):
        self.states = torch.FloatTensor(states)
        self.positions = torch.FloatTensor(trajectories[:, -1, :])

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

    model_path = Path(__file__).parent.parent / 'models' / 'flow_matching_best.pt'

    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 train_flow_matching.py를 실행하여 모델을 학습하세요.")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint['config']

    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Position Error: {checkpoint.get('val_position_error', 'N/A')}")

    # 모델 초기화
    model = FlowMatchingLocalization(
        sensor_dim=config['sensor_dim'],
        position_dim=config['position_dim'],
        d_model=config['d_model'],
        encoder_layers=config['encoder_layers'],
        velocity_layers=config['velocity_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout']
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\n[2/5] Test 데이터 로드...")

    data_dir = Path(__file__).resolve().parent.parent / 'dt' / 'processed_data_dt'

    states_test = np.load(data_dir / 'states_test.npy')
    traj_test = np.load(data_dir / 'trajectories_test.npy')

    print(f"  Test: {states_test.shape}")

    test_dataset = FlowMatchingDataset(states_test, traj_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    print(f"  Test batches: {len(test_loader)}")

    # ============================================================================
    # 다양한 inference steps로 평가
    # ============================================================================
    print("\n[3/5] 다양한 inference steps로 평가...")

    inference_steps_list = [1, 2, 5, 10, 20]
    results = {}

    for n_steps in inference_steps_list:
        print(f"\n  Evaluating with {n_steps} steps...")

        position_errors = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  {n_steps} steps", leave=False):
                sensor_data = batch['sensor_data'].to(DEVICE)
                positions = batch['position'].to(DEVICE)

                # Predict
                pred_positions = model.sample(sensor_data, n_steps=n_steps)

                # Error (Euclidean distance)
                error = torch.norm(pred_positions - positions, dim=1)
                position_errors.extend(error.cpu().numpy())

                all_predictions.append(pred_positions.cpu().numpy())
                all_targets.append(positions.cpu().numpy())

        position_errors = np.array(position_errors)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        results[n_steps] = {
            'mean_error': position_errors.mean(),
            'std_error': position_errors.std(),
            'median_error': np.median(position_errors),
            'predictions': all_predictions,
            'targets': all_targets,
            'errors': position_errors
        }

        print(f"    Mean Error: {position_errors.mean():.4f}")
        print(f"    Std Error: {position_errors.std():.4f}")
        print(f"    Median Error: {np.median(position_errors):.4f}")

    # ============================================================================
    # Inference Speed 측정
    # ============================================================================
    print("\n[4/5] Inference Speed 측정...")

    test_batch = next(iter(test_loader))
    sensor_data = test_batch['sensor_data'][:32].to(DEVICE)

    speed_results = {}

    for n_steps in inference_steps_list:
        times = []

        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model.sample(sensor_data, n_steps=n_steps)

            # Measure
            for _ in range(100):
                start = time.time()
                _ = model.sample(sensor_data, n_steps=n_steps)
                end = time.time()
                times.append(end - start)

        avg_time = np.mean(times) * 1000  # ms
        speed_results[n_steps] = avg_time

        print(f"  {n_steps:2d} steps: {avg_time:.2f} ms/batch (32 samples)")

    # ============================================================================
    # 결과 출력
    # ============================================================================
    print("\n[5/5] 평가 결과 요약...")
    print("\n" + "=" * 70)
    print("📊 Test Set Performance")
    print("=" * 70)

    print("\n위치 오차 (Position Error):")
    print("-" * 70)
    print(f"{'Steps':<10} {'Mean':<12} {'Std':<12} {'Median':<12}")
    print("-" * 70)

    for n_steps in inference_steps_list:
        res = results[n_steps]
        print(f"{n_steps:<10} {res['mean_error']:<12.4f} {res['std_error']:<12.4f} {res['median_error']:<12.4f}")

    print("\n추론 속도 (Inference Speed):")
    print("-" * 70)
    print(f"{'Steps':<10} {'Time (ms)':<12}")
    print("-" * 70)

    for n_steps in inference_steps_list:
        print(f"{n_steps:<10} {speed_results[n_steps]:<12.2f}")

    # ============================================================================
    # 시각화
    # ============================================================================
    print("\n시각화 생성 중...")

    # 1. Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error comparison
    ax = axes[0]
    means = [results[n]['mean_error'] for n in inference_steps_list]
    stds = [results[n]['std_error'] for n in inference_steps_list]

    ax.errorbar(inference_steps_list, means, yerr=stds, marker='o', linewidth=2, capsize=5)
    ax.set_xlabel('Inference Steps', fontsize=12)
    ax.set_ylabel('Position Error', fontsize=12)
    ax.set_title('Position Error vs Inference Steps', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Speed comparison
    ax = axes[1]
    speeds = [speed_results[n] for n in inference_steps_list]

    ax.plot(inference_steps_list, speeds, marker='s', linewidth=2, color='orange')
    ax.set_xlabel('Inference Steps', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Inference Speed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    plt.savefig(results_dir / 'flow_matching_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"  저장: {results_dir / 'flow_matching_evaluation.png'}")

    # 2. Predicted vs Actual positions (best model: 10 steps)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    best_n_steps = 10
    predictions = results[best_n_steps]['predictions'][:500]
    targets = results[best_n_steps]['targets'][:500]

    ax.scatter(targets[:, 0], targets[:, 1], alpha=0.3, s=10, c='blue', label='Actual')
    ax.scatter(predictions[:, 0], predictions[:, 1], alpha=0.3, s=10, c='red', label='Predicted')

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Predicted vs Actual Positions ({best_n_steps} steps)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(results_dir / 'flow_matching_positions.png', dpi=150, bbox_inches='tight')
    print(f"  저장: {results_dir / 'flow_matching_positions.png'}")

    # ============================================================================
    # 결과 저장
    # ============================================================================
    print("\n결과 저장 중...")

    np.save(results_dir / 'flow_matching_results.npy', results)

    # Summary
    summary = {
        'test_samples': len(test_dataset),
        'inference_steps': inference_steps_list,
        'mean_errors': [results[n]['mean_error'] for n in inference_steps_list],
        'std_errors': [results[n]['std_error'] for n in inference_steps_list],
        'median_errors': [results[n]['median_error'] for n in inference_steps_list],
        'inference_speeds': [speed_results[n] for n in inference_steps_list],
    }

    with open(results_dir / 'flow_matching_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Flow Matching Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Test Samples: {summary['test_samples']}\n\n")

        f.write("Position Error:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Steps':<10} {'Mean':<12} {'Std':<12} {'Median':<12}\n")
        f.write("-" * 70 + "\n")

        for i, n_steps in enumerate(inference_steps_list):
            f.write(f"{n_steps:<10} {summary['mean_errors'][i]:<12.4f} "
                   f"{summary['std_errors'][i]:<12.4f} {summary['median_errors'][i]:<12.4f}\n")

        f.write("\nInference Speed:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Steps':<10} {'Time (ms)':<12}\n")
        f.write("-" * 70 + "\n")

        for i, n_steps in enumerate(inference_steps_list):
            f.write(f"{n_steps:<10} {summary['inference_speeds'][i]:<12.2f}\n")

    print(f"  저장: {results_dir / 'flow_matching_summary.txt'}")

    print("\n" + "=" * 70)
    print("✅ 평가 완료!")
    print("=" * 70)
    print(f"\n결과 파일:")
    print(f"  - {results_dir / 'flow_matching_evaluation.png'}")
    print(f"  - {results_dir / 'flow_matching_positions.png'}")
    print(f"  - {results_dir / 'flow_matching_summary.txt'}")
    print(f"  - {results_dir / 'flow_matching_results.npy'}")

    # 최적 성능 하이라이트
    best_n_steps = min(results.keys(), key=lambda n: results[n]['mean_error'])
    print(f"\n🎯 최적 성능:")
    print(f"  Steps: {best_n_steps}")
    print(f"  Mean Error: {results[best_n_steps]['mean_error']:.4f}")
    print(f"  Inference Speed: {speed_results[best_n_steps]:.2f} ms/batch")

if __name__ == '__main__':
    evaluate()
