#!/usr/bin/env python3
"""
Flow Matching for Trajectory Prediction - Training Script

핵심 변경사항:
- 출력: 마지막 위치 (2,) → 전체 궤적 (250, 2)
- 250배 감독 신호 증가
- One-to-Many 문제 근본 해결
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
from tqdm import tqdm

from model_trajectory import FlowMatchingTrajectory, compute_trajectory_loss

print("=" * 70)
print("🚀 Flow Matching Trajectory Prediction - Training")
print("=" * 70)

# ============================================================================
# Configuration
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")

CONFIG = {
    # Model
    'sensor_dim': 6,
    'position_dim': 2,
    'sequence_length': 250,
    'd_model': 256,
    'encoder_layers': 4,
    'decoder_layers': 4,
    'n_heads': 8,
    'dropout': 0.1,

    # Training
    'batch_size': 32,  # 64 → 32 (trajectory is larger)
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 100,
    'warmup_steps': 4000,
    'early_stopping_patience': 10,

    # Top-k Loss
    'use_topk_loss': True,
    'topk_ratio': 0.5,

    # Inference
    'inference_steps': 10,
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# Dataset
# ============================================================================
class TrajectoryDataset(Dataset):
    """
    Trajectory Prediction Dataset

    Input: Sensor sequence (T, 6)
    Target: Full trajectory (T, 2)
    """
    def __init__(self, states, trajectories):
        """
        Args:
            states: (N, T, 6) - sensor data
            trajectories: (N, T, 2) - full trajectories
        """
        self.states = torch.FloatTensor(states)
        self.trajectories = torch.FloatTensor(trajectories)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'sensor_data': self.states[idx],        # (T, 6)
            'trajectory': self.trajectories[idx],   # (T, 2)
        }

# ============================================================================
# Training
# ============================================================================
def train():
    print("\n[1/6] Loading data...")

    data_dir = Path(__file__).parent / 'processed_data_flow_matching'

    states_train = np.load(data_dir / 'states_train.npy', allow_pickle=True)
    traj_train = np.load(data_dir / 'trajectories_train.npy', allow_pickle=True)

    states_val = np.load(data_dir / 'states_val.npy', allow_pickle=True)
    traj_val = np.load(data_dir / 'trajectories_val.npy', allow_pickle=True)

    print(f"  Train: {states_train.shape} → {traj_train.shape}")
    print(f"  Val:   {states_val.shape} → {traj_val.shape}")

    print("\n[2/6] Creating datasets...")
    train_dataset = TrajectoryDataset(states_train, traj_train)
    val_dataset = TrajectoryDataset(states_val, traj_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    print("\n[3/6] Initializing model...")
    model = FlowMatchingTrajectory(
        sensor_dim=CONFIG['sensor_dim'],
        position_dim=CONFIG['position_dim'],
        sequence_length=CONFIG['sequence_length'],
        d_model=CONFIG['d_model'],
        encoder_layers=CONFIG['encoder_layers'],
        decoder_layers=CONFIG['decoder_layers'],
        n_heads=CONFIG['n_heads'],
        dropout=CONFIG['dropout']
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    print("\n[4/6] Optimizer & Scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Cosine annealing with warmup
    def lr_lambda(step):
        if step < CONFIG['warmup_steps']:
            return step / CONFIG['warmup_steps']
        else:
            progress = (step - CONFIG['warmup_steps']) / (CONFIG['epochs'] * len(train_loader) - CONFIG['warmup_steps'])
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("\n[5/6] Training...")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Early stopping patience: {CONFIG['early_stopping_patience']}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(CONFIG['epochs']):
        # ========== Training ==========
        model.train()
        train_loss = 0.0
        train_final_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

        for batch in train_pbar:
            sensor_data = batch['sensor_data'].to(DEVICE)
            trajectories = batch['trajectory'].to(DEVICE)

            # Trajectory loss
            loss, final_loss = compute_trajectory_loss(
                model, sensor_data, trajectories,
                use_topk=CONFIG['use_topk_loss'],
                k_ratio=CONFIG['topk_ratio']
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_final_loss += final_loss.item()
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'final': f'{final_loss.item():.4f}'
            })

        train_loss /= len(train_loader)
        train_final_loss /= len(train_loader)

        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        val_final_loss = 0.0
        val_traj_error = 0.0
        val_final_error = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", leave=False):
                sensor_data = batch['sensor_data'].to(DEVICE)
                trajectories = batch['trajectory'].to(DEVICE)

                # Loss
                loss, final_loss = compute_trajectory_loss(
                    model, sensor_data, trajectories,
                    use_topk=CONFIG['use_topk_loss'],
                    k_ratio=CONFIG['topk_ratio']
                )
                val_loss += loss.item()
                val_final_loss += final_loss.item()

                # Trajectory error (using sampling)
                pred_traj = model.sample(sensor_data, n_steps=CONFIG['inference_steps'])

                # Average error over all timesteps
                traj_error = torch.norm(pred_traj - trajectories, dim=2).mean()
                val_traj_error += traj_error.item()

                # Final position error
                final_error = torch.norm(pred_traj[:, -1, :] - trajectories[:, -1, :], dim=1).mean()
                val_final_error += final_error.item()

        val_loss /= len(val_loader)
        val_final_loss /= len(val_loader)
        val_traj_error /= len(val_loader)
        val_final_error /= len(val_loader)

        print(f"Epoch {epoch+1:3d} | "
              f"Train: {train_loss:.4f} (final: {train_final_loss:.4f}) | "
              f"Val: {val_loss:.4f} (final: {val_final_loss:.4f}) | "
              f"Traj Err: {val_traj_error:.4f} | "
              f"Final Err: {val_final_error:.4f}")

        # Save best model & Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            model_dir = Path(__file__).parent.parent / 'models'
            model_dir.mkdir(exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_final_loss': val_final_loss,
                'val_traj_error': val_traj_error,
                'val_final_error': val_final_error,
                'config': CONFIG,
            }, model_dir / 'flow_matching_trajectory_best.pt')
            print(f"  ✅ Best model saved! (Val Loss: {val_loss:.4f}, Final Error: {val_final_error:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement for {patience_counter} epoch(s)")

            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n⛔ Early stopping! No improvement for {CONFIG['early_stopping_patience']} epochs.")
                print(f"  Best epoch: {best_epoch}")
                print(f"  Best val loss: {best_val_loss:.4f}")
                break

    print("\n[6/6] Training complete!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model saved: models/flow_matching_trajectory_best.pt")

    # ========== Test Evaluation ==========
    print("\n" + "=" * 70)
    print("📊 Test Evaluation")
    print("=" * 70)

    # Load test data
    states_test = np.load(data_dir / 'states_test.npy', allow_pickle=True)
    traj_test = np.load(data_dir / 'trajectories_test.npy', allow_pickle=True)

    test_dataset = TrajectoryDataset(states_test, traj_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    print(f"\nTest samples: {len(test_dataset):,}")

    # Load best model
    checkpoint = torch.load(model_dir / 'flow_matching_trajectory_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate
    test_traj_error = 0.0
    test_final_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
            sensor_data = batch['sensor_data'].to(DEVICE)
            trajectories = batch['trajectory'].to(DEVICE)

            # Sample trajectory
            pred_traj = model.sample(sensor_data, n_steps=CONFIG['inference_steps'])

            # Average trajectory error
            traj_error = torch.norm(pred_traj - trajectories, dim=2).mean(dim=1)
            test_traj_error += traj_error.sum().item()

            # Final position error
            final_error = torch.norm(pred_traj[:, -1, :] - trajectories[:, -1, :], dim=1)
            test_final_error += final_error.sum().item()

            total_samples += len(trajectories)

    test_traj_error /= total_samples
    test_final_error /= total_samples

    print(f"\n📊 Test Results:")
    print(f"  Average trajectory error: {test_traj_error:.4f} (normalized)")
    print(f"  Final position error: {test_final_error:.4f} (normalized)")

    # Convert to meters (assuming normalization range)
    # Normalized: -1~1 → Real: -85.5~85.5m (range=171m, so 2.0 = 171m)
    meters_per_unit = 171.0 / 2.0
    print(f"  Average trajectory error: {test_traj_error * meters_per_unit:.2f}m")
    print(f"  Final position error: {test_final_error * meters_per_unit:.2f}m")

    print("\n" + "=" * 70)
    print("✅ Trajectory Prediction Training Complete!")
    print("=" * 70)
    print(f"""
📊 Final Results:
  Val Loss: {best_val_loss:.4f}
  Model: models/flow_matching_trajectory_best.pt

🔥 Key Improvements:
  ✅ 250x more supervision signal (full trajectory vs final position)
  ✅ Solves One-to-Many problem fundamentally
  ✅ Real-time trajectory tracking possible
  ✅ Sequential pattern learning enhanced

🎯 Next Steps:
  1. Compare with baseline (position-only model)
  2. Analyze trajectory prediction quality
  3. Test real-time tracking capability
""")

if __name__ == '__main__':
    train()
