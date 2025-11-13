#!/usr/bin/env python3
"""
Hyena for Indoor Positioning - Training Script

핵심:
- Long Convolution으로 전역 패턴 학습
- 모든 timestep을 동등하게 처리
- 지자기 positioning 첫 적용
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
from tqdm import tqdm

from model import HyenaLocalization

print("=" * 70)
print("🚀 Hyena for Indoor Positioning - Training")
print("=" * 70)

# ============================================================================
# Configuration
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {DEVICE}")

CONFIG = {
    # Model
    'input_dim': 6,
    'seq_len': 250,  # 250 샘플 (5초 @ 50Hz)
    'd_model': 256,
    'n_layers': 4,
    'order': 2,
    'filter_order': 64,
    'dropout': 0.1,

    # Training
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'warmup_steps': 2000,
}

print("\n설정:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# Dataset
# ============================================================================
class HyenaDataset(Dataset):
    """
    Hyena용 데이터셋

    Input: 센서 시퀀스 (250, 6)
    Target: 위치 (2,)
    """
    def __init__(self, states, positions):
        """
        Args:
            states: (N, 250, 6) - 센서 데이터
            positions: (N, 2) - 위치
        """
        self.states = torch.FloatTensor(states)
        self.positions = torch.FloatTensor(positions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'sensor_data': self.states[idx],      # (250, 6)
            'position': self.positions[idx],       # (2,)
        }

# ============================================================================
# Training
# ============================================================================
def train():
    print("\n[1/6] 데이터 로드...")

    data_dir = Path(__file__).parent / 'processed_data_hyena'

    states_train = np.load(data_dir / 'states_train.npy')
    positions_train = np.load(data_dir / 'positions_train.npy')

    states_val = np.load(data_dir / 'states_val.npy')
    positions_val = np.load(data_dir / 'positions_val.npy')

    print(f"  Train: {states_train.shape}")
    print(f"  Val:   {states_val.shape}")

    # 시퀀스 길이 업데이트
    actual_seq_len = states_train.shape[1]
    CONFIG['seq_len'] = actual_seq_len
    print(f"  실제 시퀀스 길이: {actual_seq_len}")

    print("\n[2/6] 데이터셋 생성...")
    train_dataset = HyenaDataset(states_train, positions_train)
    val_dataset = HyenaDataset(states_val, positions_val)

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

    print("\n[3/6] 모델 초기화...")
    model = HyenaLocalization(
        input_dim=CONFIG['input_dim'],
        seq_len=CONFIG['seq_len'],
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['n_layers'],
        order=CONFIG['order'],
        filter_order=CONFIG['filter_order'],
        dropout=CONFIG['dropout']
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터: {n_params:,}")

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

    print("\n[5/6] 학습 시작...")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")

    best_val_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        # ========== Training ==========
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

        for batch in train_pbar:
            sensor_data = batch['sensor_data'].to(DEVICE)
            positions = batch['position'].to(DEVICE)

            # Forward
            pred_positions = model(sensor_data)

            # Loss: MSE for position prediction
            loss = F.mse_loss(pred_positions, positions)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        val_position_error = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", leave=False):
                sensor_data = batch['sensor_data'].to(DEVICE)
                positions = batch['position'].to(DEVICE)

                # Predict
                pred_positions = model(sensor_data)

                # Loss
                loss = F.mse_loss(pred_positions, positions)
                val_loss += loss.item()

                # Position error (Euclidean distance)
                error = torch.norm(pred_positions - positions, dim=1).mean()
                val_position_error += error.item()

        val_loss /= len(val_loader)
        val_position_error /= len(val_loader)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Pos Error: {val_position_error:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_dir = Path(__file__).parent.parent / 'models'
            model_dir.mkdir(exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_position_error': val_position_error,
                'config': CONFIG,
            }, model_dir / 'hyena_best.pt')

    print("\n[6/6] 학습 완료!")
    print(f"  최고 Val Loss: {best_val_loss:.4f}")
    print(f"  모델 저장: models/hyena_best.pt")

    print("\n" + "=" * 70)
    print("✅ Hyena 학습 완료!")
    print("=" * 70)
    print(f"""
📊 최종 결과:
  Val Loss: {best_val_loss:.4f}
  모델: models/hyena_best.pt

🔥 독창성:
  ✅ 지자기 기반 인도어 포지셔닝에 Hyena 첫 적용
  ✅ Long Convolution으로 전역 패턴 포착
  ✅ 모든 timestep 동등하게 처리
  ✅ Transformer보다 효율적 (O(N log N))

🎯 다음 단계:
  1. 평가 스크립트로 성능 측정
  2. Flow Matching과 비교
  3. 논문 작성 (완전히 새로운 접근법!)
""")

if __name__ == '__main__':
    train()
