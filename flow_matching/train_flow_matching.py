#!/usr/bin/env python3
"""
Flow Matching for Magnetic Field Indoor Positioning - Training Script

핵심:
- Conditional Flow Matching으로 센서 → 위치 학습
- 1-2 step inference로 실시간 가능
- 완전히 새로운 접근법 (논문 0개)
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
import pickle
from tqdm import tqdm

from model import FlowMatchingLocalization, compute_flow_matching_loss

print("=" * 70)
print("🚀 Flow Matching for Indoor Positioning - Training")
print("=" * 70)

# ============================================================================
# Configuration
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {DEVICE}")

CONFIG = {
    # Model
    'sensor_dim': 6,
    'position_dim': 2,
    'd_model': 384,          # 30x 데이터에 맞춰 모델 확장
    'encoder_layers': 6,
    'velocity_layers': 6,
    'n_heads': 8,
    'dropout': 0.15,

    # Data (5걸음 기준: 250 timesteps)
    'sequence_length': 250,

    # Training
    'batch_size': 128,
    'learning_rate': 5e-5,
    'weight_decay': 1e-4,
    'epochs': 200,
    'warmup_steps': 8000,
    'early_stopping_patience': 15,

    # Top-k Loss (Hard Example Mining)
    'use_topk_loss': True,
    'topk_ratio': 0.5,

    # Data Augmentation (전처리 시 적용됨, Train only)
    'augment_train': False,  # 전처리에서 이미 적용됨
    'mag_noise_std': 0.8,      # (사용 안 함)
    'orient_noise_std': 1.5,   # (사용 안 함)

    # Inference
    'inference_steps': 10,  # Can use 1-2 for real-time
    'topk_samples': 10,  # Top-k sampling: 총 샘플 수
    'topk_k': 5,  # Top-k sampling: 선택할 개수 (10개 중 상위 5개)
}

print("\n설정:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# Dataset with Real-time Augmentation
# ============================================================================
def augment_sensor_data(sensor_data, mag_noise_std=0.8, orient_noise_std=1.5):
    """
    센서 데이터 실시간 증강 (Train only)

    시퀀셜 데이터 특성을 고려한 증강:
    1. Drift (전체 시퀀스 바이어스) - 시간 불변
    2. Smooth noise (시간적으로 연속적인 노이즈)

    증강 방법:
    1. 지자기 센서 노이즈 (MagX, MagY, MagZ)
       - 70% drift: 전체 시퀀스에 동일한 바이어스 (센서 캘리브레이션 오차)
       - 30% smooth noise: 시간적으로 연속적인 노이즈 (측정 오차)

    2. 방향 센서 노이즈 (Pitch, Roll, Yaw)
       - 70% drift: 전체 시퀀스에 동일한 바이어스 (자세 추정 오차)
       - 30% smooth noise: 시간적으로 연속적인 노이즈 (각속도 누적 오차)

    증강 비율:
    - Train: 매 epoch마다 100% 샘플에 실시간 증강 적용
    - Val/Test: 증강 없음 (원본 데이터만)

    Args:
        sensor_data: (100, 6) - [MagX, MagY, MagZ, Pitch, Roll, Yaw]
        mag_noise_std: 지자기 노이즈 표준편차 (μT)
        orient_noise_std: 방향 노이즈 표준편차 (도)

    Returns:
        augmented sensor_data: (100, 6)
    """
    sensor_data = sensor_data.clone()
    T = sensor_data.shape[0]

    # 지자기 센서 노이즈 (MagX, MagY, MagZ)
    # 70% drift + 30% smooth noise
    mag_drift = torch.randn(3) * mag_noise_std * 0.7  # (3,)
    mag_smooth = torch.randn(T, 3) * mag_noise_std * 0.3  # (T, 3)
    sensor_data[:, 0:3] += mag_drift + mag_smooth

    # 방향 센서 노이즈 (Pitch, Roll, Yaw)
    # 70% drift + 30% smooth noise
    orient_drift = torch.randn(3) * orient_noise_std * 0.7  # (3,)
    orient_smooth = torch.randn(T, 3) * orient_noise_std * 0.3  # (T, 3)
    sensor_data[:, 3:6] += orient_drift + orient_smooth

    return sensor_data


class FlowMatchingDataset(Dataset):
    """
    Flow Matching용 데이터셋 (실시간 증강 지원)

    Input: 센서 시퀀스 (T, 6) - 기본 T=250
    Target: 마지막 위치 (2,)
    """
    def __init__(self, states, trajectories, augment=False,
                 mag_noise_std=0.8, orient_noise_std=1.5):
        """
        Args:
            states: (N, 100, 6) - 센서 데이터
            trajectories: (N, 100, 2) - 각 timestep의 위치
            augment: Train 시에만 True
            mag_noise_std: 지자기 노이즈 표준편차
            orient_noise_std: 방향 노이즈 표준편차
        """
        self.states = torch.FloatTensor(states)
        self.positions = torch.FloatTensor(trajectories[:, -1, :])  # 마지막 위치만
        self.augment = augment
        self.mag_noise_std = mag_noise_std
        self.orient_noise_std = orient_noise_std

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        sensor_data = self.states[idx]  # (100, 6)

        # Train 시에만 증강 적용
        if self.augment:
            sensor_data = augment_sensor_data(
                sensor_data,
                self.mag_noise_std,
                self.orient_noise_std
            )

        return {
            'sensor_data': sensor_data,           # (T, 6)
            'position': self.positions[idx],       # (2,)
        }


# ============================================================================
# Utility helpers
# ============================================================================
def load_metadata(data_dir: Path):
    """Load metadata.pkl if available"""
    metadata_path = data_dir / 'metadata.pkl'
    if not metadata_path.exists():
        print("  ⚠️ metadata.pkl을 찾을 수 없습니다. 정규화 좌표로 평가합니다.")
        return None

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata


def extract_position_bounds(metadata):
    """
    metadata에서 x/y 최소/최대 범위를 추출한다.
    새로운 전처리 스크립트는 position_bounds를 제공하고,
    기존 스크립트는 normalization에 해당 정보가 포함되어 있다.
    """
    if metadata is None:
        return None

    if 'position_bounds' in metadata:
        return metadata['position_bounds']

    norm = metadata.get('normalization')
    if norm and all(k in norm for k in ('x_min', 'x_max', 'y_min', 'y_max')):
        return {
            'x_min': norm['x_min'],
            'x_max': norm['x_max'],
            'y_min': norm['y_min'],
            'y_max': norm['y_max'],
        }
    return None


def denormalize_positions_tensor(pos_tensor, bounds):
    """(-1, 1) 정규화 좌표를 실제 (x, y)로 되돌린다."""
    x = (pos_tensor[..., 0] + 1.0) * 0.5 * (bounds['x_max'] - bounds['x_min']) + bounds['x_min']
    y = (pos_tensor[..., 1] + 1.0) * 0.5 * (bounds['y_max'] - bounds['y_min']) + bounds['y_min']
    return torch.stack([x, y], dim=-1)


def maybe_denormalize(pos_tensor, bounds):
    """bounds가 있을 때만 denormalize"""
    if bounds is None:
        return pos_tensor
    return denormalize_positions_tensor(pos_tensor, bounds)

# ============================================================================
# Training
# ============================================================================
def train():
    print("\n[1/6] 데이터 로드...")

    # 데이터 디렉토리 자동 선택 (합성 데이터 우선)
    base_dir = Path(__file__).parent
    synth_dir = base_dir / 'processed_data_flow_matching_synth'
    default_dir = base_dir / 'processed_data_flow_matching'

    if synth_dir.exists():
        data_dir = synth_dir
        print("  📦 Using synthetic dataset: processed_data_flow_matching_synth")
    else:
        data_dir = default_dir
        print("  📦 Using default dataset: processed_data_flow_matching")
    metadata = load_metadata(data_dir)
    position_bounds = extract_position_bounds(metadata)
    grid_threshold = metadata.get('grid_size') if metadata else None
    grid_metrics_enabled = position_bounds is not None and grid_threshold is not None
    unit_label = "m" if position_bounds is not None else "normalized units"

    states_train = np.load(data_dir / 'states_train.npy', allow_pickle=True)
    traj_train = np.load(data_dir / 'trajectories_train.npy', allow_pickle=True)

    states_val = np.load(data_dir / 'states_val.npy', allow_pickle=True)
    traj_val = np.load(data_dir / 'trajectories_val.npy', allow_pickle=True)

    print(f"  Train: {states_train.shape}")
    print(f"  Val:   {states_val.shape}")

    print("\n[2/6] 데이터셋 생성...")
    # Train: 실시간 증강 ON
    train_dataset = FlowMatchingDataset(
        states_train, traj_train,
        augment=CONFIG['augment_train'],
        mag_noise_std=CONFIG['mag_noise_std'],
        orient_noise_std=CONFIG['orient_noise_std']
    )
    # Val: 증강 OFF (원본 데이터만)
    val_dataset = FlowMatchingDataset(
        states_val, traj_val,
        augment=False
    )

    print(f"  ✅ Train: 실시간 증강 {'활성화' if CONFIG['augment_train'] else '비활성화'}")
    if CONFIG['augment_train']:
        print(f"     - 지자기 노이즈: std={CONFIG['mag_noise_std']}μT")
        print(f"     - 방향 노이즈: std={CONFIG['orient_noise_std']}°")
        print(f"     - 증강 비율: 매 epoch 100% 샘플")
    print(f"  ✅ Val: 원본 데이터만 사용")

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
    model = FlowMatchingLocalization(
        sensor_dim=CONFIG['sensor_dim'],
        position_dim=CONFIG['position_dim'],
        d_model=CONFIG['d_model'],
        encoder_layers=CONFIG['encoder_layers'],
        velocity_layers=CONFIG['velocity_layers'],
        n_heads=CONFIG['n_heads'],
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
    print(f"  Early stopping patience: {CONFIG['early_stopping_patience']}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(CONFIG['epochs']):
        # ========== Training ==========
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

        for batch in train_pbar:
            sensor_data = batch['sensor_data'].to(DEVICE)
            positions = batch['position'].to(DEVICE)

            # Flow Matching loss (with Top-k)
            loss = compute_flow_matching_loss(
                model, sensor_data, positions,
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
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        val_position_error = 0.0
        val_position_error_topk = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", leave=False):
                sensor_data = batch['sensor_data'].to(DEVICE)
                positions = batch['position'].to(DEVICE)

                # Flow Matching loss
                loss = compute_flow_matching_loss(
                    model, sensor_data, positions,
                    use_topk=CONFIG['use_topk_loss'],
                    k_ratio=CONFIG['topk_ratio']
                )
                val_loss += loss.item()

                # Position error (using sampling)
                pred_positions = model.sample(sensor_data, n_steps=CONFIG['inference_steps'])
                pred_eval = maybe_denormalize(pred_positions, position_bounds)
                target_eval = maybe_denormalize(positions, position_bounds)
                error = torch.norm(pred_eval - target_eval, dim=1).mean()
                val_position_error += error.item()

                # Position error with Top-k sampling
                best_pos, topk_positions, topk_scores = model.sample_topk(
                    sensor_data,
                    n_samples=CONFIG['topk_samples'],
                    k=CONFIG['topk_k'],
                    n_steps=CONFIG['inference_steps']
                )
                # 최고 신뢰도 위치 오차
                best_eval = maybe_denormalize(best_pos, position_bounds)
                error_topk = torch.norm(best_eval - target_eval, dim=1).mean()
                val_position_error_topk += error_topk.item()

        val_loss /= len(val_loader)
        val_position_error /= len(val_loader)
        val_position_error_topk /= len(val_loader)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Pos Error ({unit_label}): {val_position_error:.4f} | "
              f"Val Pos Error (Top-k, {unit_label}): {val_position_error_topk:.4f}")

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
                'val_position_error': val_position_error,
                'val_position_error_topk': val_position_error_topk,
                'config': CONFIG,
            }, model_dir / 'flow_matching_best.pt')
            print(f"  ✅ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement for {patience_counter} epoch(s)")

            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n⛔ Early stopping triggered! No improvement for {CONFIG['early_stopping_patience']} epochs.")
                print(f"  Best epoch: {best_epoch}")
                print(f"  Best val loss: {best_val_loss:.4f}")
                break

    print("\n[6/6] 학습 완료!")
    print(f"  최고 Val Loss: {best_val_loss:.4f}")
    print(f"  모델 저장: models/flow_matching_best.pt")

    # ========== Test Evaluation ==========
    print("\n" + "=" * 70)
    print("📊 Test 데이터 평가")
    print("=" * 70)

    # Test 데이터 로드
    states_test = np.load(data_dir / 'states_test.npy', allow_pickle=True)
    traj_test = np.load(data_dir / 'trajectories_test.npy', allow_pickle=True)

    test_dataset = FlowMatchingDataset(states_test, traj_test, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    print(f"\nTest 샘플: {len(test_dataset):,}개")

    # Best model 로드
    checkpoint = torch.load(model_dir / 'flow_matching_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test 평가
    test_position_error = 0.0
    test_position_error_topk = 0.0
    test_within_1grid = 0
    test_within_1grid_topk = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
            sensor_data = batch['sensor_data'].to(DEVICE)
            positions = batch['position'].to(DEVICE)

            # 일반 sampling
            pred_positions = model.sample(sensor_data, n_steps=CONFIG['inference_steps'])
            pred_eval = maybe_denormalize(pred_positions, position_bounds)
            target_eval = maybe_denormalize(positions, position_bounds)
            error = torch.norm(pred_eval - target_eval, dim=1)
            test_position_error += error.sum().item()

            if grid_metrics_enabled:
                test_within_1grid += (error <= grid_threshold).sum().item()

            # Top-k sampling (5개 후보 모두 평가)
            best_pos, topk_positions, topk_scores = model.sample_topk(
                sensor_data,
                n_samples=CONFIG['topk_samples'],
                k=CONFIG['topk_k'],
                n_steps=CONFIG['inference_steps']
            )

            # 최고 신뢰도 위치 오차
            best_eval = maybe_denormalize(best_pos, position_bounds)
            error_topk = torch.norm(best_eval - target_eval, dim=1)
            test_position_error_topk += error_topk.sum().item()

            if grid_metrics_enabled:
                candidates_eval = maybe_denormalize(topk_positions, position_bounds)
                candidate_errors = torch.norm(candidates_eval - target_eval.unsqueeze(1), dim=2)
                test_within_1grid_topk += (candidate_errors <= grid_threshold).any(dim=1).sum().item()

            total_samples += len(positions)

    test_position_error /= total_samples
    test_position_error_topk /= total_samples
    if grid_metrics_enabled:
        test_acc_1grid = test_within_1grid / total_samples * 100
        test_acc_1grid_topk = test_within_1grid_topk / total_samples * 100

    print(f"\n📊 Test 결과:")
    print(f"  평균 위치 오차 (일반): {test_position_error:.4f} {unit_label}")
    print(f"  평균 위치 오차 (Top-k 최고 신뢰도): {test_position_error_topk:.4f} {unit_label}")

    if grid_metrics_enabled:
        print(f"\n  🎯 1 Grid({grid_threshold:.2f}m) 이내 정확도:")
        print(f"    일반 샘플링 (1개): {test_acc_1grid:.2f}% ({test_within_1grid}/{total_samples})")
        print(f"    Top-5 후보 (5개 중 하나라도): {test_acc_1grid_topk:.2f}% ({test_within_1grid_topk}/{total_samples})")
    else:
        print("\n  🎯 Grid 정확도: metadata 정보가 없어 생략되었습니다.")

    # ========== Test Sampling Speed ==========
    print("\n" + "=" * 70)
    print("⚡ Inference Speed Test")
    print("=" * 70)

    model.eval()
    test_batch = next(iter(val_loader))
    sensor_data = test_batch['sensor_data'][:4].to(DEVICE)

    import time

    for n_steps in [1, 2, 5, 10]:
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model.sample(sensor_data, n_steps=n_steps)
        end = time.time()

        avg_time = (end - start) / 100 * 1000  # ms
        print(f"  {n_steps} steps: {avg_time:.2f} ms/batch (4 samples)")

    print("\n" + "=" * 70)
    print("✅ Flow Matching 학습 완료!")
    print("=" * 70)
    print(f"""
📊 최종 결과:
  Val Loss: {best_val_loss:.4f}
  모델: models/flow_matching_best.pt

🔥 독창성:
  ✅ 지자기 기반 인도어 포지셔닝에 Flow Matching 첫 적용
  ✅ Top-k Loss로 어려운 샘플에 집중 학습
  ✅ Top-k Sampling으로 안정적인 위치 예측
  ✅ 1-2 step inference로 실시간 가능
  ✅ Conditional generation으로 센서 → 위치 매핑

🎯 다음 단계:
  1. 평가 스크립트로 성능 측정
  2. LSTM/Transformer 베이스라인과 비교
  3. 논문 작성 (완전히 새로운 접근법!)
""")

if __name__ == '__main__':
    train()
