#!/usr/bin/env python3
"""Sliding Window 방식 학습 - Causal Training"""
import json
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 역정규화
COORD_CENTER = (-41.0, 0.0)
COORD_SCALE = 50.0

def denormalize_coord(x_norm: float, y_norm: float):
    x = x_norm * COORD_SCALE + COORD_CENTER[0]
    y = y_norm * COORD_SCALE + COORD_CENTER[1]
    return (x, y)

class SlidingWindowDataset(Dataset):
    """Sliding Window 데이터셋

    각 샘플: {"features": [250, n_features], "target": [x, y]}
    """
    def __init__(self, jsonl_path: Path):
        self.samples = []

        with jsonl_path.open() as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        if self.samples:
            self.n_features = len(self.samples[0]["features"][0])
            self.window_size = len(self.samples[0]["features"])
        else:
            self.n_features = 0
            self.window_size = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        features = torch.tensor(sample["features"], dtype=torch.float32)  # [250, n_features]
        target = torch.tensor(sample["target"], dtype=torch.float32)  # [2]

        return features, target

# Hyena 모델 import (기존 것 사용)
import sys
sys.path.append(str(Path(__file__).parent))
from model import HyenaPositioning

def train_sliding(
    data_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 2e-4,
    hidden_dim: int = 256,
    depth: int = 8,
    dropout: float = 0.1,
    patience: int = 10,
    checkpoint_dir: Path = Path("checkpoints_sliding"),
    device: str = "cuda",
):
    """Sliding Window 학습"""

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    meta_path = data_dir / "meta.json"

    # 메타데이터 로드
    with meta_path.open() as f:
        meta = json.load(f)

    n_features = meta["n_features"]
    window_size = meta["window_size"]

    print("=" * 80)
    print("🚀 Sliding Window 학습 시작")
    print("=" * 80)
    print(f"  Features: {n_features}")
    print(f"  Window size: {window_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Depth: {depth}")
    print()

    # Dataset
    train_ds = SlidingWindowDataset(train_path)
    val_ds = SlidingWindowDataset(val_path)
    test_ds = SlidingWindowDataset(test_path)

    print(f"📊 데이터 로드:")
    print(f"  Train: {len(train_ds)}개 샘플")
    print(f"  Val:   {len(val_ds)}개 샘플")
    print(f"  Test:  {len(test_ds)}개 샘플")
    print()

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = HyenaPositioning(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        output_dim=2,  # (x, y)
        depth=depth,
        dropout=dropout,
        num_edge_types=1,  # Sliding window에서는 edge 정보 없음
    ).to(device)

    print(f"🧠 모델: Hyena Sliding Window")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # Training
    best_val_rmse = float("inf")
    no_improve = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"

    print("🚀 학습 시작\n")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_distances = []

        for features, targets in train_loader:
            features = features.to(device)  # [batch, 250, n_features]
            targets = targets.to(device)  # [batch, 2]

            optimizer.zero_grad()

            # Hyena expects [batch, seq_len, features]
            # Edge embedding 없이 사용 (edge_ids=None 또는 0)
            edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)

            outputs = model(features, edge_ids)  # [batch, 250, 2]

            # 마지막 타임스텝만 사용
            pred = outputs[:, -1, :]  # [batch, 2]

            loss = criterion(pred, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * features.size(0)

            # 거리 계산 (역정규화)
            pred_np = pred.detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()

            for i in range(len(pred_np)):
                pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])
                dist = math.hypot(pred_pos[0] - target_pos[0], pred_pos[1] - target_pos[1])
                train_distances.append(dist)

        train_loss /= len(train_ds)
        train_rmse = np.sqrt(np.mean(np.array(train_distances) ** 2))
        train_mae = np.mean(train_distances)

        # Validation
        model.eval()
        val_loss = 0.0
        val_distances = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
                outputs = model(features, edge_ids)
                pred = outputs[:, -1, :]

                loss = criterion(pred, targets)
                val_loss += loss.item() * features.size(0)

                pred_np = pred.cpu().numpy()
                target_np = targets.cpu().numpy()

                for i in range(len(pred_np)):
                    pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                    target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])
                    dist = math.hypot(pred_pos[0] - target_pos[0], pred_pos[1] - target_pos[1])
                    val_distances.append(dist)

        val_loss /= len(val_ds)
        val_rmse = np.sqrt(np.mean(np.array(val_distances) ** 2))
        val_mae = np.mean(val_distances)
        val_median = np.median(val_distances)
        val_p90 = np.percentile(val_distances, 90)

        scheduler.step()

        print(
            f"[Epoch {epoch:03d}] "
            f"TrainLoss={train_loss:.4f} TrainRMSE={train_rmse:.3f}m | "
            f"ValRMSE={val_rmse:.3f}m MAE={val_mae:.3f}m "
            f"Median={val_median:.3f}m P90={val_p90:.3f}m"
        )

        # Early stopping
        if val_rmse < best_val_rmse - 0.01:
            best_val_rmse = val_rmse
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_rmse": val_rmse,
                    "meta": meta,
                },
                best_path,
            )
            print(f"   💾 Best model saved (RMSE={best_val_rmse:.3f}m)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

    print(f"\n✅ 학습 완료. Best checkpoint: {best_path}\n")

    # Test
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    test_distances = []

    print("📈 Test 평가 중...")

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)

            edge_ids = torch.zeros(features.size(0), dtype=torch.long, device=device)
            outputs = model(features, edge_ids)
            pred = outputs[:, -1, :]

            pred_np = pred.cpu().numpy()
            target_np = targets.cpu().numpy()

            for i in range(len(pred_np)):
                pred_pos = denormalize_coord(pred_np[i, 0], pred_np[i, 1])
                target_pos = denormalize_coord(target_np[i, 0], target_np[i, 1])
                dist = math.hypot(pred_pos[0] - target_pos[0], pred_pos[1] - target_pos[1])
                test_distances.append(dist)

    test_rmse = np.sqrt(np.mean(np.array(test_distances) ** 2))
    test_mae = np.mean(test_distances)
    test_median = np.median(test_distances)
    test_p90 = np.percentile(test_distances, 90)

    print(
        f"\n[Test Results]\n"
        f"  RMSE:   {test_rmse:.3f}m\n"
        f"  MAE:    {test_mae:.3f}m\n"
        f"  Median: {test_median:.3f}m\n"
        f"  P90:    {test_p90:.3f}m\n"
    )

    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sliding")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints_sliding")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    train_sliding(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        patience=args.patience,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=device,
    )
