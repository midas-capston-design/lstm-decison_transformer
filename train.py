import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from data_loader import get_dataloaders
from hyena_positioning_model import create_model


class PositionLoss(nn.Module):
    """
    Combined loss for position estimation
    - MSE for position coordinates
    - Optional: Euclidean distance loss
    """
    def __init__(self, use_euclidean=True, euclidean_weight=0.5):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.use_euclidean = use_euclidean
        self.euclidean_weight = euclidean_weight

    def forward(self, pred, target):
        # MSE loss
        mse = self.mse_loss(pred, target)

        if self.use_euclidean:
            # Euclidean distance loss
            euclidean = torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean()
            return mse + self.euclidean_weight * euclidean
        else:
            return mse


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_error = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            error = torch.sqrt(torch.sum((outputs - targets) ** 2, dim=1)).mean()

        total_loss += loss.item()
        total_error += error.item()

        pbar.set_postfix({
            'loss': loss.item(),
            'error': error.item()
        })

    avg_loss = total_loss / len(train_loader)
    avg_error = total_error / len(train_loader)

    return avg_loss, avg_error


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_error = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            error = torch.sqrt(torch.sum((outputs - targets) ** 2, dim=1)).mean()

            total_loss += loss.item()
            total_error += error.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    avg_error = total_error / len(val_loader)

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    return avg_loss, avg_error, all_predictions, all_targets


def plot_results(predictions, targets, save_path):
    """Plot predicted vs actual positions"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: X coordinate
    axes[0].scatter(targets[:, 0], predictions[:, 0], alpha=0.5, s=10)
    axes[0].plot([targets[:, 0].min(), targets[:, 0].max()],
                 [targets[:, 0].min(), targets[:, 0].max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual X')
    axes[0].set_ylabel('Predicted X')
    axes[0].set_title('X Coordinate Prediction')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Y coordinate
    axes[1].scatter(targets[:, 1], predictions[:, 1], alpha=0.5, s=10)
    axes[1].plot([targets[:, 1].min(), targets[:, 1].max()],
                 [targets[:, 1].min(), targets[:, 1].max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Y')
    axes[1].set_ylabel('Predicted Y')
    axes[1].set_title('Y Coordinate Prediction')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: 2D trajectory
    axes[2].scatter(targets[:, 0], targets[:, 1], alpha=0.5, s=10, label='Actual', c='blue')
    axes[2].scatter(predictions[:, 0], predictions[:, 1], alpha=0.5, s=10, label='Predicted', c='red')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title('2D Position Prediction')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"hyena_{args.model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.seed,
        num_workers=args.num_workers
    )

    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type=args.model_type,
        input_dim=6,
        d_model=args.d_model,
        num_layers=args.num_layers,
        seq_len=args.window_size,
        order=args.order,
        num_heads=args.num_heads,
        dropout=args.dropout,
        ff_mult=args.ff_mult
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = PositionLoss(use_euclidean=True, euclidean_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    else:
        scheduler = None

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None

    # Training loop
    print("\nStarting training...")
    best_val_error = float('inf')
    train_losses = []
    train_errors = []
    val_losses = []
    val_errors = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_error = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        train_losses.append(train_loss)
        train_errors.append(train_error)

        # Validate
        val_loss, val_error, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_errors.append(val_error)

        print(f"Train Loss: {train_loss:.4f}, Train Error: {train_error:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Error: {val_error:.4f}")

        # Update learning rate
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': val_error,
                'args': args
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved best model with error: {val_error:.4f}")

            # Plot results for best model
            plot_results(val_preds, val_targets, os.path.join(output_dir, 'best_predictions.png'))

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': val_error,
                'args': args
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_errors, label='Train Error')
    axes[1].plot(val_errors, label='Val Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Position Error (m)')
    axes[1].set_title('Training and Validation Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nTraining completed!")
    print(f"Best validation error: {best_val_error:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hyena-based Indoor Positioning Model')

    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed_data',
                        help='Directory containing CSV files')
    parser.add_argument('--window_size', type=int, default=250,
                        help='Window size for sequence')
    parser.add_argument('--stride', type=int, default=50,
                        help='Stride for sliding window')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size')

    # Model
    parser.add_argument('--model_type', type=str, default='v1', choices=['v1', 'v2'],
                        help='Model version')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of Hyena blocks')
    parser.add_argument('--order', type=int, default=2,
                        help='Order of Hyena operator')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--ff_mult', type=int, default=4,
                        help='Feedforward multiplier')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    main(args)
