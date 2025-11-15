import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class IndoorPositioningDataset(Dataset):
    """
    Dataset for indoor positioning using magnetic and orientation data

    Data format:
    - Input: (window_size, 6) - [MagX, MagY, MagZ, Pitch, Roll, Yaw]
    - Output: (2,) - [x, y] position
    """
    def __init__(self, data_dir, window_size=250, stride=50, mode='train',
                 test_size=0.2, random_state=42, normalize=True):
        """
        Args:
            data_dir: Directory containing CSV files
            window_size: Size of sliding window
            stride: Stride for sliding window
            mode: 'train' or 'test'
            test_size: Proportion of data for testing
            random_state: Random seed
            normalize: Whether to normalize the features
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        self.normalize = normalize

        # Load and process all CSV files
        self.windows, self.targets = self._load_data()

        # Split into train/test
        if len(self.windows) > 0:
            train_windows, test_windows, train_targets, test_targets = train_test_split(
                self.windows, self.targets, test_size=test_size, random_state=random_state
            )

            if mode == 'train':
                self.windows = train_windows
                self.targets = train_targets
            else:
                self.windows = test_windows
                self.targets = test_targets

        # Normalize features
        if self.normalize and len(self.windows) > 0:
            self._normalize_features()

        print(f"{mode.upper()} dataset: {len(self.windows)} samples")

    def _load_data(self):
        """Load all CSV files and create sliding windows"""
        all_windows = []
        all_targets = []

        # Find all CSV files
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        print(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Extract features and targets
                # Features: MagX, MagY, MagZ, Pitch, Roll, Yaw
                features = df[['MagX', 'MagY', 'MagZ', 'Pitch', 'Roll', 'Yaw']].values

                # Targets: x, y positions
                positions = df[['x', 'y']].values

                # Create sliding windows
                for i in range(0, len(features) - self.window_size + 1, self.stride):
                    window = features[i:i + self.window_size]
                    # Use the last position in the window as target
                    target = positions[i + self.window_size - 1]

                    all_windows.append(window)
                    all_targets.append(target)

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue

        return np.array(all_windows), np.array(all_targets)

    def _normalize_features(self):
        """Normalize features using StandardScaler"""
        # Reshape for normalization
        original_shape = self.windows.shape
        windows_reshaped = self.windows.reshape(-1, original_shape[-1])

        # Fit scaler on training data
        if self.mode == 'train':
            self.scaler = StandardScaler()
            windows_normalized = self.scaler.fit_transform(windows_reshaped)
        else:
            # Use the same scaler fitted on training data
            windows_normalized = windows_reshaped

        # Reshape back
        self.windows = windows_normalized.reshape(original_shape)

    def get_scaler(self):
        """Return the scaler for use in test set"""
        if hasattr(self, 'scaler'):
            return self.scaler
        return None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx])
        target = torch.FloatTensor(self.targets[idx])
        return window, target


def get_dataloaders(data_dir, window_size=250, stride=50, batch_size=32,
                   test_size=0.2, random_state=42, num_workers=4):
    """
    Create train and test dataloaders

    Args:
        data_dir: Directory containing CSV files
        window_size: Size of sliding window
        stride: Stride for sliding window
        batch_size: Batch size for training
        test_size: Proportion of data for testing
        random_state: Random seed
        num_workers: Number of workers for data loading

    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = IndoorPositioningDataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        mode='train',
        test_size=test_size,
        random_state=random_state,
        normalize=True
    )

    test_dataset = IndoorPositioningDataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        mode='test',
        test_size=test_size,
        random_state=random_state,
        normalize=True
    )

    # Apply same scaler to test set
    if train_dataset.get_scaler() is not None:
        test_dataset.scaler = train_dataset.get_scaler()
        original_shape = test_dataset.windows.shape
        windows_reshaped = test_dataset.windows.reshape(-1, original_shape[-1])
        windows_normalized = test_dataset.scaler.transform(windows_reshaped)
        test_dataset.windows = windows_normalized.reshape(original_shape)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    data_dir = "data/processed_data"

    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        window_size=250,
        stride=50,
        batch_size=32
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Test one batch
    for batch_idx, (windows, targets) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Windows shape: {windows.shape}")  # (batch, 250, 6)
        print(f"  Targets shape: {targets.shape}")  # (batch, 2)
        print(f"  Sample target: {targets[0]}")
        break
