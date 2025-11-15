import torch
import torch.nn as nn
from hyena_operator import HyenaBlock, PositionalEmbedding


class HyenaPositioningModel(nn.Module):
    """
    Hyena-based Indoor Positioning Model

    Architecture:
    1. Input embedding: Projects 6D sensor data to d_model dimensions
    2. Positional embedding: Adds positional information
    3. Hyena blocks: Process sequences with subquadratic complexity
    4. Regression head: Maps to (x, y) coordinates
    """
    def __init__(self, input_dim=6, d_model=128, num_layers=4, seq_len=250,
                 order=2, num_heads=1, dropout=0.1, ff_mult=4):
        """
        Args:
            input_dim: Input feature dimension (6 for MagX, MagY, MagZ, Pitch, Roll, Yaw)
            d_model: Model dimension
            num_layers: Number of Hyena blocks
            seq_len: Sequence length (window size)
            order: Order of Hyena operator (higher = more expressive)
            num_heads: Number of attention heads (for multi-head variant)
            dropout: Dropout rate
            ff_mult: Feedforward expansion multiplier
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional embedding
        self.pos_embedding = PositionalEmbedding(d_model, max_len=seq_len)

        # Hyena blocks
        self.hyena_blocks = nn.ModuleList([
            HyenaBlock(d_model, seq_len, order, num_heads, dropout, ff_mult)
            for _ in range(num_layers)
        ])

        # Global pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Regression head for position prediction
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2)  # Output: (x, y)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - Input sensor data

        Returns:
            positions: (batch, 2) - Predicted (x, y) positions
        """
        # Input embedding
        x = self.input_embedding(x)  # (batch, seq_len, d_model)

        # Add positional embedding
        x = self.pos_embedding(x)  # (batch, seq_len, d_model)

        # Apply Hyena blocks
        for hyena_block in self.hyena_blocks:
            x = hyena_block(x)  # (batch, seq_len, d_model)

        # Global pooling to get sequence representation
        # Transpose for pooling: (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        x = self.pooling(x)  # (batch, d_model, 1)
        x = x.squeeze(-1)  # (batch, d_model)

        # Regression to position
        positions = self.regression_head(x)  # (batch, 2)

        return positions

    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HyenaPositioningModelV2(nn.Module):
    """
    Enhanced Hyena-based Indoor Positioning Model with dual-stream architecture

    This version uses:
    1. Separate processing for magnetic and orientation data
    2. Cross-stream fusion
    3. Multi-scale feature extraction
    """
    def __init__(self, input_dim=6, d_model=128, num_layers=4, seq_len=250,
                 order=2, num_heads=1, dropout=0.1, ff_mult=4):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len

        # Separate embeddings for magnetic and orientation data
        self.mag_embedding = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.orient_embedding = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional embedding
        self.pos_embedding = PositionalEmbedding(d_model, max_len=seq_len)

        # Hyena blocks
        self.hyena_blocks = nn.ModuleList([
            HyenaBlock(d_model, seq_len, order, num_heads, dropout, ff_mult)
            for _ in range(num_layers)
        ])

        # Multi-scale pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 6) - [MagX, MagY, MagZ, Pitch, Roll, Yaw]

        Returns:
            positions: (batch, 2) - Predicted (x, y) positions
        """
        # Split into magnetic and orientation data
        mag_data = x[:, :, :3]  # (batch, seq_len, 3)
        orient_data = x[:, :, 3:]  # (batch, seq_len, 3)

        # Embed separately
        mag_embed = self.mag_embedding(mag_data)  # (batch, seq_len, d_model//2)
        orient_embed = self.orient_embedding(orient_data)  # (batch, seq_len, d_model//2)

        # Concatenate and fuse
        x = torch.cat([mag_embed, orient_embed], dim=-1)  # (batch, seq_len, d_model)
        x = self.fusion(x)

        # Add positional embedding
        x = self.pos_embedding(x)

        # Apply Hyena blocks
        for hyena_block in self.hyena_blocks:
            x = hyena_block(x)

        # Multi-scale pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        avg_pooled = self.avg_pool(x).squeeze(-1)  # (batch, d_model)
        max_pooled = self.max_pool(x).squeeze(-1)  # (batch, d_model)
        x = torch.cat([avg_pooled, max_pooled], dim=-1)  # (batch, d_model*2)

        # Regression to position
        positions = self.regression_head(x)  # (batch, 2)

        return positions

    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_type='v1', **kwargs):
    """
    Factory function to create models

    Args:
        model_type: 'v1' or 'v2'
        **kwargs: Model configuration parameters

    Returns:
        model: HyenaPositioningModel instance
    """
    if model_type == 'v1':
        model = HyenaPositioningModel(**kwargs)
    elif model_type == 'v2':
        model = HyenaPositioningModelV2(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Model: {model_type}")
    print(f"Parameters: {model.count_parameters():,}")

    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 250
    input_dim = 6

    x = torch.randn(batch_size, seq_len, input_dim)

    # Test V1
    print("=" * 50)
    print("Testing HyenaPositioningModel V1")
    print("=" * 50)
    model_v1 = create_model('v1', d_model=128, num_layers=4, seq_len=seq_len)
    output_v1 = model_v1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_v1.shape}")
    print(f"Sample output: {output_v1[0]}")

    # Test V2
    print("\n" + "=" * 50)
    print("Testing HyenaPositioningModel V2")
    print("=" * 50)
    model_v2 = create_model('v2', d_model=128, num_layers=4, seq_len=seq_len)
    output_v2 = model_v2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_v2.shape}")
    print(f"Sample output: {output_v2[0]}")
