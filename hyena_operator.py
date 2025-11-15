import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    """Positional embedding for sequence data"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class HyenaFilter(nn.Module):
    """
    Hyena filter for long-range convolution
    Uses implicit neural representation for the filter
    """
    def __init__(self, d_model, seq_len, order=2, num_heads=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.order = order
        self.num_heads = num_heads

        # Filter generator network
        self.filter_network = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, d_model * order)
        )

        # Learnable position encoding for filter
        self.register_buffer('filter_pos', torch.linspace(0, 1, seq_len).unsqueeze(-1))

    def forward(self, seq_len=None):
        """Generate filter coefficients"""
        if seq_len is None:
            seq_len = self.seq_len

        # Generate position encodings
        pos = torch.linspace(0, 1, seq_len, device=self.filter_pos.device).unsqueeze(-1)

        # Generate filter coefficients using MLP
        filter_coefs = self.filter_network(pos)  # (seq_len, d_model * order)
        filter_coefs = filter_coefs.reshape(seq_len, self.d_model, self.order)

        return filter_coefs


class HyenaOperator(nn.Module):
    """
    Hyena Operator - Subquadratic alternative to attention

    Hyena uses a combination of:
    1. Short convolutions for local interactions
    2. Long convolutions with learned filters for global interactions
    3. Element-wise multiplicative gating
    """
    def __init__(self, d_model, seq_len, order=2, num_heads=1, dropout=0.1, filter_dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.order = order
        self.num_heads = num_heads

        # Input projection (creates order+1 projections)
        self.in_proj = nn.Linear(d_model, d_model * (order + 1))

        # Short convolution for local interactions
        self.short_filter = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            groups=d_model
        )

        # Hyena filter for long-range dependencies
        self.hyena_filter = HyenaFilter(d_model, seq_len, order, num_heads, dropout)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.filter_dropout = nn.Dropout(filter_dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project input to order+1 branches
        projections = self.in_proj(x)  # (batch, seq_len, d_model * (order+1))
        projections = projections.reshape(batch, seq_len, self.d_model, self.order + 1)

        # Split into v, x1, x2, ... (order+1 total)
        v = projections[..., 0]  # (batch, seq_len, d_model)
        x_branches = [projections[..., i] for i in range(1, self.order + 1)]

        # Apply short convolution to v
        v = v.transpose(1, 2)  # (batch, d_model, seq_len)
        v = self.short_filter(v)
        v = v.transpose(1, 2)  # (batch, seq_len, d_model)

        # Get filter coefficients
        filter_coefs = self.hyena_filter(seq_len)  # (seq_len, d_model, order)

        # Apply Hyena recurrence: x_{k+1} = x_k * (h_k * v)
        output = v
        for i in range(self.order):
            # Get current branch
            x_k = x_branches[i]  # (batch, seq_len, d_model)

            # Get filter for this order
            h_k = filter_coefs[:, :, i]  # (seq_len, d_model)

            # Apply long convolution via FFT
            # This is the key to subquadratic complexity
            v_fft = torch.fft.rfft(v, n=2*seq_len, dim=1)
            h_fft = torch.fft.rfft(h_k.unsqueeze(0), n=2*seq_len, dim=0)

            # Multiply in frequency domain
            conv_result = torch.fft.irfft(v_fft * h_fft, n=2*seq_len, dim=1)
            conv_result = conv_result[:, :seq_len, :]  # (batch, seq_len, d_model)

            # Element-wise gating
            output = x_k * conv_result

        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        return output


class HyenaBlock(nn.Module):
    """
    A single Hyena block with residual connection and layer norm
    """
    def __init__(self, d_model, seq_len, order=2, num_heads=1, dropout=0.1, ff_mult=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(d_model, seq_len, order, num_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Hyena with residual
        x = x + self.hyena(self.norm1(x))

        # Feedforward with residual
        x = x + self.ff(self.norm2(x))

        return x


if __name__ == "__main__":
    # Test the Hyena operator
    batch_size = 4
    seq_len = 250
    d_model = 64

    x = torch.randn(batch_size, seq_len, d_model)

    hyena_block = HyenaBlock(d_model, seq_len, order=2, num_heads=1)
    output = hyena_block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Hyena block test passed!")
