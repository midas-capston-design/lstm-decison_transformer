#!/usr/bin/env python3
"""
Hyena Hierarchy for Magnetic Field Indoor Positioning

핵심:
- Long Convolution으로 전역 패턴 포착
- Gating mechanism으로 중요 정보 선택
- 모든 timestep을 동등하게 처리
- Transformer보다 효율적 (O(N log N))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft


class HyenaFilter(nn.Module):
    """
    Hyena의 핵심: Implicit Long Convolution Filter

    일반 Convolution: 고정된 커널 크기 (3x3, 5x5 등)
    Hyena Filter: 전체 시퀀스 길이의 필터 (250 길이)
    """
    def __init__(self, d_model, seq_len, order=2, filter_order=64):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.order = order

        # Implicit filter parameterization
        # 작은 MLP로 긴 필터 생성 (효율적)
        self.filter_fn = nn.Sequential(
            nn.Linear(filter_order, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Positional encoding for filter
        self.register_buffer(
            'pos_emb',
            self._get_pos_embedding(seq_len, filter_order)
        )

    def _get_pos_embedding(self, seq_len, dim):
        """Sinusoidal positional encoding"""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))

        pos_emb = torch.zeros(seq_len, dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)

        return pos_emb

    def forward(self, x):
        """
        Args:
            x: (B, L, D) - input sequence
        Returns:
            (B, L, D) - filtered sequence
        """
        B, L, D = x.shape

        # Generate filter from positional encoding
        filter_coeffs = self.filter_fn(self.pos_emb)  # (L, D)

        # FFT-based convolution (efficient)
        x_fft = rfft(x, n=2*L, dim=1)  # (B, L', D)
        filter_fft = rfft(filter_coeffs.unsqueeze(0), n=2*L, dim=1)  # (1, L', D)

        # Element-wise multiplication in frequency domain
        out_fft = x_fft * filter_fft

        # Inverse FFT
        out = irfft(out_fft, n=2*L, dim=1)[:, :L, :]  # (B, L, D)

        return out


class HyenaOperator(nn.Module):
    """
    Hyena Operator: Filter + Gating

    핵심 아이디어:
    1. Long convolution으로 전역 패턴 추출
    2. Gating으로 중요한 정보만 선택
    """
    def __init__(self, d_model, seq_len, order=2, filter_order=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.order = order

        # Projections for input
        self.in_proj = nn.Linear(d_model, (order + 1) * d_model)

        # Hyena filters (one for each order)
        self.filters = nn.ModuleList([
            HyenaFilter(d_model, seq_len, order, filter_order)
            for _ in range(order)
        ])

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        B, L, D = x.shape

        # Project input to multiple branches
        proj = self.in_proj(x)  # (B, L, (order+1)*D)
        v, *xs = proj.chunk(self.order + 1, dim=-1)  # v: (B,L,D), xs: list of (B,L,D)

        # Apply filters with gating
        out = v
        for i, (x_i, filter_i) in enumerate(zip(xs, self.filters)):
            # Filter
            filtered = filter_i(x_i)  # (B, L, D)

            # Gating
            out = out * filtered

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class HyenaBlock(nn.Module):
    """
    Complete Hyena Block with residual connections
    """
    def __init__(self, d_model, seq_len, order=2, filter_order=64, dropout=0.1):
        super().__init__()

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)

        # Hyena operator
        self.hyena = HyenaOperator(d_model, seq_len, order, filter_order, dropout)

        # FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        # Hyena with residual
        x = x + self.hyena(self.norm1(x))

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class HyenaLocalization(nn.Module):
    """
    Complete Hyena model for indoor positioning

    Input: 센서 시퀀스 (B, 250, 6)
    Output: 위치 (B, 2)
    """
    def __init__(
        self,
        input_dim=6,
        seq_len=250,
        d_model=256,
        n_layers=4,
        order=2,
        filter_order=64,
        dropout=0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            self._get_positional_encoding(seq_len, d_model)
        )

        # Hyena blocks
        self.layers = nn.ModuleList([
            HyenaBlock(d_model, seq_len, order, filter_order, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # (x, y) coordinates
        )

    def _get_positional_encoding(self, seq_len, d_model):
        """Sinusoidal positional encoding"""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pos_enc = torch.zeros(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return pos_enc

    def forward(self, x):
        """
        Args:
            x: (B, L, 6) - sensor sequence
        Returns:
            positions: (B, 2) - (x, y) coordinates
        """
        B, L, _ = x.shape

        # Input projection + positional encoding
        x = self.input_proj(x)  # (B, L, D)
        x = x + self.pos_encoding.unsqueeze(0)  # (B, L, D)

        # Hyena layers
        for layer in self.layers:
            x = layer(x)

        # Normalize
        x = self.norm(x)  # (B, L, D)

        # Pool over sequence
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.pool(x).squeeze(-1)  # (B, D)

        # Predict position
        positions = self.head(x)  # (B, 2)

        return positions


if __name__ == '__main__':
    """Test the model"""
    print("=" * 70)
    print("🔬 Hyena Localization Model Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model
    model = HyenaLocalization(
        input_dim=6,
        seq_len=250,
        d_model=256,
        n_layers=4,
        order=2,
        filter_order=64,
        dropout=0.1
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Test forward pass
    B = 4
    L = 250
    sensor_data = torch.randn(B, L, 6, device=device)

    print(f"\nTest input:")
    print(f"  Sensor data: {sensor_data.shape}")

    positions = model(sensor_data)
    print(f"\nOutput:")
    print(f"  Positions: {positions.shape}")
    print(f"  Sample output: {positions[0].tolist()}")

    print("\n✅ Model test passed!")

    print("\n" + "=" * 70)
    print("Hyena 모델 특징:")
    print("=" * 70)
    print("1. Long Convolution: 전체 250개 샘플의 전역 패턴 포착")
    print("2. Gating Mechanism: 중요한 정보 선택적 추출")
    print("3. FFT 기반: O(N log N) 효율성")
    print("4. 모든 timestep 동등 처리: 어느 순간도 무시하지 않음")
    print("=" * 70)
