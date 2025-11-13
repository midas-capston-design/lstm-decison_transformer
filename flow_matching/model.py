#!/usr/bin/env python3
"""
Flow Matching for Magnetic Field Indoor Positioning

핵심 아이디어:
- Gaussian noise에서 실제 위치로의 "flow" 학습
- 센서 데이터로 conditioning
- 1-2 step inference로 실시간 가능
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SensorEncoder(nn.Module):
    """
    센서 시퀀스를 context vector로 인코딩

    Input: (B, T, 6) - 센서 시퀀스
    Output: (B, d_model) - context vector
    """
    def __init__(self, input_dim=6, d_model=256, n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=300)  # 250 timesteps 지원

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection (pool to single vector)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: (B, T, 6) - 센서 시퀀스
        Returns:
            context: (B, d_model) - context vector
        """
        B, T, _ = x.shape

        # Embed and add positional encoding
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, T, d_model)

        # Pool to single vector
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.pool(x).squeeze(-1)  # (B, d_model)

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VelocityNet(nn.Module):
    """
    Flow Matching velocity network

    v_θ(x_t, t, context) → velocity (2D)

    Predicts the vector field at position x_t, time t, given sensor context
    """
    def __init__(self, position_dim=2, d_model=256, n_layers=4, dropout=0.1):
        super().__init__()
        self.position_dim = position_dim
        self.d_model = d_model

        # Time embedding (sinusoidal)
        self.time_embed_dim = 64

        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(position_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Main network: combines position, time, context
        self.net = nn.ModuleList()
        for _ in range(n_layers):
            self.net.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))

        # Output projection
        self.output = nn.Linear(d_model, position_dim)

    def get_time_embedding(self, t):
        """
        Sinusoidal time embedding

        Args:
            t: (B,) - timestep in [0, 1]
        Returns:
            (B, time_embed_dim)
        """
        B = t.shape[0]
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, time_embed_dim)
        return emb

    def forward(self, x_t, t, context):
        """
        Args:
            x_t: (B, 2) - current position at time t
            t: (B,) - timestep in [0, 1]
            context: (B, d_model) - sensor context from encoder

        Returns:
            velocity: (B, 2) - predicted velocity
        """
        # Embed position
        pos_emb = self.pos_embed(x_t)  # (B, d_model)

        # Embed time
        time_emb = self.get_time_embedding(t)  # (B, time_embed_dim)
        time_emb = self.time_mlp(time_emb)  # (B, d_model)

        # Combine: position + time + context
        h = pos_emb + time_emb + context  # (B, d_model)

        # Process through network
        for layer in self.net:
            h = layer(h) + h  # Residual connection

        # Output velocity
        velocity = self.output(h)  # (B, 2)

        return velocity


class FlowMatchingLocalization(nn.Module):
    """
    Complete Flow Matching model for indoor positioning

    Training: learns v_θ to match x_1 - x_0
    Inference: solves ODE from noise to position
    """
    def __init__(self,
                 sensor_dim=6,
                 position_dim=2,
                 d_model=256,
                 encoder_layers=4,
                 velocity_layers=4,
                 n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.position_dim = position_dim

        # Sensor encoder
        self.encoder = SensorEncoder(
            input_dim=sensor_dim,
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # Velocity network
        self.velocity_net = VelocityNet(
            position_dim=position_dim,
            d_model=d_model,
            n_layers=velocity_layers,
            dropout=dropout
        )

    def forward(self, x_t, t, sensor_data):
        """
        Args:
            x_t: (B, 2) - position at time t
            t: (B,) - timestep in [0, 1]
            sensor_data: (B, T, 6) - sensor sequence

        Returns:
            velocity: (B, 2) - predicted velocity
        """
        # Encode sensor data
        context = self.encoder(sensor_data)  # (B, d_model)

        # Predict velocity
        velocity = self.velocity_net(x_t, t, context)  # (B, 2)

        return velocity

    @torch.no_grad()
    def sample(self, sensor_data, n_steps=10, return_trajectory=False):
        """
        Sample positions from the model (inference)

        Args:
            sensor_data: (B, T, 6) - sensor sequence
            n_steps: number of ODE steps (default 10, can use 1-2 for speed)
            return_trajectory: if True, return full trajectory

        Returns:
            positions: (B, 2) - predicted positions
            trajectory: (B, n_steps+1, 2) if return_trajectory=True
        """
        B = sensor_data.shape[0]
        device = sensor_data.device

        # Start from Gaussian noise
        x = torch.randn(B, self.position_dim, device=device)

        if return_trajectory:
            trajectory = [x.clone()]

        # Euler method for ODE solving
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.ones(B, device=device) * (step / n_steps)

            # Predict velocity
            v = self.forward(x, t, sensor_data)

            # Update position
            x = x + v * dt

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, torch.stack(trajectory, dim=1)
        else:
            return x

    @torch.no_grad()
    def sample_topk(self, sensor_data, n_samples=10, k=5, n_steps=10):
        """
        Top-k sampling: 여러 노이즈에서 샘플링 후 신뢰도 높은 k개 중 최고 1개 선택

        Args:
            sensor_data: (B, T, 6) - sensor sequence
            n_samples: 총 샘플링 횟수
            k: 후보 개수 (이 중 최고 신뢰도 1개 선택)
            n_steps: ODE step 수

        Returns:
            positions: (B, 2) - 최고 신뢰도 위치
            topk_positions: (B, k, 2) - top-k 후보 위치들
            topk_scores: (B, k) - top-k 신뢰도 점수들
        """
        B = sensor_data.shape[0]
        device = sensor_data.device

        # Encode sensor data once
        context = self.encoder(sensor_data)  # (B, d_model)

        all_positions = []
        all_scores = []

        # 여러 노이즈에서 샘플링
        for _ in range(n_samples):
            # 각 샘플마다 다른 노이즈에서 시작
            pos = self.sample(sensor_data, n_steps=n_steps)  # (B, 2)
            all_positions.append(pos)

            # 신뢰도 점수 계산: 최종 velocity의 크기가 작을수록 수렴했다는 의미
            t_final = torch.ones(B, device=device)
            v_final = self.velocity_net(pos, t_final, context)
            score = -torch.norm(v_final, dim=1)  # (B,) - 낮은 velocity = 높은 점수
            all_scores.append(score)

        # Stack: (B, n_samples, 2), (B, n_samples)
        all_positions = torch.stack(all_positions, dim=1)
        all_scores = torch.stack(all_scores, dim=1)

        # 각 배치별로 top-k 선택
        best_positions = []
        topk_positions_list = []
        topk_scores_list = []

        for b in range(B):
            # b번째 샘플의 top-k 인덱스 (신뢰도 높은 순)
            topk_scores_b, topk_idx = torch.topk(all_scores[b], k)
            topk_pos_b = all_positions[b, topk_idx]  # (k, 2)

            # Top-k 중 최고 신뢰도 (첫 번째) 선택
            best_pos = topk_pos_b[0]  # (2,)

            best_positions.append(best_pos)
            topk_positions_list.append(topk_pos_b)
            topk_scores_list.append(topk_scores_b)

        best_positions = torch.stack(best_positions, dim=0)  # (B, 2)
        topk_positions = torch.stack(topk_positions_list, dim=0)  # (B, k, 2)
        topk_scores = torch.stack(topk_scores_list, dim=0)  # (B, k)

        return best_positions, topk_positions, topk_scores


def compute_flow_matching_loss(model, sensor_data, target_positions, use_topk=False, k_ratio=0.5):
    """
    Compute Flow Matching training loss

    Loss = E_t,x_0,x_1 [ ||v_θ(x_t, t, c) - (x_1 - x_0)||^2 ]

    Args:
        model: FlowMatchingLocalization model
        sensor_data: (B, T, 6) - sensor sequences
        target_positions: (B, 2) - true positions
        use_topk: if True, use top-k hard example mining
        k_ratio: ratio of samples to use (0.5 = top 50% hardest samples)

    Returns:
        loss: scalar loss value
    """
    B = sensor_data.shape[0]
    device = sensor_data.device

    # Sample noise (x_0)
    x_0 = torch.randn_like(target_positions)

    # x_1 = target positions
    x_1 = target_positions

    # Sample random time
    t = torch.rand(B, device=device)

    # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
    t_expanded = t[:, None]  # (B, 1)
    x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

    # True velocity (optimal transport)
    true_velocity = x_1 - x_0

    # Predicted velocity
    pred_velocity = model(x_t, t, sensor_data)

    if use_topk:
        # Top-k Loss: 가장 어려운 샘플들에 집중
        losses = F.mse_loss(pred_velocity, true_velocity, reduction='none').mean(dim=1)  # (B,)
        k = max(1, int(B * k_ratio))
        topk_losses, _ = torch.topk(losses, k)
        loss = topk_losses.mean()
    else:
        # MSE loss
        loss = F.mse_loss(pred_velocity, true_velocity)

    return loss


if __name__ == '__main__':
    """Test the model"""
    print("=" * 70)
    print("🔬 Flow Matching Model Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model
    model = FlowMatchingLocalization(
        sensor_dim=6,
        position_dim=2,
        d_model=256,
        encoder_layers=4,
        velocity_layers=4,
        n_heads=8,
        dropout=0.1
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Test forward pass
    B = 4
    T = 100
    sensor_data = torch.randn(B, T, 6, device=device)
    target_pos = torch.randn(B, 2, device=device)

    print(f"\nTest input:")
    print(f"  Sensor data: {sensor_data.shape}")
    print(f"  Target pos: {target_pos.shape}")

    # Test training loss
    loss = compute_flow_matching_loss(model, sensor_data, target_pos)
    print(f"\nTraining loss: {loss.item():.4f}")

    # Test sampling
    model.eval()
    positions, trajectory = model.sample(sensor_data, n_steps=10, return_trajectory=True)
    print(f"\nSampling:")
    print(f"  Final positions: {positions.shape}")
    print(f"  Trajectory: {trajectory.shape}")

    print("\n✅ Model test passed!")
