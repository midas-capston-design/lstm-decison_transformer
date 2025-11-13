#!/usr/bin/env python3
"""
Flow Matching for Trajectory Prediction (전체 궤적 예측)

핵심 변경사항:
- 출력: 마지막 위치 (B, 2) → 전체 궤적 (B, T, 2)
- 250개 위치 모두 예측 → 250배 감독 신호
- One-to-Many 문제 근본 해결
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SensorEncoder(nn.Module):
    """
    센서 시퀀스를 시퀀스 representation으로 인코딩 (궤적 예측용)

    Input: (B, T, 6) - 센서 시퀀스
    Output: (B, T, d_model) - 시퀀스 representation
    """
    def __init__(self, input_dim=6, d_model=256, n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=300)

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

    def forward(self, x):
        """
        Args:
            x: (B, T, 6) - 센서 시퀀스
        Returns:
            seq_repr: (B, T, d_model) - 시퀀스 representation
        """
        # Embed and add positional encoding
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, T, d_model)

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, dropout=0.1, max_len=300):
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


class TrajectoryDecoder(nn.Module):
    """
    궤적 예측 디코더 (Flow Matching 기반)

    시퀀스 representation → 전체 궤적 (B, T, 2)
    """
    def __init__(self, d_model=256, position_dim=2, n_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.position_dim = position_dim

        # Time embedding for flow matching
        self.time_embed_dim = 64

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Trajectory decoder layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))

        # Output projection: (B, T, d_model) → (B, T, 2)
        self.output_proj = nn.Linear(d_model, position_dim)

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

    def forward(self, seq_repr, t):
        """
        Args:
            seq_repr: (B, T, d_model) - sequence representation from encoder
            t: (B,) - flow matching timestep in [0, 1]

        Returns:
            trajectory: (B, T, 2) - predicted trajectory
        """
        B, T, _ = seq_repr.shape

        # Time embedding
        time_emb = self.get_time_embedding(t)  # (B, time_embed_dim)
        time_emb = self.time_mlp(time_emb)  # (B, d_model)
        time_emb = time_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model)

        # Combine sequence representation with time
        h = seq_repr + time_emb  # (B, T, d_model)

        # Process through decoder layers
        for layer in self.decoder_layers:
            h = layer(h) + h  # Residual connection

        # Project to trajectory
        trajectory = self.output_proj(h)  # (B, T, 2)

        return trajectory


class FlowMatchingTrajectory(nn.Module):
    """
    Flow Matching for Trajectory Prediction

    Training: learns to predict entire trajectory (B, T, 2)
    Inference: solves ODE from noise to trajectory
    """
    def __init__(self,
                 sensor_dim=6,
                 position_dim=2,
                 sequence_length=250,
                 d_model=256,
                 encoder_layers=4,
                 decoder_layers=4,
                 n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.position_dim = position_dim
        self.sequence_length = sequence_length

        # Sensor encoder (시퀀스 → 시퀀스)
        self.encoder = SensorEncoder(
            input_dim=sensor_dim,
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            d_model=d_model,
            position_dim=position_dim,
            n_layers=decoder_layers,
            dropout=dropout
        )

    def forward(self, traj_t, t, sensor_data):
        """
        Args:
            traj_t: (B, T, 2) - trajectory at time t
            t: (B,) - timestep in [0, 1]
            sensor_data: (B, T, 6) - sensor sequence

        Returns:
            velocity: (B, T, 2) - predicted velocity field
        """
        # Encode sensor data
        seq_repr = self.encoder(sensor_data)  # (B, T, d_model)

        # Decode to trajectory
        velocity = self.decoder(seq_repr, t)  # (B, T, 2)

        return velocity

    @torch.no_grad()
    def sample(self, sensor_data, n_steps=10, return_trajectory=False):
        """
        Sample trajectories from the model (inference)

        Args:
            sensor_data: (B, T, 6) - sensor sequence
            n_steps: number of ODE steps
            return_trajectory: if True, return full flow trajectory

        Returns:
            trajectory: (B, T, 2) - predicted trajectory
            flow_trajectory: (B, n_steps+1, T, 2) if return_trajectory=True
        """
        B, T, _ = sensor_data.shape
        device = sensor_data.device

        # Start from Gaussian noise
        traj = torch.randn(B, T, self.position_dim, device=device)

        if return_trajectory:
            flow_traj = [traj.clone()]

        # Euler method for ODE solving
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.ones(B, device=device) * (step / n_steps)

            # Predict velocity
            v = self.forward(traj, t, sensor_data)

            # Update trajectory
            traj = traj + v * dt

            if return_trajectory:
                flow_traj.append(traj.clone())

        if return_trajectory:
            return traj, torch.stack(flow_traj, dim=1)
        else:
            return traj


def compute_trajectory_loss(model, sensor_data, target_trajectories, use_topk=False, k_ratio=0.5):
    """
    Compute Flow Matching loss for trajectory prediction

    Loss = E_t,traj_0,traj_1 [ ||v_θ(traj_t, t, c) - (traj_1 - traj_0)||^2 ]

    Args:
        model: FlowMatchingTrajectory model
        sensor_data: (B, T, 6) - sensor sequences
        target_trajectories: (B, T, 2) - true trajectories
        use_topk: if True, use top-k hard example mining
        k_ratio: ratio of samples to use (0.5 = top 50% hardest samples)

    Returns:
        loss: scalar loss value
        final_pos_loss: loss on final position only (for logging)
    """
    B, T, _ = sensor_data.shape
    device = sensor_data.device

    # Sample noise (traj_0)
    traj_0 = torch.randn_like(target_trajectories)

    # traj_1 = target trajectories
    traj_1 = target_trajectories

    # Sample random time
    t = torch.rand(B, device=device)

    # Linear interpolation: traj_t = (1-t)*traj_0 + t*traj_1
    t_expanded = t[:, None, None]  # (B, 1, 1)
    traj_t = (1 - t_expanded) * traj_0 + t_expanded * traj_1

    # True velocity (optimal transport)
    true_velocity = traj_1 - traj_0  # (B, T, 2)

    # Predicted velocity
    pred_velocity = model(traj_t, t, sensor_data)  # (B, T, 2)

    if use_topk:
        # Top-k Loss: 가장 어려운 샘플들에 집중
        # Per-sample loss (average over T and 2)
        losses = F.mse_loss(pred_velocity, true_velocity, reduction='none').mean(dim=[1, 2])  # (B,)
        k = max(1, int(B * k_ratio))
        topk_losses, _ = torch.topk(losses, k)
        loss = topk_losses.mean()
    else:
        # MSE loss over entire trajectory
        loss = F.mse_loss(pred_velocity, true_velocity)

    # Also compute final position loss for logging
    final_pos_loss = F.mse_loss(pred_velocity[:, -1, :], true_velocity[:, -1, :])

    return loss, final_pos_loss


if __name__ == '__main__':
    """Test the trajectory model"""
    print("=" * 70)
    print("🔬 Flow Matching Trajectory Model Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model
    model = FlowMatchingTrajectory(
        sensor_dim=6,
        position_dim=2,
        sequence_length=250,
        d_model=256,
        encoder_layers=4,
        decoder_layers=4,
        n_heads=8,
        dropout=0.1
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Test forward pass
    B = 4
    T = 250
    sensor_data = torch.randn(B, T, 6, device=device)
    target_traj = torch.randn(B, T, 2, device=device)

    print(f"\nTest input:")
    print(f"  Sensor data: {sensor_data.shape}")
    print(f"  Target trajectory: {target_traj.shape}")

    # Test training loss
    loss, final_loss = compute_trajectory_loss(model, sensor_data, target_traj)
    print(f"\nTraining loss:")
    print(f"  Full trajectory: {loss.item():.4f}")
    print(f"  Final position: {final_loss.item():.4f}")

    # Test sampling
    model.eval()
    trajectory, flow_traj = model.sample(sensor_data, n_steps=10, return_trajectory=True)
    print(f"\nSampling:")
    print(f"  Final trajectory: {trajectory.shape}")
    print(f"  Flow trajectory: {flow_traj.shape}")

    print("\n✅ Trajectory model test passed!")
