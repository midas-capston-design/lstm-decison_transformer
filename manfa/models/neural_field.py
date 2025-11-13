from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def _fourier_features(coords: torch.Tensor, num_features: int) -> torch.Tensor:
    """Apply random Fourier features to (x, y) coordinates."""
    if num_features <= 0:
        return coords
    batch_shape = coords.shape[:-1]
    device = coords.device
    freq = torch.linspace(1.0, math.pi, num_features, device=device)
    proj = coords[..., None] * freq  # [..., 2, F]
    sin = torch.sin(proj)
    cos = torch.cos(proj)
    return torch.cat([coords, sin.reshape(*batch_shape, -1), cos.reshape(*batch_shape, -1)], dim=-1)


class MagneticFieldMLP(nn.Module):
    def __init__(
        self,
        sensor_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        fourier_features: int = 32,
        dropout: float = 0.1,
        output_variance: bool = True,
    ) -> None:
        super().__init__()
        self.sensor_dim = sensor_dim
        self.fourier_features = fourier_features
        self.output_variance = output_variance

        input_dim = 2 + 2 + 1  # xy + heading sin/cos + normalized time
        if fourier_features > 0:
            input_dim += 2 * 2 * fourier_features

        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.out_mean = nn.Linear(hidden_dim, sensor_dim)
        if output_variance:
            self.out_logvar = nn.Linear(hidden_dim, sensor_dim)

    @staticmethod
    def _infer_heading(positions: torch.Tensor) -> torch.Tensor:
        diffs = torch.diff(positions, dim=-2, prepend=positions[..., :1, :])
        heading = torch.atan2(diffs[..., 1], diffs[..., 0])
        return heading

    def _prepare_inputs(
        self,
        positions: torch.Tensor,
        headings: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        batch_shape = positions.shape[:-2]
        seq_len = positions.shape[-2]

        if headings is None:
            headings = self._infer_heading(positions)
        sin_h = torch.sin(headings)
        cos_h = torch.cos(headings)

        coords = positions.view(-1, seq_len, 2)
        sin_h = sin_h.view(-1, seq_len, 1)
        cos_h = cos_h.view(-1, seq_len, 1)

        t = torch.linspace(0.0, 1.0, seq_len, device=positions.device)
        t = t.view(1, seq_len, 1).expand(coords.size(0), -1, -1)

        xy = coords
        if self.fourier_features > 0:
            xy = _fourier_features(coords, self.fourier_features)

        feats = torch.cat([xy, sin_h, cos_h, t], dim=-1)
        return feats, seq_len

    def forward(
        self,
        positions: torch.Tensor,
        headings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            positions: (..., T, 2) coordinates in meters.
            headings:  (..., T) optional heading in radians.
        Returns:
            mean_seq, logvar_seq (if enabled) shaped (..., T, sensor_dim).
        """
        batch_shape = positions.shape[:-2]
        feats, seq_len = self._prepare_inputs(positions, headings)
        hidden = self.mlp(feats)
        mean = self.out_mean(hidden).view(*batch_shape, seq_len, self.sensor_dim)

        if not self.output_variance:
            return mean, None
        logvar = self.out_logvar(hidden).view(*batch_shape, seq_len, self.sensor_dim)
        return mean, logvar
