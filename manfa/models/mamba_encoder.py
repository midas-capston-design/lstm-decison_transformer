from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - informative error for missing dep
    Mamba = None
    _import_error = exc
else:
    _import_error = None


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba-ssm is required for SequenceEncoder. Install it via `pip install mamba-ssm`."
            ) from _import_error
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mamba(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim) normalized sensor sequence.
        Returns:
            seq_latent: (B, T, d_model)
            pooled_latent: (B, d_model)
        """
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        seq_latent = self.norm(h)
        pooled = seq_latent.mean(dim=1)
        return seq_latent, pooled
