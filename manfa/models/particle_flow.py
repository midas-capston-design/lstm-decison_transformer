from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ParticleInitializer(nn.Module):
    def __init__(self, num_particles: int, noise_std: float = 0.05) -> None:
        super().__init__()
        self.num_particles = num_particles
        self.noise_std = noise_std

    def forward(self, traj_real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj_real: (B, T, 2) trajectory in meters.
        Returns:
            particles: (B, num_particles, 2)
        """
        final_pos = traj_real[:, -1, :2]
        noise = torch.randn(final_pos.size(0), self.num_particles, 2, device=traj_real.device)
        return final_pos.unsqueeze(1) + noise * self.noise_std


class ParticleFlowAligner(nn.Module):
    def __init__(self, latent_dim: int, num_steps: int = 3, entropy_reg: float = 0.01) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.entropy_reg = entropy_reg
        hidden = max(128, latent_dim // 2)
        self.velocity = nn.Sequential(
            nn.Linear(latent_dim * 2 + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        obs_latent: torch.Tensor,
        particle_latent: torch.Tensor,
        particles: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_latent: (B, D)
            particle_latent: (B, P, D)
            particles: (B, P, 2)
        Returns:
            posterior_mean: (B, 2)
            weights: (B, P)
            updated_particles: (B, P, 2)
        """
        coords = particles
        for _ in range(self.num_steps):
            obs_rep = obs_latent.unsqueeze(1).expand_as(particle_latent)
            flow_in = torch.cat([obs_rep, particle_latent, coords], dim=-1)
            delta = self.velocity(flow_in)
            coords = coords + delta

        similarity = torch.einsum("bd,bpd->bp", obs_latent, particle_latent)
        weights = torch.softmax(similarity, dim=-1)
        posterior = torch.sum(weights.unsqueeze(-1) * coords, dim=1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        return posterior, weights, coords, entropy
