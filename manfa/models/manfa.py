from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .mamba_encoder import SequenceEncoder
from .neural_field import MagneticFieldMLP
from .particle_flow import ParticleFlowAligner, ParticleInitializer


class MaNFA(nn.Module):
    def __init__(
        self,
        field: MagneticFieldMLP,
        encoder: SequenceEncoder,
        flow: ParticleFlowAligner,
        initializer: ParticleInitializer,
        traj_noise: float = 0.05,
    ) -> None:
        super().__init__()
        self.field = field
        self.encoder = encoder
        self.flow = flow
        self.initializer = initializer
        self.traj_noise = traj_noise

    def _sample_particle_paths(self, traj_real: torch.Tensor, particles: torch.Tensor) -> torch.Tensor:
        """
        Expand each ground-truth trajectory with smooth noise so every particle
        carries a plausible candidate path.
        """
        B, T, _ = traj_real.shape
        P = particles.shape[1]
        base = traj_real.unsqueeze(1).expand(-1, P, -1, -1)
        noise = torch.randn_like(base) * self.traj_noise
        time_scale = torch.linspace(0, 1, T, device=traj_real.device).view(1, 1, T, 1)
        perturbed = base + noise * time_scale
        perturbed[:, :, -1, :] = particles
        return perturbed

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = batch["states"]  # (B, T, 6)
        traj_real = batch["trajectory_real"]  # (B, T, 2)
        coords_real = batch["coords_real"]  # (B, 2)

        seq_latent, obs_latent = self.encoder(states)

        field_seq, _ = self.field(traj_real)
        _, field_latent = self.encoder(field_seq)

        particles = self.initializer(traj_real)
        particle_traj = self._sample_particle_paths(traj_real, particles)
        particle_seq, _ = self.field(particle_traj)

        B, P, T, C = particle_seq.shape
        particle_seq = particle_seq.view(B * P, T, C)
        _, particle_latent = self.encoder(particle_seq)
        particle_latent = particle_latent.view(B, P, -1)

        posterior, weights, updated_particles, entropy = self.flow(obs_latent, particle_latent, particles)

        return {
            "field_seq": field_seq,
            "seq_latent": seq_latent,
            "obs_latent": obs_latent,
            "field_latent": field_latent,
            "posterior": posterior,
            "particle_weights": weights,
            "particles": updated_particles,
            "coords_real": coords_real,
            "traj_real": traj_real,
            "entropy": entropy,
        }
