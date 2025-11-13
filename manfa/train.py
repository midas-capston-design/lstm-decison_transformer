from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import MaNFAConfig, load_config_from_dict
from .data import WindowDataset
from .models import MaNFA, MagneticFieldMLP, ParticleFlowAligner, ParticleInitializer, SequenceEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaNFA prototype")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--stage", type=str, default="full", choices=["field", "full"])
    parser.add_argument("--save-dir", type=str, default="results/manfa")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def info_nce(obs: torch.Tensor, field_latent: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = torch.matmul(obs, field_latent.t()) / temperature
    targets = torch.arange(obs.size(0), device=obs.device)
    return F.cross_entropy(logits, targets)


def build_model(cfg: MaNFAConfig, sensor_dim: int) -> MaNFA:
    field = MagneticFieldMLP(
        sensor_dim=sensor_dim,
        hidden_dim=cfg.field.hidden_dim,
        num_layers=cfg.field.num_layers,
        fourier_features=cfg.field.fourier_features,
        dropout=cfg.field.dropout,
        output_variance=cfg.field.output_variance,
    )
    encoder = SequenceEncoder(
        input_dim=sensor_dim,
        d_model=cfg.encoder.d_model,
        num_layers=cfg.encoder.num_layers,
        d_state=cfg.encoder.d_state,
        d_conv=cfg.encoder.d_conv,
        expand=cfg.encoder.expand,
        dropout=cfg.encoder.dropout,
    )
    initializer = ParticleInitializer(cfg.flow.num_particles, cfg.flow.noise_std)
    flow = ParticleFlowAligner(cfg.encoder.d_model, cfg.flow.num_steps, cfg.flow.entropy_reg)
    return MaNFA(field, encoder, flow, initializer, traj_noise=cfg.flow.noise_std)


def main() -> None:
    args = parse_args()

    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    cfg = load_config_from_dict(cfg_dict)
    set_seed(cfg.seed)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = WindowDataset(
        root=cfg.data.root,
        split=cfg.data.split,
        variance_thresh=cfg.data.variance_thresh,
        yaw_flip_thresh_deg=cfg.data.yaw_flip_thresh_deg,
        arc_over_net_tol=cfg.data.arc_over_net_tol,
        cache_clean_indices=cfg.data.cache_clean_indices,
        filter_quality=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = build_model(cfg, dataset.sensor_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    recon_loss = nn.SmoothL1Loss()

    config_text = Path(args.config).read_text()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config_used.yaml").write_text(config_text)

    for epoch in range(1, cfg.optim.epochs + 1):
        model.train()
        running = {"field": 0.0, "contrastive": 0.0, "particle": 0.0, "total": 0.0}
        steps = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.optim.epochs}")

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)

            loss_field = recon_loss(out["field_seq"], batch["states"])
            loss_contrast = info_nce(out["obs_latent"], out["field_latent"])
            total_loss = cfg.loss.lambda_field * loss_field + cfg.loss.lambda_contrastive * loss_contrast

            if args.stage == "full":
                loss_particle = recon_loss(out["posterior"], batch["coords_real"])
                total_loss += cfg.loss.lambda_particle * loss_particle + cfg.flow.entropy_reg * out["entropy"]
                running["particle"] += loss_particle.item()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()

            running["field"] += loss_field.item()
            running["contrastive"] += loss_contrast.item()
            running["total"] += total_loss.item()
            steps += 1

            pbar.set_postfix(
                field=f"{running['field']/max(steps,1):.3f}",
                contrast=f"{running['contrastive']/max(steps,1):.3f}",
                particle=f"{running['particle']/max(steps,1):.3f}",
                total=f"{running['total']/max(steps,1):.3f}",
            )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
