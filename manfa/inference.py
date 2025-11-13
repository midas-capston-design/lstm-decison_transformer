from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config_from_dict
from .data import WindowDataset
from .models import MaNFA, MagneticFieldMLP, ParticleFlowAligner, ParticleInitializer, SequenceEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MaNFA streaming inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--window-dir", type=str, default="flow_matching/processed_data_flow_matching")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out", type=str, default="results/manfa/predictions.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    cfg = load_config_from_dict(cfg_dict)

    dataset = WindowDataset(
        root=args.window_dir,
        split=args.split,
        variance_thresh=cfg.data.variance_thresh,
        yaw_flip_thresh_deg=cfg.data.yaw_flip_thresh_deg,
        arc_over_net_tol=cfg.data.arc_over_net_tol,
        cache_clean_indices=False,
        filter_quality=False,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.optim.batch_size, shuffle=False)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    field = MagneticFieldMLP(
        sensor_dim=dataset.sensor_dim,
        hidden_dim=cfg.field.hidden_dim,
        num_layers=cfg.field.num_layers,
        fourier_features=cfg.field.fourier_features,
        dropout=cfg.field.dropout,
        output_variance=cfg.field.output_variance,
    )
    encoder = SequenceEncoder(
        input_dim=dataset.sensor_dim,
        d_model=cfg.encoder.d_model,
        num_layers=cfg.encoder.num_layers,
        d_state=cfg.encoder.d_state,
        d_conv=cfg.encoder.d_conv,
        expand=cfg.encoder.expand,
        dropout=cfg.encoder.dropout,
    )
    initializer = ParticleInitializer(cfg.flow.num_particles, cfg.flow.noise_std)
    flow = ParticleFlowAligner(cfg.encoder.d_model, cfg.flow.num_steps, cfg.flow.entropy_reg)
    model = MaNFA(field, encoder, flow, initializer, traj_noise=cfg.flow.noise_std).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch)
                coords_pred = out["posterior"].cpu().numpy()
                coords_true = batch["coords_real"].cpu().numpy()
                for pred, true in zip(coords_pred, coords_true):
                    record = {"pred_xy": pred.tolist(), "true_xy": true.tolist()}
                    f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
