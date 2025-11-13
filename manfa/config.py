from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    root: Path = Path("flow_matching/processed_data_flow_matching")
    split: str = "train"
    window_size: int = 250
    stride: int = 50
    sensor_cols: List[str] = dc_field(
        default_factory=lambda: ["MagX", "MagY", "MagZ", "Pitch", "Roll", "Yaw"]
    )
    variance_thresh: float = 1e-3
    yaw_flip_thresh_deg: float = 25.0
    arc_over_net_tol: float = 1.8
    cache_clean_indices: bool = True


@dataclass
class FieldConfig:
    hidden_dim: int = 256
    num_layers: int = 4
    fourier_features: int = 32
    dropout: float = 0.1
    output_variance: bool = True


@dataclass
class EncoderConfig:
    d_model: int = 256
    num_layers: int = 4
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1


@dataclass
class ParticleFlowConfig:
    num_particles: int = 64
    num_steps: int = 3
    noise_std: float = 0.05
    entropy_reg: float = 0.01


@dataclass
class OptimConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 50
    grad_clip: float = 1.0


@dataclass
class LossConfig:
    lambda_field: float = 1.0
    lambda_contrastive: float = 0.3
    lambda_particle: float = 0.5


@dataclass
class MaNFAConfig:
    data: DataConfig = dc_field(default_factory=DataConfig)
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    encoder: EncoderConfig = dc_field(default_factory=EncoderConfig)
    flow: ParticleFlowConfig = dc_field(default_factory=ParticleFlowConfig)
    optim: OptimConfig = dc_field(default_factory=OptimConfig)
    loss: LossConfig = dc_field(default_factory=LossConfig)
    device: Optional[str] = None
    seed: int = 17


def load_config_from_dict(cfg_dict: dict) -> MaNFAConfig:
    """Utility to convert nested dicts (from YAML) into dataclasses."""

    def merge(dc_cls, values):
        if values is None:
            return dc_cls()
        kwargs = {}
        for field_name in dc_cls.__dataclass_fields__:
            val = values.get(field_name)
            field_type = dc_cls.__dataclass_fields__[field_name].type
            if hasattr(field_type, "__mro__") and hasattr(field_type, "__dataclass_fields__"):
                kwargs[field_name] = merge(field_type, val)
            else:
                kwargs[field_name] = val if val is not None else getattr(dc_cls(), field_name)
        return dc_cls(**kwargs)  # type: ignore[arg-type]

    return MaNFAConfig(
        data=merge(DataConfig, cfg_dict.get("data", {})),
        field=merge(FieldConfig, cfg_dict.get("field", {})),
        encoder=merge(EncoderConfig, cfg_dict.get("encoder", {})),
        flow=merge(ParticleFlowConfig, cfg_dict.get("flow", {})),
        optim=merge(OptimConfig, cfg_dict.get("optim", {})),
        loss=merge(LossConfig, cfg_dict.get("loss", {})),
        device=cfg_dict.get("device"),
        seed=cfg_dict.get("seed", 17),
    )
