from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .quality import denormalize_coords, run_quality_checks


def load_split_arrays(root: Path, split: str) -> Tuple[np.ndarray, ...]:
    states = np.load(root / f"states_{split}.npy")
    coords = np.load(root / f"coords_{split}.npy")
    trajectories = np.load(root / f"trajectories_{split}.npy")
    labels = np.load(root / f"labels_{split}.npy")
    with open(root / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return states, coords, trajectories, labels, metadata


class WindowDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        variance_thresh: float,
        yaw_flip_thresh_deg: float,
        arc_over_net_tol: float,
        cache_clean_indices: bool = True,
        filter_quality: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        variance_thresh = float(variance_thresh)
        yaw_flip_thresh_deg = float(yaw_flip_thresh_deg)
        arc_over_net_tol = float(arc_over_net_tol)
        (
            states_np,
            coords_np,
            traj_np,
            labels_np,
            metadata,
        ) = load_split_arrays(self.root, split)

        self.metadata = metadata
        self.norm_info: Dict[str, Dict[str, float]] = metadata["normalization"]

        keep_indices = np.arange(len(states_np))
        self.metrics: Dict[int, Dict[str, float]] = {}

        cache_path = self.root / f"clean_indices_{split}.json"
        if filter_quality:
            if cache_clean_indices and cache_path.exists():
                keep_indices = np.array(json.loads(cache_path.read_text()), dtype=np.int64)
            else:
                mask = []
                for idx in range(len(states_np)):
                    ok, metrics = run_quality_checks(
                        states_np[idx],
                        traj_np[idx],
                        self.norm_info,
                        variance_thresh,
                        yaw_flip_thresh_deg,
                        arc_over_net_tol,
                    )
                    if ok:
                        mask.append(idx)
                        self.metrics[int(len(mask) - 1)] = metrics
                keep_indices = np.array(mask, dtype=np.int64)
                if cache_clean_indices:
                    cache_path.write_text(json.dumps(keep_indices.tolist()))

            states_np = states_np[keep_indices]
            coords_np = coords_np[keep_indices]
            traj_np = traj_np[keep_indices]
            labels_np = labels_np[keep_indices]

        self.states = torch.from_numpy(states_np).float()
        self.coords = torch.from_numpy(coords_np).float()
        self.trajectories = torch.from_numpy(traj_np).float()
        self.labels = torch.from_numpy(labels_np).long()

        self.coords_real = torch.from_numpy(
            denormalize_coords(
                coords_np,
                self.norm_info["x_min"],
                self.norm_info["x_max"],
                self.norm_info["y_min"],
                self.norm_info["y_max"],
            )
        ).float()
        self.trajectory_real = torch.from_numpy(
            denormalize_coords(
                traj_np,
                self.norm_info["x_min"],
                self.norm_info["x_max"],
                self.norm_info["y_min"],
                self.norm_info["y_max"],
            )
        ).float()

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "states": self.states[idx],
            "coords": self.coords[idx],
            "coords_real": self.coords_real[idx],
            "trajectory": self.trajectories[idx],
            "trajectory_real": self.trajectory_real[idx],
            "label": self.labels[idx],
        }

    @property
    def window_size(self) -> int:
        return int(self.states.shape[1])

    @property
    def sensor_dim(self) -> int:
        return int(self.states.shape[2])
