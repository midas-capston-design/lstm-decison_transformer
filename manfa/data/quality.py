from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def denormalize_states(states: np.ndarray, sensor_mean: np.ndarray, sensor_std: np.ndarray) -> np.ndarray:
    return states * sensor_std[None, None, :] + sensor_mean[None, None, :]


def denormalize_coords(
    coords: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    coords_real = np.empty_like(coords)
    coords_real[..., 0] = ((coords[..., 0] + 1) * 0.5) * (x_max - x_min) + x_min
    coords_real[..., 1] = ((coords[..., 1] + 1) * 0.5) * (y_max - y_min) + y_min
    return coords_real


def trajectory_arc_ratio(traj_xy: np.ndarray) -> float:
    diffs = np.diff(traj_xy, axis=0)
    arc = np.linalg.norm(diffs, axis=-1).sum()
    net = np.linalg.norm(traj_xy[-1] - traj_xy[0]) + 1e-6
    if net < 1e-3:
        return float("inf") if arc > 1e-3 else 1.0
    return float(arc / net)


def heading_magnetic_corr(states_raw: np.ndarray) -> float:
    yaw = states_raw[:, 5]
    yaw_rad = np.unwrap(np.deg2rad(yaw))
    dyaw = np.diff(yaw_rad)

    mag_x, mag_y = states_raw[:, 0], states_raw[:, 1]
    mag_phase = np.unwrap(np.arctan2(mag_y, mag_x))
    dphase = np.diff(mag_phase)

    dyaw = dyaw.reshape(-1)
    dphase = dphase.reshape(-1)
    if dyaw.size == 0 or dphase.size == 0:
        return 1.0

    dyaw -= dyaw.mean()
    dphase -= dphase.mean()
    denom = np.linalg.norm(dyaw) * np.linalg.norm(dphase)
    if denom < 1e-6:
        return 0.0
    return float(np.dot(dyaw, dphase) / denom)


def magnet_variance(states_raw: np.ndarray) -> float:
    mag = states_raw[:, :3]
    return float(np.var(mag))


def run_quality_checks(
    states_norm: np.ndarray,
    traj_norm: np.ndarray,
    metadata: Dict[str, Dict[str, float]],
    variance_thresh: float,
    yaw_flip_thresh_deg: float,
    arc_over_net_tol: float,
) -> Tuple[bool, Dict[str, float]]:
    """Return (is_valid, metrics dict)."""
    sensor_mean = np.asarray(metadata["sensor_mean"])
    sensor_std = np.asarray(metadata["sensor_std"])
    x_min, x_max = metadata["x_min"], metadata["x_max"]
    y_min, y_max = metadata["y_min"], metadata["y_max"]

    states_raw = denormalize_states(states_norm, sensor_mean, sensor_std)
    traj_real = denormalize_coords(traj_norm, x_min, x_max, y_min, y_max)

    arc_ratio = trajectory_arc_ratio(traj_real)
    mag_var = magnet_variance(states_raw)
    yaw_corr = heading_magnetic_corr(states_raw)
    yaw_series = np.asarray(states_raw[:, 5]).reshape(-1)
    yaw_swing = float(abs(yaw_series[-1] - yaw_series[0]))

    metrics = {
        "arc_ratio": arc_ratio,
        "mag_var": mag_var,
        "yaw_corr": yaw_corr,
        "yaw_swing": yaw_swing,
    }

    if arc_ratio > arc_over_net_tol:
        return False, metrics
    if mag_var < variance_thresh:
        return False, metrics
    if yaw_swing > yaw_flip_thresh_deg and yaw_corr < 0.2:
        return False, metrics
    return True, metrics
