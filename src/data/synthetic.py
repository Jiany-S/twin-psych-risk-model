"""Synthetic dataset generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .schema import DataSchema


def generate_synthetic_dataset(cfg: Mapping[str, object], schema: DataSchema) -> pd.DataFrame:
    synth_cfg = cfg.get("synthetic", {})
    rng = np.random.default_rng(int(synth_cfg.get("seed", 123)))
    num_workers = int(synth_cfg.get("num_workers", 8))
    num_timesteps = int(synth_cfg.get("num_timesteps", 300))
    missing_rate = float(synth_cfg.get("missing_rate", 0.02))

    rows = []
    for worker_id in range(num_workers):
        base_hr = rng.normal(75 + worker_id * 2, 5)
        base_hrv = rng.normal(45 - worker_id * 1.5, 3)
        base_gsr = rng.normal(0.5 + worker_id * 0.03, 0.05)
        specialization = int(rng.integers(0, 4))
        experience_level = int(rng.integers(1, 6))

        stress_state = 0.0
        for t in range(num_timesteps):
            stress_state = 0.85 * stress_state + 0.15 * rng.normal(0.0, 0.4)
            hr = base_hr + 10 * stress_state + rng.normal(0, 1.5)
            hrv = base_hrv - 6 * stress_state + rng.normal(0, 1.2)
            gsr = base_gsr + 0.25 * stress_state + rng.normal(0, 0.05)

            distance = max(0.2, rng.normal(4.0 - 1.5 * stress_state, 0.5))
            speed = max(0.1, rng.normal(1.0 + 0.6 * stress_state, 0.2))
            hazard_zone = int(distance < 2.0)
            task_phase = "critical" if hazard_zone else "normal"

            risk_score = 0.4 * stress_state + 0.3 * (1 / (distance + 0.2)) + 0.3 * speed
            risk_score = 1 / (1 + np.exp(-risk_score))

            rows.append(
                {
                    schema.time_idx: t,
                    schema.worker_id: str(worker_id),
                    schema.target: float(risk_score),
                    "hr": float(hr),
                    "hrv_rmssd": float(hrv),
                    "gsr": float(gsr),
                    "distance_to_robot": float(distance),
                    "robot_speed": float(speed),
                    schema.hazard_zone: hazard_zone,
                    schema.task_phase: task_phase,
                    "specialization_id": specialization,
                    "experience_level": experience_level,
                }
            )

    df = pd.DataFrame(rows)

    if missing_rate > 0:
        mask = rng.random(df.shape) < missing_rate
        for col in schema.physiology + schema.robot_context:
            df.loc[mask[:, df.columns.get_loc(col)], col] = np.nan

    return df
