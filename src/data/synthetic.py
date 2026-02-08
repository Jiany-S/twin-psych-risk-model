"""Synthetic dataset generator."""

from __future__ import annotations

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
        base_ecg = rng.normal(0.2 * worker_id, 0.2)
        base_eda = rng.normal(2.0 + 0.1 * worker_id, 0.4)
        base_temp = rng.normal(33.0 + 0.1 * worker_id, 0.2)
        specialization = int(rng.integers(0, 4))
        experience_level = int(rng.integers(1, 6))
        stress_state = 0.0
        for t in range(num_timesteps):
            stress_state = 0.88 * stress_state + 0.12 * rng.normal(0.0, 0.5)
            ecg = base_ecg + 1.4 * stress_state + rng.normal(0, 0.1)
            eda = base_eda + 1.8 * max(stress_state, -0.5) + rng.normal(0, 0.2)
            temp = base_temp - 0.4 * stress_state + rng.normal(0, 0.05)
            resp = 0.25 + 0.2 * stress_state + rng.normal(0, 0.05)

            distance = max(0.2, rng.normal(4.0 - 1.0 * stress_state, 0.6))
            speed = max(0.1, rng.normal(1.0 + 0.5 * stress_state, 0.25))
            hazard_zone = int(distance < 2.0)
            task_phase = "critical" if hazard_zone else "normal"
            risk_score = 1 / (1 + np.exp(-(0.6 * stress_state + 0.2 * speed + 0.2 / (distance + 0.3))))
            y_stress = float(risk_score > 0.55)
            y_comfort = float(np.clip(1.0 - 0.8 * risk_score, 0.0, 1.0))

            rows.append(
                {
                    schema.timestamp: float(t),
                    schema.time_idx: t,
                    schema.worker_id: str(worker_id),
                    "ecg": float(ecg),
                    "eda": float(eda),
                    "temp": float(temp),
                    "resp": float(resp),
                    "distance_to_robot": float(distance),
                    "robot_speed": float(speed),
                    schema.hazard_zone: hazard_zone,
                    schema.task_phase: task_phase,
                    schema.stress_target: y_stress,
                    schema.comfort_target: y_comfort,
                    schema.protocol_label: "stress" if y_stress > 0.5 else "baseline",
                    schema.specialization_col: specialization,
                    schema.experience_col: experience_level,
                }
            )

    df = pd.DataFrame(rows)
    if missing_rate > 0:
        for col in set(list(schema.physiology) + list(schema.robot_context)):
            if col in df.columns:
                mask = rng.random(len(df)) < missing_rate
                df.loc[mask, col] = np.nan
    return df
