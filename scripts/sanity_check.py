"""Lightweight regression checks on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import DataSchema  # noqa: E402
from src.data.synthetic import generate_synthetic_dataset  # noqa: E402
from src.data.preprocess import preprocess_dataframe  # noqa: E402
from src.data.windowing import build_windows, create_time_splits, engineer_window_features  # noqa: E402
from src.training.metrics import classification_metrics  # noqa: E402


def main() -> None:
    cfg = {
        "features": {
            "timestamp": "timestamp",
            "time_idx": "time_idx",
            "worker_id": "worker_id",
            "protocol_label": "protocol_label",
            "physiology": ["ecg", "eda", "temp"],
            "robot_context": ["distance_to_robot", "robot_speed"],
            "use_robot_context": True,
            "optional": {
                "hazard_zone": "hazard_zone",
                "task_phase": "task_phase",
                "specialization_col": "specialization_index",
                "experience_col": "experience_level",
            },
        },
        "targets": {"stress": {"label_col": "y_stress"}, "comfort": {"label_col": "y_comfort_proxy"}},
        "synthetic": {"num_workers": 4, "num_timesteps": 400, "missing_rate": 0.0, "seed": 123},
        "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
    }
    schema = DataSchema.from_config(cfg)
    df = generate_synthetic_dataset(cfg, schema)
    df = preprocess_dataframe(cfg, df, schema)
    train_df, val_df, test_df, _ = create_time_splits(df, schema, 0.7, 0.15, 0.15)
    train_w = build_windows(train_df, schema, window_length=30, horizon_steps=3, window_step=1)
    test_w = build_windows(test_df, schema, window_length=30, horizon_steps=3, window_step=1)

    X_train, _ = engineer_window_features(train_w.X_windows, schema, sampling_rate_hz=4.0, include_freq_domain=False, scr_threshold=0.05, min_scr_distance=3)
    X_test, _ = engineer_window_features(test_w.X_windows, schema, sampling_rate_hz=4.0, include_freq_domain=False, scr_threshold=0.05, min_scr_distance=3)
    assert X_train.shape[0] > 100 and X_test.shape[0] > 100, "Not enough windows for sanity check."

    # Simulate predictions to verify metric shapes.
    rng = np.random.default_rng(0)
    preds = rng.random(len(test_w.y_stress))
    metrics = classification_metrics(test_w.y_stress, preds)
    cm = metrics["confusion_matrix_default"]
    assert np.array(cm).shape == (2, 2), "Confusion matrix should be 2x2."
    if metrics["auroc"] != metrics["auroc"]:
        print("Warning: AUROC is NaN; test split may be single-class.")

    out = Path("experiments/sanity_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
