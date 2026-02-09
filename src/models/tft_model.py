"""Temporal Fusion Transformer utilities."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


def _require_tft() -> Any:
    try:
        import pytorch_forecasting
        import pytorch_lightning
    except ImportError as exc:
        raise ImportError(
            "pytorch-forecasting and pytorch-lightning are required. Install via requirements.txt."
        ) from exc
    return pytorch_forecasting


def build_tft_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    schema,
    target_col: str,
    window_length: int,
    horizon: int,
    use_profiles: bool,
    known_categoricals: Sequence[str] | None = None,
) -> tuple[Any, Any]:
    pf = _require_tft()
    TimeSeriesDataSet = pf.TimeSeriesDataSet

    min_encoder_length = max(2, window_length // 2)
    static_reals = []
    if use_profiles:
        static_reals = [f"baseline_mu_{f}" for f in schema.physiology] + [f"baseline_sigma_{f}" for f in schema.physiology]
    categorical_encoders = {
        "worker_id": pf.data.encoders.NaNLabelEncoder(add_nan=True),
        "specialization_index": pf.data.encoders.NaNLabelEncoder(add_nan=True),
        "experience_level": pf.data.encoders.NaNLabelEncoder(add_nan=True),
        "task_phase": pf.data.encoders.NaNLabelEncoder(add_nan=True),
    }

    training = TimeSeriesDataSet(
        train_df,
        time_idx=schema.time_idx,
        target=target_col,
        group_ids=[schema.worker_id],
        max_encoder_length=window_length,
        min_encoder_length=min_encoder_length,
        max_prediction_length=horizon,
        min_prediction_length=horizon,
        static_categoricals=["worker_id", "specialization_index", "experience_level"],
        static_reals=static_reals,
        time_varying_known_reals=list(schema.robot_context) + [schema.hazard_zone],
        time_varying_known_categoricals=list(known_categoricals or ["task_phase"]),
        time_varying_unknown_reals=list(schema.physiology),
        add_relative_time_idx=True,
        add_target_scales=False,
        target_normalizer=None,
        categorical_encoders=categorical_encoders,
    )
    # Validation should be created with predict=False for proper loss/early stopping.
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    return training, validation


def create_tft_model(training_dataset: Any, cfg: dict[str, Any]) -> Any:
    pf = _require_tft()
    TemporalFusionTransformer = pf.TemporalFusionTransformer
    QuantileLoss = pf.metrics.QuantileLoss
    loss_mode = cfg.get("tft_loss", "quantile")
    if loss_mode == "bce":
        try:
            import torch

            loss_fn = torch.nn.BCEWithLogitsLoss()
        except Exception:
            loss_fn = QuantileLoss(quantiles=[0.5])
            loss_mode = "quantile"
    else:
        loss_fn = QuantileLoss(quantiles=[0.5])
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=cfg.get("learning_rate", 1e-3),
        hidden_size=cfg.get("hidden_size", 32),
        lstm_layers=cfg.get("lstm_layers", 1),
        dropout=cfg.get("dropout", 0.1),
        attention_head_size=4,
        loss=loss_fn,
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
