"""Temporal Fusion Transformer utilities."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import torch
from torch import nn


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
    static_reals: Sequence[str] | None = None,
    time_varying_known_reals: Sequence[str] | None = None,
    static_categoricals: Sequence[str] | None = None,
) -> tuple[Any, Any]:
    pf = _require_tft()
    TimeSeriesDataSet = pf.TimeSeriesDataSet

    # Match XGB window semantics: require a full encoder window for each prediction.
    min_encoder_length = int(window_length)
    static_real_cols = list(static_reals or [])
    time_varying_known_real_cols = list(time_varying_known_reals or [])
    static_categorical_cols = list(static_categoricals or [])
    assert_no_worker_static_embedding(schema.worker_id, static_categorical_cols)
    categorical_encoders = {
        "task_phase": pf.data.encoders.NaNLabelEncoder(add_nan=True),
    }
    for col in static_categorical_cols:
        categorical_encoders[col] = pf.data.encoders.NaNLabelEncoder(add_nan=True)

    training = TimeSeriesDataSet(
        train_df,
        time_idx=schema.time_idx,
        target=target_col,
        group_ids=[schema.worker_id],
        max_encoder_length=window_length,
        min_encoder_length=min_encoder_length,
        max_prediction_length=horizon,
        min_prediction_length=horizon,
        min_prediction_idx=0,
        static_categoricals=static_categorical_cols,
        static_reals=static_real_cols,
        time_varying_known_reals=list(schema.robot_context) + [schema.hazard_zone] + time_varying_known_real_cols,
        time_varying_known_categoricals=list(known_categoricals or ["task_phase"]),
        time_varying_unknown_reals=list(schema.physiology),
        add_relative_time_idx=True,
        add_target_scales=False,
        target_normalizer=None,
        allow_missing_timesteps=True,
        categorical_encoders=categorical_encoders,
    )
    # Validation should be created with predict=False for proper loss/early stopping.
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    return training, validation


def assert_no_worker_static_embedding(worker_id_col: str, static_categoricals: Sequence[str]) -> None:
    if str(worker_id_col) in {str(c) for c in static_categoricals}:
        raise ValueError("worker_id must remain a group identifier and must not be a learned static categorical.")


def create_tft_model(training_dataset: Any, cfg: dict[str, Any]) -> Any:
    pf = _require_tft()
    TemporalFusionTransformer = pf.TemporalFusionTransformer
    loss_fn = resolve_tft_loss(cfg, task_type=str(cfg.get("task_type", "classification")))
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


def resolve_tft_loss(cfg: dict[str, Any], task_type: str) -> Any:
    """Resolve TFT loss without silent classification fallback."""
    pf = _require_tft()
    loss_mode = str(cfg.get("loss", cfg.get("tft_loss", "bce" if task_type == "classification" else "quantile"))).lower()
    if task_type == "classification":
        if loss_mode != "bce":
            raise ValueError("Binary TFT classification requires loss='bce'; quantile loss is invalid.")
        return torch.nn.BCEWithLogitsLoss()
    if loss_mode == "quantile":
        return pf.metrics.QuantileLoss(quantiles=cfg.get("quantiles", [0.5]))
    if loss_mode in {"mse", "regression"}:
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported TFT loss for task_type={task_type}: {loss_mode}")


class SlowTFTForecaster(nn.Module):
    """Compact multi-horizon neural forecaster used by the slow TFT smoke runner.

    The public experiment treats this as the slow TFT-family path: it consumes a
    full causal encoder context and emits one logit per configured horizon.
    """

    def __init__(
        self,
        input_channels: int,
        n_horizons: int,
        hidden_size: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_channels, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.horizon_embedding = nn.Embedding(n_horizons, hidden_size)
        self.head = nn.Linear(hidden_size * 2, 1)
        self.n_horizons = int(n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, channels, time]
        encoded, _ = self.encoder(x.transpose(1, 2))
        context = self.dropout(encoded[:, -1, :])
        horizon_ids = torch.arange(self.n_horizons, device=x.device)
        horizon = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(x.shape[0], -1, -1)
        repeated_context = context.unsqueeze(1).expand(-1, self.n_horizons, -1)
        logits = self.head(torch.cat([repeated_context, horizon], dim=-1)).squeeze(-1)
        return logits


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_size_bytes(model: nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))
