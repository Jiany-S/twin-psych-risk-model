"""Temporal Fusion Transformer utilities."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from ..data.schema import DataSchema


def build_tft_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    schema: DataSchema,
    max_encoder_length: int,
    max_prediction_length: int,
    additional_time_varying_known_categoricals: Sequence[str] | None = None,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Construct PyTorch Forecasting TimeSeriesDataSet objects."""

    time_varying_known_categoricals = list(additional_time_varying_known_categoricals or [])
    training = TimeSeriesDataSet(
        train_df,
        time_idx=schema.time_idx,
        target=schema.target,
        group_ids=[schema.worker_id],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=list(schema.static_categorical),
        static_reals=list(schema.static_real),
        time_varying_known_reals=list(schema.known_covariates),
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_reals=list(schema.observed_covariates),
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=False,
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
    return training, validation


def create_tft_model(
    training_dataset: TimeSeriesDataSet,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    lstm_layers: int = 1,
    dropout: float = 0.1,
) -> TemporalFusionTransformer:
    """Instantiate a Temporal Fusion Transformer from a dataset definition."""
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        attention_head_size=4,
        loss=QuantileLoss(quantiles=[0.5]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return tft
