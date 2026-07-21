from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess_dataframe
from src.data.schema import DataSchema
from src.models import tft_model
from src.profiles.worker_profile import (
    CalibrationPolicy,
    fit_calibration_table,
    fit_global_normalization,
)
from src.training import tft_train
from src.training.xgb_train import _append_static_features


def _schema() -> DataSchema:
    return DataSchema.from_config(
        {
            "features": {
                "physiology": ["ecg", "eda", "temp"],
                "use_robot_context": False,
            },
            "targets": {
                "stress": {"label_col": "y_stress", "task_type": "classification"},
                "comfort": {"label_col": "y_comfort_proxy", "task_type": "regression"},
            },
        }
    )


def _frame(worker: str = "S1", stress_value: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "worker_id": [worker] * 5,
            "timestamp": np.arange(5, dtype=float),
            "time_idx": np.arange(5),
            "protocol_label": ["baseline", "baseline", "task", "task", "task"],
            "task_phase": ["rest", "rest", "stress", "stress", "stress"],
            "ecg": [1.0, 3.0, 100.0, 110.0, 120.0],
            "eda": [2.0, 4.0, 200.0, 210.0, 220.0],
            "temp": [30.0, 32.0, 40.0, 41.0, 42.0],
            "y_stress": [stress_value] * 5,
            "y_comfort_proxy": [1.0] * 5,
        }
    )


def test_xgb_profile_on_and_off_feature_matrices_differ() -> None:
    X = np.ones((4, 2), dtype=float)
    meta = pd.DataFrame({"worker_id": ["A", "A", "B", "B"]})
    static = pd.DataFrame(
        {
            "worker_id": ["A", "B"],
            "calib_ecg_mean": [1.0, 2.0],
            "calib_ecg_std": [0.5, 0.7],
        }
    )

    X_off, names_off = _append_static_features(X, meta, static, use_profiles=False)
    X_on, names_on = _append_static_features(X, meta, static, use_profiles=True)

    assert X_off.shape == (4, 2)
    assert names_off == []
    assert X_on.shape == (4, 4)
    assert names_on == ["calib_ecg_mean", "calib_ecg_std"]
    assert not np.array_equal(X_on[:, :2], X_on[:, 2:])


def test_calibration_stats_use_only_allowed_baseline_rows() -> None:
    schema = _schema()
    train = _frame()
    global_stats = fit_global_normalization(train, schema.physiology)
    table, diagnostics = fit_calibration_table(
        train,
        schema.worker_id,
        schema.protocol_label,
        schema.task_phase,
        schema.physiology,
        CalibrationPolicy(protocol_labels=("baseline",), task_phases=()),
        global_stats,
    )

    assert diagnostics["subjects"]["S1"]["n_calibration_rows"] == 2
    assert float(table.loc[0, "calib_ecg_mean"]) == pytest.approx(2.0)
    assert float(table.loc[0, "calib_eda_mean"]) == pytest.approx(3.0)
    assert float(table.loc[0, "calib_temp_mean"]) == pytest.approx(31.0)


def test_heldout_calibration_does_not_depend_on_target_labels() -> None:
    schema = _schema()
    fallback = fit_global_normalization(_frame("TRAIN"), schema.physiology)
    heldout_a = _frame("HELDOUT", stress_value=0.0)
    heldout_b = _frame("HELDOUT", stress_value=1.0)
    policy = CalibrationPolicy(protocol_labels=("baseline",), task_phases=())

    table_a, _ = fit_calibration_table(
        heldout_a, schema.worker_id, schema.protocol_label, schema.task_phase, schema.physiology, policy, fallback
    )
    table_b, _ = fit_calibration_table(
        heldout_b, schema.worker_id, schema.protocol_label, schema.task_phase, schema.physiology, policy, fallback
    )

    pd.testing.assert_frame_equal(table_a, table_b)


def test_missing_real_metadata_uses_explicit_unknown_values() -> None:
    schema = _schema()
    frame = preprocess_dataframe({}, _frame(), schema)

    assert set(frame[schema.specialization_col].unique()) == {-1}
    assert set(frame[schema.experience_col].unique()) == {0}


def test_tft_worker_id_is_group_only_not_static_categorical(monkeypatch) -> None:
    captured = {}

    class FakeEncoder:
        def __init__(self, add_nan=True):
            self.add_nan = add_nan

    class FakeDataSet:
        def __init__(self, df, **kwargs):
            captured.update(kwargs)

        @classmethod
        def from_dataset(cls, training, val_df, predict=False, stop_randomization=True):
            return cls(val_df, predict=predict, stop_randomization=stop_randomization)

    fake_pf = SimpleNamespace(
        TimeSeriesDataSet=FakeDataSet,
        data=SimpleNamespace(encoders=SimpleNamespace(NaNLabelEncoder=FakeEncoder)),
    )
    monkeypatch.setattr(tft_model, "_require_tft", lambda: fake_pf)
    schema = _schema()
    train = _frame("TRAIN")
    val = _frame("VAL")

    tft_model.build_tft_datasets(
        train,
        val,
        schema,
        target_col=schema.stress_target,
        window_length=2,
        horizon=1,
        use_profiles=True,
        static_reals=["calib_ecg_mean"],
        static_categoricals=[],
    )

    assert captured["group_ids"] == [schema.worker_id]
    assert schema.worker_id not in captured["static_categoricals"]
    assert "calib_ecg_mean" in captured["static_reals"]


def test_tft_training_does_not_insert_test_worker_ids(monkeypatch, tmp_path) -> None:
    captured_train_ids = set()

    def sentinel_build(train_df, val_df, **kwargs):
        captured_train_ids.update(train_df["worker_id"].astype(str).unique())
        raise RuntimeError("sentinel build")

    monkeypatch.setattr(tft_train, "build_tft_datasets", sentinel_build)
    schema = _schema()
    cfg = {
        "debug": False,
        "reproducibility": {"seed": 1},
        "tft": {"gpus": 0},
        "profiles": {"tft_static_real_cols": []},
    }

    with pytest.raises(RuntimeError, match="sentinel build"):
        tft_train.train_tft_task(
            cfg,
            train_df=_frame("TRAIN"),
            val_df=_frame("VAL"),
            test_df=_frame("TEST"),
            schema=schema,
            target_col=schema.stress_target,
            task_type="classification",
            run_dir=Path(tmp_path),
            window_length=2,
            horizon=1,
            window_step=1,
            model_name="tft_stress",
            use_profiles=False,
        )

    assert captured_train_ids == {"TRAIN"}
