from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_multiphysio import load_multiphysio_dataset
from src.data.schema import DataSchema
from src.data.windowing import engineer_window_features
from src.data.load_wesad import _load_wesad_csvs


def _write_fixture(root, labels: pd.DataFrame | None = None) -> None:
    features = root / "features"
    features.mkdir(parents=True)
    bio = pd.DataFrame(
        {
            "ID": ["P1", "P1", "P1"],
            "Class": ["rest", "rest", "rest"],
            "Repetition": [0, 0, 0],
            "Window": [0, 1, 2],
            "HRV_MeanNN": [700.0, 710.0, 720.0],
            "EDA_mean": [1.0, 1.2, 1.4],
            "EMG_RMSE": [10.0, 11.0, 12.0],
            "RRV_MeanBB": [2.0, 2.1, 2.2],
        }
    )
    bio.to_csv(features / "bio_features_60s.csv", index=False)
    if labels is None:
        labels = pd.DataFrame(
            {
                "ID": ["P1"],
                "Class": ["rest"],
                "Repetition": [0],
                "STAI": [50],
                "NASA": [45.0],
                "Valence": [4],
                "Arousal": [2],
                "Dominance": [3],
                "Experience": [1],
            }
        )
    labels.to_csv(features / "labels.csv", index=False)
    pd.DataFrame({"ID": ["P1"], "rest_0": ["x"]}).to_csv(root / "participants_task_overview.csv", index=False)


def _cfg(primary: str = "cognitive_load_binary") -> dict:
    return {
        "experiment": {"primary_target": primary},
        "targets": {
            "multi_head": {"enabled": True},
            "stress": {
                "label_col": "y_stress",
                "source_col": "STAI",
                "source_questionnaire": "STAI-Y1",
                "source_range": [20, 80],
                "task_type": "regression",
                "threshold": None,
            },
            "cognitive_load": {
                "label_col": "y_cognitive_load",
                "source_col": "NASA",
                "source_questionnaire": "NASA-TLX",
                "source_range": [0, 100],
                "task_type": "regression",
                "threshold": None,
            },
            "cognitive_load_binary": {
                "label_col": "y_cognitive_load_binary",
                "source_col": "NASA",
                "source_questionnaire": "NASA-TLX",
                "source_range": [0, 100],
                "task_type": "classification",
                "threshold": 40.0,
            },
            "comfort": {
                "label_col": "y_comfort_proxy",
                "source_col": "Valence",
                "source_questionnaire": "SAM Valence",
                "source_range": [1, 5],
                "task_type": "regression",
            },
            "arousal": {
                "label_col": "y_arousal",
                "source_col": "Arousal",
                "source_questionnaire": "SAM Arousal",
                "source_range": [1, 5],
                "task_type": "regression",
            },
        },
        "features": {
            "timestamp": "timestamp",
            "time_idx": "time_idx",
            "worker_id": "worker_id",
            "protocol_label": "protocol_label",
            "physiology": ["hrv_mean_nn", "eda_mean", "emg_rmse", "rrv_mean_bb"],
            "engineering_mode": "precomputed",
            "use_robot_context": False,
            "optional": {
                "hazard_zone": "hazard_zone",
                "task_phase": "task_phase",
                "specialization_col": "specialization_index",
                "experience_col": "experience_level",
            },
        },
    }


def _dataset_cfg() -> dict:
    return {
        "feature_columns": {
            "hrv_mean_nn": "HRV_MeanNN",
            "eda_mean": "EDA_mean",
            "emg_rmse": "EMG_RMSE",
            "rrv_mean_bb": "RRV_MeanBB",
        },
        "repetition_offset": 1,
        "min_rows_per_worker": 1,
    }


def test_multiphysio_targets_and_modalities_are_explicit(tmp_path) -> None:
    _write_fixture(tmp_path)
    cfg = _cfg()
    schema = DataSchema.from_config(cfg)
    df = load_multiphysio_dataset(tmp_path, schema, _dataset_cfg(), cfg["targets"])

    assert set(["y_stress", "y_cognitive_load", "y_cognitive_load_binary", "y_comfort_proxy", "y_arousal"]).issubset(df.columns)
    assert float(df["y_stress"].iloc[0]) == pytest.approx((50 - 20) / 60)
    assert float(df["y_cognitive_load"].iloc[0]) == pytest.approx(0.45)
    assert float(df["y_cognitive_load_binary"].iloc[0]) == 1.0
    assert float(df["y_comfort_proxy"].iloc[0]) == pytest.approx(0.75)
    assert float(df["y_arousal"].iloc[0]) == pytest.approx(0.25)
    assert "emg_rmse" in df.columns
    assert "temp" not in df.columns
    assert df.attrs["target_metadata"]["cognitive_load_binary"]["source_column"] == "NASA"
    assert df.attrs["target_metadata"]["stress"]["source_column"] == "STAI"


def test_missing_stai_does_not_fall_back_to_nasa(tmp_path) -> None:
    labels = pd.DataFrame(
        {
            "ID": ["P1"],
            "Class": ["rest"],
            "Repetition": [0],
            "NASA": [45.0],
            "Valence": [4],
            "Arousal": [2],
        }
    )
    _write_fixture(tmp_path, labels=labels)
    cfg = _cfg()
    schema = DataSchema.from_config(cfg)
    with pytest.raises(ValueError, match="STAI"):
        load_multiphysio_dataset(tmp_path, schema, _dataset_cfg(), cfg["targets"])


def test_binary_threshold_requires_explicit_configuration(tmp_path) -> None:
    _write_fixture(tmp_path)
    cfg = _cfg()
    cfg["targets"]["cognitive_load_binary"] = dict(cfg["targets"]["cognitive_load_binary"])
    cfg["targets"]["cognitive_load_binary"]["threshold"] = None
    schema = DataSchema.from_config(cfg)
    with pytest.raises(ValueError, match="requires an explicit threshold"):
        load_multiphysio_dataset(tmp_path, schema, _dataset_cfg(), cfg["targets"])


def test_continuous_target_does_not_create_binary_without_binary_target(tmp_path) -> None:
    _write_fixture(tmp_path)
    cfg = _cfg(primary="cognitive_load")
    cfg["targets"].pop("cognitive_load_binary")
    schema = DataSchema.from_config(cfg)
    df = load_multiphysio_dataset(tmp_path, schema, _dataset_cfg(), cfg["targets"])
    assert "y_cognitive_load" in df.columns
    assert "y_cognitive_load_binary" not in df.columns


def test_invalid_emg_to_temperature_mapping_fails(tmp_path) -> None:
    _write_fixture(tmp_path)
    cfg = _cfg()
    cfg["features"]["physiology"] = ["temp"]
    schema = DataSchema.from_config(cfg)
    bad_dataset_cfg = {"feature_columns": {"temp": "EMG_RMSE"}, "min_rows_per_worker": 1}
    with pytest.raises(ValueError, match="EMG cannot be mapped to temperature"):
        load_multiphysio_dataset(tmp_path, schema, bad_dataset_cfg, cfg["targets"])


def test_precomputed_features_do_not_call_raw_peak_detection(monkeypatch) -> None:
    cfg = _cfg()
    schema = DataSchema.from_config(cfg)

    def fail_find_peaks(*args, **kwargs):
        raise AssertionError("raw peak detection should not run for precomputed features")

    monkeypatch.setattr("src.data.features.find_peaks", fail_find_peaks)
    windows = pd.DataFrame(
        {
            "hrv_mean_nn": [700.0, 710.0, 720.0],
            "eda_mean": [1.0, 1.2, 1.4],
            "emg_rmse": [10.0, 11.0, 12.0],
            "rrv_mean_bb": [2.0, 2.1, 2.2],
        }
    ).to_numpy(dtype="float32")[None, :, :]
    X, names = engineer_window_features(windows, schema, 1 / 60, False, 0.05, 3)
    assert X.shape[0] == 1
    assert "emg_rmse_mean" in names
    assert not any(name.startswith("temp_") for name in names)


def test_wesad_csv_still_exposes_temperature(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "worker_id": ["S1", "S1"],
            "time_idx": [0, 1],
            "timestamp": [0.0, 1.0],
            "protocol_label": [1, 2],
            "ecg": [0.1, 0.2],
            "eda": [1.0, 1.1],
            "temp": [32.1, 32.2],
        }
    )
    frame.to_csv(tmp_path / "S1.csv", index=False)
    cfg = _cfg(primary="stress")
    cfg["features"]["physiology"] = ["ecg", "eda", "temp"]
    cfg["features"]["engineering_mode"] = "raw_signals"
    schema = DataSchema.from_config(cfg)
    df = _load_wesad_csvs(tmp_path, schema, stress_include_amusement=True)
    assert "temp" in df.columns
    assert list(df["temp"]) == [32.1, 32.2]
