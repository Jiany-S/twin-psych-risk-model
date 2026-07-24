from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.safety.adapters import FastDetectorAdapter, SlowForecasterAdapter
from src.safety.physical_kernel import PhysicalSafetyKernel, SafetyState
from src.safety.state_machine import SafetyStateMachine
from src.streaming.prediction_artifacts import load_prediction_artifacts_from_manifest, validate_prediction_contract
from src.streaming.replay import ReplayEngine, _load_config, build_events


def _physical_sample(timestamp: float = 0.0, **overrides):
    row = {
        "timestamp": timestamp,
        "sensor_timestamp": timestamp,
        "human_robot_distance_m": 5.0,
        "human_velocity_mps": 0.0,
        "robot_velocity_mps": 0.0,
        "relative_closing_speed_mps": 0.0,
        "robot_stopping_distance_m": 0.5,
        "configured_separation_margin_m": 1.0,
        "hazard_zone_active": False,
        "emergency_input": False,
        "protective_stop_input": False,
    }
    row.update(overrides)
    return row


def _kernel():
    return PhysicalSafetyKernel(
        {
            "minimum_separation_m": 1.5,
            "caution_separation_m": 2.5,
            "critical_ttc_seconds": 1.0,
            "max_sensor_age_seconds": 0.25,
            "invalid_input_policy": "degraded",
        }
    )


def _write_prediction_pair(tmp_path, *, target_name="stress", positive_class="stress", subjects=("S8",), slow_shift=0.0):
    fast_run = tmp_path / "fast"
    slow_run = tmp_path / "slow"
    fast_run.mkdir()
    slow_run.mkdir()
    base_cfg = {
        "experiment": {"primary_target": target_name},
        "targets": {"stress": {"positive_class": positive_class}},
        "stream": {"representation": "raw_signal"},
        "split": {"train_subjects": ["S2"], "validation_subjects": ["S6"], "test_subjects": list(subjects)},
        "fast_model": {"context_seconds": 3.0, "forecast_horizon_seconds": 0.0, "inference_stride_seconds": 0.25},
        "slow_model": {"context_seconds": 30.0, "inference_stride_seconds": 1.0},
    }
    (fast_run / "config_resolved.yaml").write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    (slow_run / "config_resolved.yaml").write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    fast_rows = []
    slow_rows = []
    for subject in subjects:
        for ts in (10.0, 11.0):
            fast_rows.append(
                {
                    "worker_id": subject,
                    "session_id": subject,
                    "prediction_timestamp": ts,
                    "target_timestamp": ts,
                    "predicted_probability": 0.4,
                    "target": 0,
                }
            )
            for horizon in (5.0, 30.0):
                slow_rows.append(
                    {
                        "worker_id": subject,
                        "session_id": subject,
                        "prediction_timestamp": ts + slow_shift,
                        "target_timestamp": ts + slow_shift + horizon,
                        "horizon_seconds": horizon,
                        "model": "xgboost",
                        "uncalibrated_probability": 0.4,
                        "calibrated_probability": 0.4,
                        "target": 0,
                    }
                )
    pd.DataFrame(fast_rows).to_csv(fast_run / "predictions_tcn.csv", index=False)
    pd.DataFrame(slow_rows).to_csv(slow_run / "predictions_long.csv", index=False)
    manifest = {
        "replay_manifest_version": 1,
        "fast": {"run_dir": str(fast_run), "prediction_file": "predictions_tcn.csv", "metadata_file": "config_resolved.yaml", "model_name": "tcn"},
        "slow": {"run_dir": str(slow_run), "prediction_file": "predictions_long.csv", "metadata_file": "config_resolved.yaml", "model_name": "xgboost"},
        "alignment": {
            "subjects": list(subjects),
            "target_name": target_name,
            "positive_class": positive_class,
            "probability_column": "calibrated_probability",
            "timestamp_unit": "seconds",
            "stream_representation": "raw_signal",
            "join_policy": "exact_prediction_timestamp",
            "minimum_overlap_fraction": 0.01,
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest_path, manifest, fast_run, slow_run


def test_emergency_stop_cannot_be_overridden_and_requires_reset_hold():
    sm = SafetyStateMachine({"emergency_latching": True, "emergency_reset_hold_seconds": 1.0, "cooldown_seconds": 0.0})
    phys_emergency = _kernel().evaluate(_physical_sample(emergency_input=True), 0.0)
    fast = FastDetectorAdapter({"consecutive_positive_required": 1}).update({"timestamp": 0.0, "probability": 0.1})
    slow = SlowForecasterAdapter({}).update({"prediction_timestamp": 0.0, "probability_by_horizon": {5.0: 0.1, 30.0: 0.1}}, 0.0)
    out = sm.update(phys_emergency, fast, slow, 0.0)
    assert out.new_state == SafetyState.EMERGENCY_STOP
    normal_phys = _kernel().evaluate(_physical_sample(timestamp=0.5), 0.5)
    rejected = sm.update(normal_phys, fast, slow, 0.5, explicit_reset_command=True)
    assert rejected.new_state == SafetyState.EMERGENCY_STOP
    assert rejected.reason_code == "EMERGENCY_RESET_REJECTED"
    accepted = sm.update(normal_phys, fast, slow, 1.6, explicit_reset_command=True)
    assert accepted.new_state == SafetyState.NORMAL


def test_protective_stop_not_reduced_by_ml_layers():
    sm = SafetyStateMachine({"cooldown_seconds": 0.0})
    phys = _kernel().evaluate(_physical_sample(human_robot_distance_m=1.8, robot_stopping_distance_m=1.0), 0.0)
    fast = FastDetectorAdapter({"consecutive_positive_required": 1}).update({"timestamp": 0.0, "probability": 0.1})
    slow = SlowForecasterAdapter({}).update({"prediction_timestamp": 0.0, "probability_by_horizon": {5.0: 0.1, 30.0: 0.1}}, 0.0)
    out = sm.update(phys, fast, slow, 0.0)
    assert out.new_state == SafetyState.PROTECTIVE_STOP
    assert out.requested_action.action_name == "protective_stop"


def test_physical_kernel_runs_when_ml_fails_and_stale_sensor_degrades():
    result = _kernel().evaluate(_physical_sample(timestamp=2.0, sensor_timestamp=0.0), 2.0)
    assert result.requested_state == SafetyState.DEGRADED
    assert result.reason_code == "PHYSICAL_SENSOR_STALE"


def test_fast_active_when_slow_unavailable_and_hysteresis_prevents_oscillation():
    adapter = FastDetectorAdapter(
        {
            "activation_threshold": 0.7,
            "release_threshold": 0.45,
            "consecutive_positive_required": 2,
            "consecutive_negative_required": 2,
            "cooldown_seconds": 1.0,
        }
    )
    assert adapter.update({"timestamp": 0.0, "probability": 0.72}).persistent_positive is False
    assert adapter.update({"timestamp": 0.25, "probability": 0.71}).persistent_positive is True
    assert adapter.update({"timestamp": 0.5, "probability": 0.68}).persistent_positive is True
    assert adapter.update({"timestamp": 0.75, "probability": 0.46}).persistent_positive is True


def test_fast_persistent_high_can_request_controlled_stop_not_emergency():
    adapter = FastDetectorAdapter(
        {
            "activation_threshold": 0.7,
            "release_threshold": 0.45,
            "consecutive_positive_required": 1,
            "controlled_stop_after_seconds": 1.0,
        }
    )
    assert adapter.update({"timestamp": 0.0, "probability": 0.9}).requested_state == SafetyState.HIGH_ALERT
    result = adapter.update({"timestamp": 1.1, "probability": 0.9})
    assert result.requested_state == SafetyState.CONTROLLED_STOP
    assert result.requested_state != SafetyState.EMERGENCY_STOP


def test_slow_forecast_cannot_directly_issue_emergency():
    adapter = SlowForecasterAdapter({"horizon_5s": {"consecutive_positive_required": 1}})
    result = adapter.update({"prediction_timestamp": 0.0, "probability_by_horizon": {5.0: 0.99, 30.0: 0.99}}, 0.0)
    assert result.requested_state == SafetyState.CAUTION


def test_invalid_ml_probability_cannot_recover_to_normal():
    fast = FastDetectorAdapter({"invalid_probability_policy": "degraded"}).update({"timestamp": 0.0, "probability": float("nan")})
    assert fast.requested_state == SafetyState.DEGRADED
    sm = SafetyStateMachine({"cooldown_seconds": 0.0})
    phys = _kernel().evaluate(_physical_sample(), 0.0)
    out = sm.update(phys, fast, None, 0.0)
    assert out.new_state == SafetyState.DEGRADED


def test_slow_threshold_release_validation_and_action_fields():
    with pytest.raises(ValueError):
        SlowForecasterAdapter({"horizon_5s": {"activation_threshold": 0.5, "release_threshold": 0.6}}).update(
            {"prediction_timestamp": 0.0, "probability_by_horizon": {5.0: 0.7}},
            0.0,
        )
    out = SafetyStateMachine({"cooldown_seconds": 0.0}).update(_kernel().evaluate(_physical_sample(), 0.0), None, None, 0.0)
    assert out.trigger_layer
    assert out.reason_code
    assert out.requested_action.priority >= 0
    assert out.requested_action.source_layer


def test_replay_deterministic_after_timestamp_sorting(tmp_path):
    cfg = _load_config("src/config/streaming_multirate.yaml")
    cfg["paths"]["run_root"] = str(tmp_path)
    events, _ = build_events(cfg)
    shuffled = events.sample(frac=1.0, random_state=42).reset_index(drop=True)
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()
    ReplayEngine(cfg).run(events, run1)
    ReplayEngine(cfg).run(shuffled, run2)
    t1 = pd.read_csv(run1 / "timeline.csv")
    t2 = pd.read_csv(run2 / "timeline.csv")
    pd.testing.assert_series_equal(t1["new_state"], t2["new_state"], check_names=False)
    pd.testing.assert_series_equal(t1["requested_action"], t2["requested_action"], check_names=False)


def test_invalid_probability_is_not_deadline_miss(tmp_path):
    cfg = _load_config("src/config/streaming_multirate.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = pd.DataFrame(
        [
            {"timestamp": 0.0, "worker_id": "A", "session_id": "A", "event_type": "physical", **_physical_sample(0.0)},
            {"timestamp": 0.0, "worker_id": "A", "session_id": "A", "event_type": "fast", "probability": float("nan")},
        ]
    )
    ReplayEngine(cfg).run(events, run_dir)
    latency = __import__("json").loads((run_dir / "latency.json").read_text())
    assert latency["fast_invalid_probability_count"] == 1
    assert latency["fast_deadline_miss_count"] == 0


def test_prediction_artifact_contract_rejects_bad_probabilities_and_duplicates():
    good = pd.DataFrame(
        {
            "worker_id": ["S1"],
            "session_id": ["S1"],
            "prediction_timestamp": [1.0],
            "target_timestamp": [1.0],
            "horizon_seconds": [0.0],
            "model_name": ["tcn"],
            "probability": [0.3],
            "calibrated_probability": [0.3],
            "target": [0],
            "split": ["test"],
            "model_version": ["fast_tcn"],
            "config_hash": ["abc"],
        }
    )
    assert len(validate_prediction_contract(good, expected_test_subjects=["S1"])) == 1
    bad = pd.concat([good, good], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_contract(bad)
    bad_prob = good.copy()
    bad_prob["calibrated_probability"] = 1.2
    with pytest.raises(ValueError, match="outside"):
        validate_prediction_contract(bad_prob)


def test_replay_manifest_accepts_compatible_artifacts(tmp_path):
    manifest_path, _, _, _ = _write_prediction_pair(tmp_path)
    pair = load_prediction_artifacts_from_manifest({"replay": {"manifest_path": str(manifest_path)}})

    assert pair.report["compatibility_decision"] == "accepted"
    assert pair.report["shared_subjects"] == ["S8"]
    assert pair.report["timestamp_coverage"]["aligned_prediction_origins"] == 2


def test_replay_manifest_rejects_different_targets(tmp_path):
    manifest_path, manifest, _, slow_run = _write_prediction_pair(tmp_path)
    slow_cfg = yaml.safe_load((slow_run / "config_resolved.yaml").read_text(encoding="utf-8"))
    slow_cfg["experiment"]["primary_target"] = "cognitive_load"
    (slow_run / "config_resolved.yaml").write_text(yaml.safe_dump(slow_cfg), encoding="utf-8")
    manifest["alignment"]["target_name"] = "stress"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="target_name_match"):
        load_prediction_artifacts_from_manifest({"replay": {"manifest_path": str(manifest_path)}})


def test_replay_manifest_rejects_disjoint_subjects(tmp_path):
    manifest_path, manifest, _, slow_run = _write_prediction_pair(tmp_path)
    slow_cfg = yaml.safe_load((slow_run / "config_resolved.yaml").read_text(encoding="utf-8"))
    slow_cfg["split"]["test_subjects"] = ["S9"]
    (slow_run / "config_resolved.yaml").write_text(yaml.safe_dump(slow_cfg), encoding="utf-8")
    slow_df = pd.read_csv(slow_run / "predictions_long.csv")
    slow_df["worker_id"] = "S9"
    slow_df["session_id"] = "S9"
    slow_df.to_csv(slow_run / "predictions_long.csv", index=False)
    manifest["alignment"]["subjects"] = ["S8"]
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="subjects"):
        load_prediction_artifacts_from_manifest({"replay": {"manifest_path": str(manifest_path)}})


def test_replay_manifest_rejects_timestamp_unit_mismatch(tmp_path):
    manifest_path, manifest, _, _ = _write_prediction_pair(tmp_path)
    manifest["alignment"]["timestamp_unit"] = "milliseconds"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="timestamp_unit_match"):
        load_prediction_artifacts_from_manifest({"replay": {"manifest_path": str(manifest_path)}})


def test_replay_manifest_rejects_duplicate_prediction_keys(tmp_path):
    manifest_path, _, fast_run, _ = _write_prediction_pair(tmp_path)
    fast_df = pd.read_csv(fast_run / "predictions_tcn.csv")
    pd.concat([fast_df, fast_df.iloc[[0]]], ignore_index=True).to_csv(fast_run / "predictions_tcn.csv", index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_prediction_artifacts_from_manifest({"replay": {"manifest_path": str(manifest_path)}})


def test_replay_manifest_rejects_missing_metadata(tmp_path):
    manifest_path, _, fast_run, _ = _write_prediction_pair(tmp_path)
    (fast_run / "config_resolved.yaml").unlink()

    with pytest.raises(FileNotFoundError, match="metadata"):
        load_prediction_artifacts_from_manifest({"replay": {"manifest_path": str(manifest_path)}})


def test_replay_manifest_rejects_incompatible_class_orientation(tmp_path):
    manifest_path, manifest, _, slow_run = _write_prediction_pair(tmp_path)
    slow_cfg = yaml.safe_load((slow_run / "config_resolved.yaml").read_text(encoding="utf-8"))
    slow_cfg["targets"]["stress"]["positive_class"] = "nonstress"
    (slow_run / "config_resolved.yaml").write_text(yaml.safe_dump(slow_cfg), encoding="utf-8")
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="positive_class_match"):
        load_prediction_artifacts_from_manifest({"replay": {"manifest_path": str(manifest_path)}})


def test_replay_manifest_rejects_insufficient_overlap(tmp_path):
    manifest_path, _, _, _ = _write_prediction_pair(tmp_path, slow_shift=1000.0)

    with pytest.raises(ValueError, match="timestamp_overlap_sufficient"):
        load_prediction_artifacts_from_manifest({"replay": {"manifest_path": str(manifest_path)}})


def test_cross_worker_state_contamination_does_not_occur(tmp_path):
    cfg = _load_config("src/config/streaming_multirate.yaml")
    events = pd.DataFrame(
        [
            {"timestamp": 0.0, "worker_id": "A", "session_id": "A", "event_type": "physical", **_physical_sample(0.0, emergency_input=True)},
            {"timestamp": 0.0, "worker_id": "B", "session_id": "B", "event_type": "physical", **_physical_sample(0.0)},
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ReplayEngine(cfg).run(events, run_dir)
    timeline = pd.read_csv(run_dir / "timeline.csv")
    assert timeline[timeline["worker_id"] == "A"]["new_state"].iloc[-1] == "EMERGENCY_STOP"
    assert timeline[timeline["worker_id"] == "B"]["new_state"].iloc[-1] == "NORMAL"
