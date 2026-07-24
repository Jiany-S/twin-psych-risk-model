"""Timestamp-driven multi-rate streaming replay evaluator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.load_wesad import load_wesad_dataset
from src.data.schema import DataSchema
from src.safety.adapters import FastDetectorAdapter, SlowForecasterAdapter
from src.safety.physical_kernel import PhysicalKernelResult, PhysicalSafetyKernel, SafetyState
from src.safety.state_machine import SafetyStateMachine, TransitionResult
from src.streaming.prediction_artifacts import load_prediction_artifacts_from_manifest
from src.utils.artifacts import config_hash, git_commit
from src.utils.io import load_config_with_base


LOGGER = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def _config_hash(cfg: dict[str, Any]) -> str:
    return config_hash(cfg)


def _run_dir(root: str | Path, category: str) -> Path:
    path = Path(root) / f"streaming_{category}_{time.strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def result_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    mode = str(cfg.get("replay", {}).get("mode", "combined_simulation"))
    source = str(cfg.get("replay", {}).get("probability_source", "saved_predictions"))
    if mode == "physiological" and source == "saved_predictions":
        category = "real_model_physiological_replay"
        interpretation = "Replay of held-out WESAD physiological model predictions; not construction hazard validation."
        return {
            "result_category": category,
            "uses_real_sensor_data": True,
            "uses_real_model_predictions": True,
            "uses_synthetic_physical_inputs": False,
            "scientific_interpretation": interpretation,
        }
    if mode == "physiological" and source == "oracle_labels":
        return {
            "result_category": "oracle_label_policy_simulation",
            "uses_real_sensor_data": True,
            "uses_real_model_predictions": False,
            "uses_synthetic_physical_inputs": False,
            "scientific_interpretation": "Oracle-label physiological policy simulation; warning metrics are not model performance.",
        }
    if mode == "physiological" and source == "live_models":
        raise NotImplementedError(
            "replay.probability_source=live_models is intentionally not implemented yet; "
            "use saved_predictions from validated training artifacts or oracle_labels for policy-only simulation."
        )
    if mode == "synthetic_physical":
        return {
            "result_category": "synthetic_physical_safety_simulation",
            "uses_real_sensor_data": False,
            "uses_real_model_predictions": False,
            "uses_synthetic_physical_inputs": True,
            "scientific_interpretation": "Software-only deterministic physical safety simulation.",
        }
    return {
        "result_category": "combined_integration_simulation",
        "uses_real_sensor_data": False,
        "uses_real_model_predictions": False,
        "uses_synthetic_physical_inputs": True,
        "scientific_interpretation": "Software-only integration simulation for priority, persistence, and replay invariants.",
    }


def _default_physical_row(timestamp: float, worker_id: str = "synthetic", session_id: str = "session") -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "worker_id": worker_id,
        "session_id": session_id,
        "human_robot_distance_m": 5.0,
        "human_velocity_mps": 0.0,
        "robot_velocity_mps": 0.0,
        "relative_closing_speed_mps": 0.0,
        "robot_stopping_distance_m": 0.5,
        "configured_separation_margin_m": 1.0,
        "hazard_zone_active": False,
        "emergency_input": False,
        "protective_stop_input": False,
        "sensor_timestamp": timestamp,
    }


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def synthetic_physical_events(cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dt = float(cfg.get("physical_kernel", {}).get("update_interval_seconds", 0.05))
    scenarios = [
        ("normal_operation", 0, 2),
        ("slow_caution_only", 2, 4),
        ("fast_high_alert", 4, 6),
        ("fast_controlled_stop", 6, 8),
        ("physical_protective_stop", 8, 10),
        ("physical_emergency_stop", 10, 12),
        ("stale_physical_sensor", 12, 14),
        ("invalid_ml_probability", 14, 16),
        ("simultaneous_slow_warning_and_physical_emergency", 16, 18),
        ("emergency_reset_attempt_while_active", 18, 20),
        ("valid_emergency_reset_after_clear", 20, 24),
    ]
    for name, start, end in scenarios:
        ts = np.arange(start, end, dt)
        for t in ts:
            row = _default_physical_row(float(t), worker_id=name, session_id=name)
            if name == "physical_protective_stop":
                row["human_robot_distance_m"] = 1.8
                row["relative_closing_speed_mps"] = 0.3
                row["robot_stopping_distance_m"] = 1.0
            elif name in {"physical_emergency_stop", "simultaneous_slow_warning_and_physical_emergency"}:
                row["emergency_input"] = True
            elif name == "stale_physical_sensor":
                row["sensor_timestamp"] = float(t) - 5.0
            elif name == "emergency_reset_attempt_while_active":
                row["emergency_input"] = True
                row["explicit_reset_command"] = True
            elif name == "valid_emergency_reset_after_clear":
                row["emergency_input"] = t < 21.0
                row["explicit_reset_command"] = t > 22.0
            rows.append(row)
    return pd.DataFrame(rows)


def synthetic_ml_events(cfg: dict[str, Any], physical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in physical.iterrows():
        t = float(row["timestamp"])
        scenario = str(row["worker_id"])
        fast_p = 0.1
        slow5 = 0.2
        slow30 = 0.2
        if scenario == "slow_caution_only":
            slow5 = 0.8
            slow30 = 0.76
        if scenario in {"fast_high_alert", "fast_controlled_stop"}:
            fast_p = 0.85
        if scenario == "invalid_ml_probability":
            fast_p = float("nan")
        if scenario == "simultaneous_slow_warning_and_physical_emergency":
            slow5 = 0.9
            slow30 = 0.9
        rows.append({"timestamp": t, "worker_id": row["worker_id"], "session_id": row["session_id"], "event_type": "fast", "probability": fast_p})
        rows.append({"timestamp": t, "worker_id": row["worker_id"], "session_id": row["session_id"], "event_type": "slow", "probability_5s": slow5, "probability_30s": slow30})
    return pd.DataFrame(rows)


def physiological_events(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    source = str(cfg.get("replay", {}).get("probability_source", "saved_predictions"))
    if source == "saved_predictions":
        pair = load_prediction_artifacts_from_manifest(cfg)
        fast = pair.fast
        slow = pair.slow
        shared_subjects = set(pair.report.get("shared_subjects", []))
        origin_keys = fast[fast["worker_id"].isin(shared_subjects)][["worker_id", "session_id", "prediction_timestamp"]].drop_duplicates()
        origin_keys = origin_keys.merge(
            slow[slow["worker_id"].isin(shared_subjects)][["worker_id", "session_id", "prediction_timestamp"]].drop_duplicates(),
            on=["worker_id", "session_id", "prediction_timestamp"],
            how="inner",
        )
        fast = fast.merge(origin_keys, on=["worker_id", "session_id", "prediction_timestamp"], how="inner")
        slow = slow.merge(origin_keys, on=["worker_id", "session_id", "prediction_timestamp"], how="inner")
        events: list[dict[str, Any]] = []
        for _, row in fast.iterrows():
            events.append(
                {
                    "timestamp": float(row["prediction_timestamp"]),
                    "worker_id": str(row["worker_id"]),
                    "session_id": str(row["session_id"]),
                    "event_type": "fast",
                    "probability": float(row["calibrated_probability"]),
                    "target": int(row["target"]),
                    "protocol_label": "wesad_protocol",
                    "model_name": row["model_name"],
                    "model_version": row["model_version"],
                    "model_artifact_path": row["source_artifact_path"],
                    "model_artifact_hash": row["source_artifact_hash"],
                }
            )
        for (worker, session, pred_ts), group in slow.groupby(["worker_id", "session_id", "prediction_timestamp"], observed=True):
            probs = {float(r["horizon_seconds"]): float(r["calibrated_probability"]) for _, r in group.iterrows()}
            targets = {float(r["horizon_seconds"]): int(r["target"]) for _, r in group.iterrows()}
            events.append(
                {
                    "timestamp": float(pred_ts),
                    "worker_id": str(worker),
                    "session_id": str(session),
                    "event_type": "slow",
                    "probability_5s": probs.get(5.0, np.nan),
                    "probability_30s": probs.get(30.0, np.nan),
                    "target_5s": targets.get(5.0, np.nan),
                    "target_30s": targets.get(30.0, np.nan),
                    "target": targets.get(5.0, np.nan),
                    "protocol_label": "wesad_protocol",
                    "model_name": ",".join(sorted(group["model_name"].astype(str).unique())),
                    "model_version": ",".join(sorted(group["model_version"].astype(str).unique())),
                    "model_artifact_path": ",".join(sorted(group["source_artifact_path"].astype(str).unique())),
                    "model_artifact_hash": ",".join(sorted(group["source_artifact_hash"].astype(str).unique())),
                }
            )
        return pd.DataFrame(events), pair.report
    if source != "oracle_labels":
        raise ValueError(
            "replay.probability_source must be saved_predictions, live_models, or oracle_labels. "
            "live_models is not implemented because replay must not duplicate preprocessing."
        )
    schema = DataSchema.from_config(cfg)
    dataset_cfg = cfg.get("dataset", {})
    stream = cfg.get("stream", {})
    subjects = dataset_cfg.get("subjects", ["S8"])
    df = load_wesad_dataset(
        dataset_cfg.get("path", "data/wesad/wesad_subset"),
        schema=schema,
        data_format=dataset_cfg.get("format", "auto"),
        subjects=subjects,
        max_rows_per_subject=int(dataset_cfg.get("max_rows_per_subject", 2000)),
        target_sampling_rate_hz=float(stream.get("target_sampling_rate_hz", 4.0)),
        stress_include_amusement=bool(dataset_cfg.get("stress_include_amusement", True)),
    )
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        p_fast = 0.15 + 0.7 * float(row[schema.primary_target])
        p_slow_5 = 0.2 + 0.65 * float(row[schema.primary_target])
        p_slow_30 = 0.2 + 0.55 * float(row[schema.primary_target])
        base = {"timestamp": float(row[schema.timestamp]), "worker_id": str(row[schema.worker_id]), "session_id": str(row[schema.worker_id])}
        rows.append({**base, "event_type": "fast", "probability": p_fast, "target": int(row[schema.primary_target]), "protocol_label": row[schema.protocol_label], "model_version": "oracle_label_generator_v1"})
        rows.append({**base, "event_type": "slow", "probability_5s": p_slow_5, "probability_30s": p_slow_30, "target": int(row[schema.primary_target]), "target_5s": int(row[schema.primary_target]), "target_30s": int(row[schema.primary_target]), "protocol_label": row[schema.protocol_label], "model_version": "oracle_label_generator_v1"})
    return pd.DataFrame(rows), None


class ReplayEngine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.config_hash = _config_hash(cfg)
        self.physical_kernel = PhysicalSafetyKernel(cfg.get("physical_kernel", {}))
        self.fast_adapter = FastDetectorAdapter(cfg.get("fast_policy", {}), model_version="synthetic_or_replayed_fast")
        self.slow_adapter = SlowForecasterAdapter(cfg.get("slow_policy", {}), model_version="synthetic_or_replayed_slow")
        self.state_machine = SafetyStateMachine(cfg.get("state_machine", {}))
        self.last_physical: PhysicalKernelResult | None = None
        self.last_fast = None
        self.last_slow = None
        self.latencies: dict[str, list[float]] = {"physical": [], "fast": [], "slow": [], "state_machine": [], "end_to_end": []}
        self.counts = {
            "fast_invalid_probability_count": 0,
            "slow_invalid_probability_count": 0,
            "fast_deadline_miss_count": 0,
            "slow_deadline_miss_count": 0,
            "fast_stale_result_count": 0,
            "slow_stale_result_count": 0,
            "fast_inference_failure_count": 0,
            "slow_inference_failure_count": 0,
        }
        self.plot_failures: list[str] = []

    def reset_session(self) -> None:
        self.state_machine.reset_session()
        self.last_physical = None
        self.last_fast = None
        self.last_slow = None
        self.fast_adapter = FastDetectorAdapter(self.cfg.get("fast_policy", {}), model_version="synthetic_or_replayed_fast")
        self.slow_adapter = SlowForecasterAdapter(self.cfg.get("slow_policy", {}), model_version="synthetic_or_replayed_slow")

    def run(self, events: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
        order = {"physical": 0, "fast": 1, "slow": 2}
        events = events.copy()
        events["_event_order"] = events.get("event_type", "physical").map(order).fillna(9)
        events = events.sort_values(["timestamp", "worker_id", "_event_order"], kind="mergesort").reset_index(drop=True)
        timeline: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        last_session = None
        previous_state = SafetyState.NORMAL
        for _, event in events.iterrows():
            started = time.perf_counter()
            session_id = str(event.get("session_id", event.get("worker_id", "session")))
            if last_session is not None and session_id != last_session:
                self.reset_session()
                previous_state = SafetyState.NORMAL
            last_session = session_id
            t = float(event["timestamp"])
            event_type = str(event.get("event_type", "physical"))
            if event_type == "physical":
                sample = event.dropna().to_dict()
                self.last_physical = self.physical_kernel.evaluate(sample, current_timestamp=t)
                self.latencies["physical"].append(float(self.last_physical.derived_values.get("latency_ms") or 0.0))
            elif event_type == "fast":
                if bool(self.cfg.get("fast_model", {}).get("enabled", True)):
                    self.last_fast = self.fast_adapter.update({"timestamp": t, "probability": event.get("probability")})
                    self.latencies["fast"].append(self.last_fast.latency_ms)
                    if self.last_fast.input_quality == "invalid_probability":
                        self.counts["fast_invalid_probability_count"] += 1
            elif event_type == "slow":
                if bool(self.cfg.get("slow_model", {}).get("enabled", True)):
                    probs = {5.0: event.get("probability_5s"), 30.0: event.get("probability_30s")}
                    self.last_slow = self.slow_adapter.update({"prediction_timestamp": t, "probability_by_horizon": probs}, current_timestamp=t)
                    self.latencies["slow"].append(self.last_slow.latency_ms)
                    if self.last_slow.input_quality == "invalid_probability":
                        self.counts["slow_invalid_probability_count"] += 1
                    if self.last_slow.input_quality == "stale":
                        self.counts["slow_stale_result_count"] += 1

            physical_for_sm = self.last_physical
            if not bool(self.cfg.get("physical_kernel", {}).get("enabled", True)) and physical_for_sm is None:
                physical_for_sm = PhysicalKernelResult(SafetyState.NORMAL, "PHYSICAL_NOT_AVAILABLE", 0, t, {"sensor_age_seconds": None, "time_to_collision_seconds": None}, {"valid": True, "enabled": False})
            result = self.state_machine.update(
                physical_for_sm,
                self.last_fast,
                self.last_slow,
                current_timestamp=t,
                explicit_reset_command=_truthy(event.get("explicit_reset_command", False)),
            )
            self.latencies["state_machine"].append(result.latency_ms)
            self.latencies["end_to_end"].append((time.perf_counter() - started) * 1000.0)
            if result.new_state != previous_state or result.reason_code in {"EMERGENCY_RESET_REJECTED", "EMERGENCY_RESET_ACCEPTED"}:
                transitions.append(self._transition_row(result, event))
            previous_state = result.new_state
            actions.append(self._action_row(result))
            timeline.append(self._timeline_row(t, event, result))
        pd.DataFrame(timeline).to_csv(run_dir / "timeline.csv", index=False)
        pd.DataFrame(transitions).to_csv(run_dir / "state_transitions.csv", index=False)
        pd.DataFrame(actions).to_csv(run_dir / "actions.csv", index=False)
        timeline_df = pd.DataFrame(timeline)
        transition_df = pd.DataFrame(transitions)
        action_df = pd.DataFrame(actions)
        metrics = self._summary_metrics(timeline_df, transition_df, action_df)
        metrics.update(result_metadata(self.cfg))
        (run_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
        latency = self._latency_metrics()
        (run_dir / "latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
        validation = self._scenario_validation(timeline_df, transition_df, str(self.cfg.get("replay", {}).get("mode", "combined_simulation")))
        (run_dir / "scenario_validation.json").write_text(json.dumps(validation, indent=2, default=_json_default), encoding="utf-8")
        (run_dir / "model_versions.json").write_text(json.dumps(self._model_versions(timeline_df), indent=2, default=_json_default), encoding="utf-8")
        run_meta = {**result_metadata(self.cfg), "config_hash": self.config_hash, "git_commit": git_commit(), "plot_failures": self.plot_failures}
        (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, default=_json_default), encoding="utf-8")
        self._write_plots(timeline_df, run_dir)
        if self.plot_failures:
            run_meta["plot_failures"] = self.plot_failures
            (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, default=_json_default), encoding="utf-8")
        return metrics

    def _timeline_row(self, t: float, event: pd.Series, result: TransitionResult) -> dict[str, Any]:
        physical = self.last_physical
        fast = self.last_fast
        slow = self.last_slow
        return {
            "timestamp": t,
            "worker_id": str(event.get("worker_id", "")),
            "session_id": str(event.get("session_id", "")),
            "physical_state_request": physical.requested_state.value if physical else "NOT_AVAILABLE",
            "physical_reason_code": physical.reason_code if physical else "PHYSICAL_NOT_AVAILABLE",
            "distance_m": event.get("human_robot_distance_m", np.nan),
            "relative_closing_speed_mps": event.get("relative_closing_speed_mps", np.nan),
            "ttc_seconds": physical.derived_values.get("time_to_collision_seconds") if physical else np.nan,
            "physical_input_age_seconds": physical.derived_values.get("sensor_age_seconds") if physical else np.nan,
            "fast_ready": bool(fast.ready) if fast else False,
            "fast_probability": fast.probability if fast else np.nan,
            "fast_persistent_positive": bool(fast.persistent_positive) if fast else False,
            "fast_latency_ms": fast.latency_ms if fast else np.nan,
            "slow_ready": bool(slow.ready) if slow else False,
            "slow_probability_5s": slow.probability_by_horizon.get(5.0, np.nan) if slow else np.nan,
            "slow_probability_30s": slow.probability_by_horizon.get(30.0, np.nan) if slow else np.nan,
            "slow_latency_ms": slow.latency_ms if slow else np.nan,
            "previous_state": result.previous_state.value,
            "new_state": result.new_state.value,
            "requested_action": result.requested_action.action_name,
            "trigger_layer": result.trigger_layer,
            "transition_reason_code": result.reason_code,
            "model_versions": "fast=synthetic_or_replayed_fast;slow=synthetic_or_replayed_slow",
            "config_hash": self.config_hash,
            "target": event.get("target", np.nan),
            "target_5s": event.get("target_5s", np.nan),
            "target_30s": event.get("target_30s", np.nan),
            "protocol_label": event.get("protocol_label", ""),
            "event_type": event.get("event_type", ""),
            "model_version": event.get("model_version", ""),
            "model_artifact_path": event.get("model_artifact_path", ""),
            "model_artifact_hash": event.get("model_artifact_hash", ""),
        }

    @staticmethod
    def _transition_row(result: TransitionResult, event: pd.Series) -> dict[str, Any]:
        return {
            "timestamp": result.timestamp,
            "worker_id": str(event.get("worker_id", "")),
            "session_id": str(event.get("session_id", "")),
            "previous_state": result.previous_state.value,
            "new_state": result.new_state.value,
            "trigger_layer": result.trigger_layer,
            "reason_code": result.reason_code,
            "latched": result.latched,
            "transition_allowed": result.transition_allowed,
        }

    @staticmethod
    def _action_row(result: TransitionResult) -> dict[str, Any]:
        action = result.requested_action
        return {
            "timestamp": action.timestamp,
            "action_name": action.action_name,
            "priority": action.priority,
            "source_layer": action.source_layer,
            "reason_code": action.reason_code,
            "parameters": json.dumps(action.parameters, sort_keys=True),
            "action_event_type": "state_entry" if result.transition_allowed else "state_hold",
        }

    def _latency_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.counts)
        for key, vals in self.latencies.items():
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                out[key] = {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
            else:
                out[key] = {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)), "mean_ms": float(arr.mean())}
        return out

    @staticmethod
    def _summary_metrics(timeline: pd.DataFrame, transitions: pd.DataFrame, actions: pd.DataFrame) -> dict[str, Any]:
        states = timeline["new_state"].value_counts().to_dict() if not timeline.empty else {}
        durations = _durations_by_state(timeline)
        total_duration = sum(durations.values()) or 1.0
        metrics = {
            "row_count_by_state": states,
            "duration_seconds_by_state": durations,
            "duration_fraction_by_state": {k: float(v / total_duration) for k, v in durations.items()},
            "number_of_transitions": int(len(transitions)),
            "caution_transitions": int((transitions["new_state"] == "CAUTION").sum()) if not transitions.empty else 0,
            "high_alert_transitions": int((transitions["new_state"] == "HIGH_ALERT").sum()) if not transitions.empty else 0,
            "controlled_stops": int((transitions["new_state"] == "CONTROLLED_STOP").sum()) if not transitions.empty else 0,
            "protective_stops": int((transitions["new_state"] == "PROTECTIVE_STOP").sum()) if not transitions.empty else 0,
            "emergency_stops": int((transitions["new_state"] == "EMERGENCY_STOP").sum()) if not transitions.empty else 0,
            "degraded_entries": int((transitions["new_state"] == "DEGRADED").sum()) if not transitions.empty else 0,
            "reset_attempts": int(timeline["transition_reason_code"].isin(["EMERGENCY_RESET_REJECTED", "EMERGENCY_RESET_ACCEPTED"]).sum()) if not timeline.empty else 0,
            "failed_reset_attempts": int((timeline["transition_reason_code"] == "EMERGENCY_RESET_REJECTED").sum()) if not timeline.empty else 0,
            "state_oscillation_count": int((timeline["new_state"] != timeline["new_state"].shift()).sum()) if len(timeline) else 0,
        }
        for state in ["CAUTION", "HIGH_ALERT", "CONTROLLED_STOP", "DEGRADED", "PROTECTIVE_STOP", "EMERGENCY_STOP"]:
            metrics[f"{state.lower()}_state_entries"] = int((transitions["new_state"] == state).sum()) if not transitions.empty else 0
            metrics[f"{state.lower()}_state_exits"] = int((transitions["previous_state"] == state).sum()) if not transitions.empty else 0
            metrics[f"{state.lower()}_hold_rows"] = int((timeline["new_state"] == state).sum()) if not timeline.empty else 0
            metrics[f"{state.lower()}_action_emissions"] = metrics[f"{state.lower()}_hold_rows"]
        if "target" in timeline.columns and timeline["target"].notna().any():
            metrics["physiological_replay_metrics"] = physiological_policy_metrics(timeline)
        return metrics

    @staticmethod
    def _scenario_validation(timeline: pd.DataFrame, transitions: pd.DataFrame, mode: str = "combined_simulation") -> dict[str, Any]:
        if timeline.empty:
            return {"passed": False, "violated_invariants": ["empty_timeline"]}
        violations: list[str] = []
        emergency_rows = timeline[timeline["physical_reason_code"].eq("EMERGENCY_INPUT_ACTIVE")]
        if not emergency_rows.empty and not emergency_rows["new_state"].eq("EMERGENCY_STOP").all():
            violations.append("emergency_not_highest_priority")
        protective_rows = timeline[timeline["physical_reason_code"].eq("STOPPING_DISTANCE_VIOLATION")]
        if not protective_rows.empty and not protective_rows["new_state"].isin(["PROTECTIVE_STOP", "EMERGENCY_STOP"]).all():
            violations.append("protective_stop_reduced")
        scenario_results = {}
        manifest_path = Path("data/synthetic_replay/scenario_manifest.csv")
        if not manifest_path.exists() or mode == "physiological":
            return {"passed": not violations, "violated_invariants": violations, "transition_count": int(len(transitions)), "scenario_results": scenario_results}
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["applies_to"].astype(str).str.contains(mode, regex=False)]
        for _, expected in manifest.iterrows():
            scenario = str(expected["scenario"])
            start = float(expected["start_timestamp"])
            end = float(expected["end_timestamp"])
            rows = timeline[(timeline["worker_id"].astype(str).eq(scenario)) & (timeline["timestamp"] >= start) & (timeline["timestamp"] < end)]
            if rows.empty:
                scenario_results[scenario] = {"passed": False, "fail_reason": "missing_scenario_interval"}
                violations.append(f"scenario_{scenario}_missing")
                continue
            observed_sequence = _compressed_sequence(rows["new_state"].astype(str).tolist())
            expected_sequence = [s for s in str(expected["expected_state_sequence"]).split(">") if s]
            forbidden = [s for s in str(expected.get("forbidden_states", "")).split("|") if s and s != "nan"]
            required_reason = str(expected.get("required_reason_code", "")).strip()
            scenario_transitions = transitions[(transitions["worker_id"].astype(str).eq(scenario)) & (transitions["timestamp"] >= start) & (transitions["timestamp"] < end)]
            order_ok = _is_subsequence(expected_sequence, observed_sequence)
            forbidden_seen = sorted(set(observed_sequence) & set(forbidden))
            reason_ok = True if not required_reason or required_reason == "nan" else required_reason in set(scenario_transitions["reason_code"].astype(str)) or required_reason in set(rows["transition_reason_code"].astype(str))
            entry_ok = observed_sequence[0] == str(expected["expected_entry_state"]) if observed_sequence else False
            final_ok = observed_sequence[-1] == str(expected["expected_final_state"]) if observed_sequence else False
            explicit_reset_expected = _truthy(expected.get("explicit_reset_expected", False))
            reset_ok = True
            if explicit_reset_expected:
                reset_ok = rows["transition_reason_code"].astype(str).str.contains("RESET").any()
            passed = bool(order_ok and not forbidden_seen and reason_ok and entry_ok and final_ok and reset_ok)
            scenario_results[scenario] = {
                "expected_state_sequence": expected_sequence,
                "observed_state_sequence": observed_sequence,
                "entry_ok": entry_ok,
                "final_ok": final_ok,
                "order_ok": order_ok,
                "forbidden_seen": forbidden_seen,
                "required_reason_code": required_reason,
                "reason_ok": reason_ok,
                "explicit_reset_expected": explicit_reset_expected,
                "reset_ok": reset_ok,
                "passed": passed,
            }
            if not passed:
                violations.append(f"scenario_{scenario}_failed")
        return {"passed": not violations, "violated_invariants": violations, "transition_count": int(len(transitions)), "scenario_results": scenario_results}

    def _write_plots(self, timeline: pd.DataFrame, run_dir: Path) -> None:
        try:
            import matplotlib.pyplot as plt
            state_codes = {state: idx for idx, state in enumerate(["NORMAL", "CAUTION", "HIGH_ALERT", "CONTROLLED_STOP", "DEGRADED", "PROTECTIVE_STOP", "EMERGENCY_STOP"])}
            plt.figure(figsize=(10, 3))
            plt.plot(timeline["timestamp"], timeline["new_state"].map(state_codes))
            plt.yticks(list(state_codes.values()), list(state_codes.keys()))
            plt.tight_layout()
            plt.savefig(run_dir / "state_timeline.png")
            plt.close()
            plt.figure(figsize=(10, 3))
            plt.plot(timeline["timestamp"], timeline["fast_probability"], label="fast")
            plt.plot(timeline["timestamp"], timeline["slow_probability_5s"], label="slow_5s")
            plt.plot(timeline["timestamp"], timeline["slow_probability_30s"], label="slow_30s")
            plt.legend()
            plt.tight_layout()
            plt.savefig(run_dir / "probability_timeline.png")
            plt.close()
            if timeline["distance_m"].notna().any():
                plt.figure(figsize=(10, 3))
                plt.plot(timeline["timestamp"], timeline["distance_m"], label="distance_m")
                plt.legend()
                plt.tight_layout()
                plt.savefig(run_dir / "physical_input_timeline.png")
                plt.close()
        except Exception as exc:
            message = f"plot generation failed: {exc}"
            LOGGER.warning(message)
            self.plot_failures.append(message)
            if bool(self.cfg.get("replay", {}).get("strict_plots", False)):
                raise

    @staticmethod
    def _model_versions(timeline: pd.DataFrame) -> dict[str, Any]:
        rows = timeline[timeline.get("model_version", "").astype(str).str.len() > 0] if not timeline.empty and "model_version" in timeline else pd.DataFrame()
        if rows.empty:
            return {"generators": ["synthetic_probability_scenario_v1"]}
        return {
            "model_versions": sorted(set(rows["model_version"].astype(str))),
            "model_artifact_paths": sorted(set(rows["model_artifact_path"].dropna().astype(str))),
            "model_artifact_hashes": sorted(set(rows["model_artifact_hash"].dropna().astype(str))),
        }


def build_events(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    mode = str(cfg.get("replay", {}).get("mode", "combined_simulation"))
    if mode == "physiological":
        return physiological_events(cfg)
    physical = synthetic_physical_events(cfg)
    physical["event_type"] = "physical"
    if mode == "synthetic_physical":
        return physical, None
    if mode == "combined_simulation":
        ml = synthetic_ml_events(cfg, physical)
        return pd.concat([physical, ml], ignore_index=True), None
    raise ValueError(f"Unsupported replay mode: {mode}")


def action_name_for_state(state: str) -> str:
    return {
        "NORMAL": "continue_normal",
        "CAUTION": "reduce_speed",
        "HIGH_ALERT": "request_operator_attention",
        "CONTROLLED_STOP": "controlled_stop",
        "DEGRADED": "controlled_stop",
        "PROTECTIVE_STOP": "protective_stop",
        "EMERGENCY_STOP": "emergency_stop",
    }[state]


def _durations_by_state(timeline: pd.DataFrame) -> dict[str, float]:
    if timeline.empty:
        return {}
    totals: dict[str, float] = {}
    for _, group in timeline.sort_values(["session_id", "timestamp"]).groupby("session_id", observed=True):
        ts = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy(dtype=float)
        states = group["new_state"].astype(str).to_numpy()
        if len(ts) == 0:
            continue
        diffs = np.diff(ts, append=ts[-1])
        positive = diffs[diffs > 0]
        final_delta = float(np.median(positive)) if len(positive) else 0.0
        diffs[-1] = final_delta
        diffs = np.clip(diffs, 0.0, None)
        for state, duration in zip(states, diffs, strict=False):
            totals[state] = totals.get(state, 0.0) + float(duration)
    return totals


def physiological_policy_metrics(timeline: pd.DataFrame) -> dict[str, Any]:
    label = "WESAD experimental stress episode policy metrics; not construction hazard events"
    metrics: dict[str, Any] = {"label": label}
    metrics["combined_policy_metrics"] = _policy_metrics_for_frame(timeline, "target", timeline["new_state"].isin(["CAUTION", "HIGH_ALERT", "CONTROLLED_STOP"]))
    fast_rows = timeline[timeline["event_type"].astype(str).eq("fast")].drop_duplicates(["worker_id", "timestamp", "target"])
    if not fast_rows.empty:
        metrics["fast_policy_metrics"] = _policy_metrics_for_frame(fast_rows, "target", fast_rows["fast_probability"] >= 0.70)
    slow_rows = timeline[timeline["event_type"].astype(str).eq("slow")]
    if not slow_rows.empty:
        slow5 = slow_rows.drop_duplicates(["worker_id", "timestamp", "target_5s"])
        slow30 = slow_rows.drop_duplicates(["worker_id", "timestamp", "target_30s"])
        metrics["slow_5s_policy_metrics"] = _policy_metrics_for_frame(slow5, "target_5s", slow5["slow_probability_5s"] >= 0.75)
        metrics["slow_30s_policy_metrics"] = _policy_metrics_for_frame(slow30, "target_30s", slow30["slow_probability_30s"] >= 0.70)
    return metrics


def _policy_metrics_for_frame(frame: pd.DataFrame, target_col: str, warning: pd.Series) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    if frame.empty or target_col not in frame:
        return {}
    y = pd.to_numeric(frame[target_col], errors="coerce")
    valid = y.notna()
    y = y[valid].astype(int)
    warning = warning.loc[y.index].astype(bool)
    probs = None
    for col in ["fast_probability", "slow_probability_5s", "slow_probability_30s"]:
        if col in frame and frame.loc[y.index, col].notna().any():
            probs = pd.to_numeric(frame.loc[y.index, col], errors="coerce").fillna(0.0)
            break
    duration_hours = max(1e-9, (float(frame["timestamp"].max()) - float(frame["timestamp"].min())) / 3600.0)
    out = {
        "warning_coverage": float((warning & (y == 1)).sum() / max(1, int((y == 1).sum()))),
        "false_alarms_per_hour": float((warning & (y == 0)).sum() / duration_hours),
        "missed_stress_rows": int(((y == 1) & ~warning).sum()),
        "unique_prediction_origins": int(frame.loc[y.index, ["worker_id", "timestamp"]].drop_duplicates().shape[0]),
        "prediction_cadence_seconds": _median_cadence(frame.loc[y.index]),
        "f1": float(f1_score(y, warning, zero_division=0)) if len(y) else 0.0,
    }
    if probs is not None and y.nunique() == 2:
        out["auroc"] = float(roc_auc_score(y, probs))
        out["auprc"] = float(average_precision_score(y, probs))
    episodes = _stress_episode_metrics(frame.loc[y.index].assign(_target=y.to_numpy(), _warning=warning.to_numpy()))
    out.update(episodes)
    return out


def _median_cadence(frame: pd.DataFrame) -> float:
    vals = []
    for _, group in frame.groupby("worker_id", observed=True):
        ts = np.sort(pd.to_numeric(group["timestamp"], errors="coerce").dropna().unique())
        if len(ts) > 1:
            vals.extend(np.diff(ts).tolist())
    return float(np.median(vals)) if vals else 0.0


def _stress_episode_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    episodes = []
    for worker, group in frame.sort_values(["worker_id", "timestamp"]).groupby("worker_id", observed=True):
        active = False
        start = None
        rows = []
        for _, row in group.iterrows():
            if int(row["_target"]) == 1 and not active:
                active = True
                start = float(row["timestamp"])
                rows = []
            if active:
                rows.append(row)
            if active and int(row["_target"]) == 0:
                active = False
                episodes.append((worker, start, rows[:-1]))
        if active:
            episodes.append((worker, start, rows))
    detected = 0
    delays = []
    leads = []
    for _, start, rows in episodes:
        warnings = [float(r["timestamp"]) for r in rows if bool(r["_warning"])]
        if warnings:
            detected += 1
            first_warning = min(warnings)
            delays.append(max(0.0, first_warning - float(start)))
            if "target_timestamp" in frame.columns:
                first_rows = [r for r in rows if bool(r["_warning"]) and float(r["timestamp"]) == first_warning]
                if first_rows:
                    target_ts = pd.to_numeric(pd.Series([first_rows[0].get("target_timestamp")]), errors="coerce").iloc[0]
                    if pd.notna(target_ts):
                        leads.append(max(0.0, float(target_ts) - first_warning))
    cadence = _median_cadence(frame)
    total_warning_rows = int(frame["_warning"].sum())
    false_warning_rows = int(((frame["_target"] == 0) & frame["_warning"]).sum())
    return {
        "number_of_stress_episodes": int(len(episodes)),
        "detected_episodes": int(detected),
        "missed_episodes": int(len(episodes) - detected),
        "warning_delay_seconds_per_episode": delays,
        "warning_lead_seconds_per_episode": leads,
        "total_warning_duration_rows": total_warning_rows,
        "total_warning_duration_seconds": float(total_warning_rows * cadence),
        "false_warnings_during_non_stress_protocols": false_warning_rows,
        "false_warning_duration_seconds": float(false_warning_rows * cadence),
    }


def _compressed_sequence(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def _is_subsequence(expected: list[str], observed: list[str]) -> bool:
    if not expected:
        return True
    pos = 0
    for item in observed:
        if item == expected[pos]:
            pos += 1
            if pos == len(expected):
                return True
    return False


def run(config_path: str | Path) -> Path:
    cfg = _load_config(config_path)
    meta = result_metadata(cfg)
    run_dir = _run_dir(cfg.get("paths", {}).get("run_root", "experiments/runs"), meta["result_category"])
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    events, pairing_report = build_events(cfg)
    engine = ReplayEngine(cfg)
    engine.run(events, run_dir)
    if pairing_report is not None:
        (run_dir / "artifact_pairing_report.json").write_text(
            json.dumps(pairing_report, indent=2, default=_json_default),
            encoding="utf-8",
        )
    print(f"Streaming replay artifacts saved to {run_dir}")
    return run_dir


def _load_config(path: str | Path) -> dict[str, Any]:
    return load_config_with_base(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config/streaming_multirate.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
