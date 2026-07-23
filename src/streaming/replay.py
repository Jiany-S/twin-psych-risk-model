"""Timestamp-driven multi-rate streaming replay evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def _config_hash(cfg: dict[str, Any]) -> str:
    payload = yaml.safe_dump(cfg, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _run_dir(root: str | Path, mode: str) -> Path:
    path = Path(root) / f"streaming_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=False)
    return path


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


def physiological_events(cfg: dict[str, Any]) -> pd.DataFrame:
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
        rows.append({**base, "event_type": "fast", "probability": p_fast, "target": int(row[schema.primary_target]), "protocol_label": row[schema.protocol_label]})
        rows.append({**base, "event_type": "slow", "probability_5s": p_slow_5, "probability_30s": p_slow_30, "target": int(row[schema.primary_target]), "protocol_label": row[schema.protocol_label]})
    return pd.DataFrame(rows)


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
        self.missed_deadlines = 0
        self.stale_results = 0

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
                    if self.last_fast.input_quality != "valid":
                        self.missed_deadlines += 1
            elif event_type == "slow":
                if bool(self.cfg.get("slow_model", {}).get("enabled", True)):
                    probs = {5.0: event.get("probability_5s"), 30.0: event.get("probability_30s")}
                    self.last_slow = self.slow_adapter.update({"prediction_timestamp": t, "probability_by_horizon": probs}, current_timestamp=t)
                    self.latencies["slow"].append(self.last_slow.latency_ms)
                    if self.last_slow.input_quality in {"stale", "invalid_probability"}:
                        self.stale_results += 1

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
        metrics = self._summary_metrics(pd.DataFrame(timeline), pd.DataFrame(transitions))
        (run_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
        latency = self._latency_metrics()
        (run_dir / "latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
        validation = self._scenario_validation(pd.DataFrame(timeline), pd.DataFrame(transitions))
        (run_dir / "scenario_validation.json").write_text(json.dumps(validation, indent=2, default=_json_default), encoding="utf-8")
        (run_dir / "model_versions.json").write_text(json.dumps({"fast": "synthetic_or_replayed_fast", "slow": "synthetic_or_replayed_slow"}, indent=2), encoding="utf-8")
        self._write_plots(pd.DataFrame(timeline), run_dir)
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
            "protocol_label": event.get("protocol_label", ""),
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
        }

    def _latency_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {"missed_model_deadlines": self.missed_deadlines, "stale_result_count": self.stale_results}
        for key, vals in self.latencies.items():
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                out[key] = {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
            else:
                out[key] = {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)), "mean_ms": float(arr.mean())}
        return out

    @staticmethod
    def _summary_metrics(timeline: pd.DataFrame, transitions: pd.DataFrame) -> dict[str, Any]:
        states = timeline["new_state"].value_counts().to_dict() if not timeline.empty else {}
        metrics = {
            "time_spent_rows_by_state": states,
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
        if "target" in timeline.columns and timeline["target"].notna().any():
            target = pd.to_numeric(timeline["target"], errors="coerce")
            warning = timeline["new_state"].isin(["CAUTION", "HIGH_ALERT", "CONTROLLED_STOP"])
            duration_hours = max(1e-9, (float(timeline["timestamp"].max()) - float(timeline["timestamp"].min())) / 3600.0)
            stress = target == 1
            false_warning_rows = int((warning & (target == 0)).sum())
            first_stress_ts = float(timeline.loc[stress, "timestamp"].min()) if stress.any() else None
            warning_after = timeline[stress & warning]
            first_warning_ts = float(warning_after["timestamp"].min()) if not warning_after.empty else None
            metrics["physiological_replay_metrics"] = {
                "label": "WESAD protocol stress-state warning metrics; not construction hazard events",
                "false_alarms_per_hour": float(false_warning_rows / duration_hours),
                "stress_state_warning_coverage": float((warning & stress).sum() / max(1, int(stress.sum()))),
                "time_to_first_warning_after_protocol_stress_seconds": None if first_stress_ts is None or first_warning_ts is None else max(0.0, first_warning_ts - first_stress_ts),
                "missed_stress_rows": int((stress & ~warning).sum()),
            }
        return metrics

    @staticmethod
    def _scenario_validation(timeline: pd.DataFrame, transitions: pd.DataFrame) -> dict[str, Any]:
        if timeline.empty:
            return {"passed": False, "violated_invariants": ["empty_timeline"]}
        violations: list[str] = []
        emergency_rows = timeline[timeline["physical_reason_code"].eq("EMERGENCY_INPUT_ACTIVE")]
        if not emergency_rows.empty and not emergency_rows["new_state"].eq("EMERGENCY_STOP").all():
            violations.append("emergency_not_highest_priority")
        protective_rows = timeline[timeline["physical_reason_code"].eq("STOPPING_DISTANCE_VIOLATION")]
        if not protective_rows.empty and not protective_rows["new_state"].isin(["PROTECTIVE_STOP", "EMERGENCY_STOP"]).all():
            violations.append("protective_stop_reduced")
        expected = {
            "physical_protective_stop": "PROTECTIVE_STOP",
            "physical_emergency_stop": "EMERGENCY_STOP",
            "stale_physical_sensor": "DEGRADED",
            "simultaneous_slow_warning_and_physical_emergency": "EMERGENCY_STOP",
        }
        has_ml_events = bool(timeline["fast_ready"].any() or timeline["slow_ready"].any())
        if has_ml_events:
            expected.update(
                {
                    "slow_caution_only": "CAUTION",
                    "fast_high_alert": "HIGH_ALERT",
                    "fast_controlled_stop": "CONTROLLED_STOP",
                }
            )
        scenario_results = {}
        for scenario, expected_state in expected.items():
            rows = timeline[timeline["worker_id"].astype(str).eq(scenario)]
            actual_states = set(rows["new_state"].astype(str)) if not rows.empty else set()
            if rows.empty:
                continue
            passed = expected_state in actual_states or (expected_state == "HIGH_ALERT" and "CONTROLLED_STOP" in actual_states)
            scenario_results[scenario] = {"expected_state": expected_state, "observed_states": sorted(actual_states), "passed": passed}
            if not passed:
                violations.append(f"scenario_{scenario}_missing_{expected_state}")
        return {"passed": not violations, "violated_invariants": violations, "transition_count": int(len(transitions)), "scenario_results": scenario_results}

    @staticmethod
    def _write_plots(timeline: pd.DataFrame, run_dir: Path) -> None:
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
        except Exception:
            pass


def build_events(cfg: dict[str, Any]) -> pd.DataFrame:
    mode = str(cfg.get("replay", {}).get("mode", "combined_simulation"))
    if mode == "physiological":
        return physiological_events(cfg)
    physical = synthetic_physical_events(cfg)
    physical["event_type"] = "physical"
    if mode == "synthetic_physical":
        return physical
    if mode == "combined_simulation":
        ml = synthetic_ml_events(cfg, physical)
        return pd.concat([physical, ml], ignore_index=True)
    raise ValueError(f"Unsupported replay mode: {mode}")


def run(config_path: str | Path) -> Path:
    cfg = _load_config(config_path)
    mode = str(cfg.get("replay", {}).get("mode", "combined_simulation"))
    run_dir = _run_dir(cfg.get("paths", {}).get("run_root", "experiments/runs"), mode)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    events = build_events(cfg)
    engine = ReplayEngine(cfg)
    engine.run(events, run_dir)
    print(f"Streaming replay artifacts saved to {run_dir}")
    return run_dir


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if isinstance(cfg, dict) and cfg.get("_base_"):
        base_path = cfg_path.parent / str(cfg["_base_"])
        base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        return _deep_merge(base_cfg, cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config/streaming_multirate.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
