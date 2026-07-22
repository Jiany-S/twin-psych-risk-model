"""Schema helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DataSchema:
    timestamp: str
    time_idx: str
    worker_id: str
    stress_target: str
    comfort_target: str
    primary_target_name: str
    primary_target: str
    primary_task_type: str
    target_map: Mapping[str, str]
    protocol_label: str
    physiology: Sequence[str]
    feature_engineering_mode: str
    robot_context: Sequence[str]
    hazard_zone: str
    task_phase: str
    specialization_col: str
    experience_col: str

    @classmethod
    def from_config(cls, cfg: Mapping[str, object]) -> "DataSchema":
        features = cfg.get("features", {})
        optional = features.get("optional", {})
        use_robot_context = bool(features.get("use_robot_context", True))
        targets = cfg.get("targets", {})
        if not isinstance(targets, Mapping):
            targets = {}
        experiment = cfg.get("experiment", {})
        if not isinstance(experiment, Mapping):
            experiment = {}

        def target_cfg(name: str) -> Mapping[str, Any]:
            block = targets.get(name, {})
            return block if isinstance(block, Mapping) else {}

        def target_col(name: str, default: str) -> str:
            block = target_cfg(name)
            return str(block.get("label_col", default))

        target_map = {
            "stress": target_col("stress", "y_stress"),
            "stress_binary": target_col("stress_binary", "y_stress_binary"),
            "cognitive_load": target_col("cognitive_load", "y_cognitive_load"),
            "cognitive_load_binary": target_col("cognitive_load_binary", "y_cognitive_load_binary"),
            "comfort": target_col("comfort", "y_comfort_proxy"),
            "comfort_binary": target_col("comfort_binary", "y_comfort_binary"),
            "valence_binary": target_col("valence_binary", "y_valence_binary"),
            "arousal": target_col("arousal", "y_arousal"),
            "arousal_binary": target_col("arousal_binary", "y_arousal_binary"),
        }
        primary_name = str(experiment.get("primary_target", target_cfg("primary").get("name", "stress")))
        primary_col = str(target_cfg("primary").get("label_col", target_map.get(primary_name, "y_stress")))
        primary_task_type = str(
            target_cfg("primary").get(
                "task_type",
                "classification" if primary_name.endswith("_binary") or primary_name == "stress" else "regression",
            )
        )
        return cls(
            timestamp=str(features.get("timestamp", "timestamp")),
            time_idx=str(features.get("time_idx", "time_idx")),
            worker_id=str(features.get("worker_id", "worker_id")),
            # Existing training code uses stress_target as the primary classification target.
            stress_target=primary_col,
            comfort_target=target_map["comfort"],
            primary_target_name=primary_name,
            primary_target=primary_col,
            primary_task_type=primary_task_type,
            target_map=target_map,
            protocol_label=str(features.get("protocol_label", "protocol_label")),
            physiology=list(features.get("physiology", ["ecg", "eda", "temp"])),
            feature_engineering_mode=str(features.get("engineering_mode", "raw_signals")),
            robot_context=list(features.get("robot_context", ["distance_to_robot", "robot_speed"])) if use_robot_context else [],
            hazard_zone=str(optional.get("hazard_zone", "hazard_zone")),
            task_phase=str(optional.get("task_phase", "task_phase")),
            specialization_col=str(optional.get("specialization_col", "specialization_index")),
            experience_col=str(optional.get("experience_col", "experience_level")),
        )

    def required_columns(self) -> list[str]:
        return [
            self.timestamp,
            self.time_idx,
            self.worker_id,
            self.stress_target,
            self.comfort_target,
            *self.physiology,
        ]

    def target_col(self, name: str) -> str:
        return str(self.target_map.get(name, f"y_{name}"))

    def configured_target_columns(self) -> list[str]:
        cols = [self.primary_target, self.comfort_target]
        cols.extend(str(c) for c in self.target_map.values())
        return sorted(set(cols))
