"""Schema helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DataSchema:
    timestamp: str
    time_idx: str
    worker_id: str
    stress_target: str
    comfort_target: str
    protocol_label: str
    physiology: Sequence[str]
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
        stress_cfg = targets.get("stress", {})
        comfort_cfg = targets.get("comfort", {})
        return cls(
            timestamp=str(features.get("timestamp", "timestamp")),
            time_idx=str(features.get("time_idx", "time_idx")),
            worker_id=str(features.get("worker_id", "worker_id")),
            stress_target=str(stress_cfg.get("label_col", "y_stress")),
            comfort_target=str(comfort_cfg.get("label_col", "y_comfort_proxy")),
            protocol_label=str(features.get("protocol_label", "protocol_label")),
            physiology=list(features.get("physiology", ["ecg", "eda", "temp"])),
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
