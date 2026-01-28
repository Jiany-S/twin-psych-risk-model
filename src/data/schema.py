"""Schema helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DataSchema:
    time_idx: str
    worker_id: str
    target: str
    physiology: Sequence[str]
    robot_context: Sequence[str]
    hazard_zone: str
    task_phase: str

    @classmethod
    def from_config(cls, cfg: Mapping[str, object]) -> "DataSchema":
        features = cfg.get("features", {})
        optional = features.get("optional", {})
        return cls(
            time_idx=str(features.get("time_idx", "time_idx")),
            worker_id=str(features.get("worker_id", "worker_id")),
            target=str(features.get("target", "target")),
            physiology=list(features.get("physiology", ["hr", "hrv_rmssd", "gsr"])),
            robot_context=list(features.get("robot_context", ["distance_to_robot", "robot_speed"])),
            hazard_zone=str(optional.get("hazard_zone", "hazard_zone")),
            task_phase=str(optional.get("task_phase", "task_phase")),
        )

    def required_columns(self) -> list[str]:
        return [self.time_idx, self.worker_id, self.target, *self.physiology, *self.robot_context]
