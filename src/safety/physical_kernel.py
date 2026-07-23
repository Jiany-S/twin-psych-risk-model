"""Configurable deterministic physical safety logic.

This module is inspired by separation-monitoring and stopping-distance
principles, but it is not an ISO-compliance implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any


class SafetyState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    HIGH_ALERT = "HIGH_ALERT"
    CONTROLLED_STOP = "CONTROLLED_STOP"
    DEGRADED = "DEGRADED"
    PROTECTIVE_STOP = "PROTECTIVE_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True)
class PhysicalKernelResult:
    requested_state: SafetyState
    reason_code: str
    severity: int
    timestamp: float
    derived_values: dict[str, float | None]
    input_validity: dict[str, Any]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _num(sample: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = sample.get(key, default)
    if value is None:
        return None
    return float(value)


class PhysicalSafetyKernel:
    """Evaluate timestamped physical inputs without any ML dependency."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def evaluate(self, sample: dict[str, Any], current_timestamp: float | None = None) -> PhysicalKernelResult:
        started = time.perf_counter()
        now = float(current_timestamp if current_timestamp is not None else sample.get("current_timestamp", sample.get("timestamp", 0.0)))
        sensor_ts = float(sample.get("sensor_timestamp", sample.get("timestamp", now)))
        sensor_age = max(0.0, now - sensor_ts)
        enabled = bool(self.config.get("enabled", True))
        if not enabled:
            return PhysicalKernelResult(
                SafetyState.NORMAL,
                "PHYSICAL_KERNEL_DISABLED",
                0,
                now,
                {"sensor_age_seconds": sensor_age, "latency_ms": (time.perf_counter() - started) * 1000.0},
                {"valid": True, "enabled": False},
            )

        required = [
            "human_robot_distance_m",
            "relative_closing_speed_mps",
            "robot_stopping_distance_m",
            "configured_separation_margin_m",
        ]
        missing = [key for key in required if key not in sample or not _finite(sample.get(key))]
        distance = _num(sample, "human_robot_distance_m")
        closing = _num(sample, "relative_closing_speed_mps")
        stopping = _num(sample, "robot_stopping_distance_m")
        margin = _num(sample, "configured_separation_margin_m")
        human_velocity = _num(sample, "human_velocity_mps", 0.0)
        robot_velocity = _num(sample, "robot_velocity_mps", 0.0)
        max_sensor_age = float(self.config.get("max_sensor_age_seconds", 0.25))
        min_sep = float(self.config.get("minimum_separation_m", 1.5))
        caution_sep = float(self.config.get("caution_separation_m", 2.5))
        critical_ttc = float(self.config.get("critical_ttc_seconds", 1.0))
        caution_ttc = float(self.config.get("caution_ttc_seconds", 3.0))
        stop_margin = float(self.config.get("stopping_distance_margin_m", 0.5))
        min_distance_bound = float(self.config.get("min_distance_bound_m", 0.0))
        max_distance_bound = float(self.config.get("max_distance_bound_m", 50.0))

        invalid_reasons = []
        if missing:
            invalid_reasons.append(f"missing_or_nonfinite:{','.join(missing)}")
        if distance is not None and not (min_distance_bound <= distance <= max_distance_bound):
            invalid_reasons.append("distance_out_of_bounds")
        if human_velocity is not None and abs(human_velocity) > float(self.config.get("max_human_speed_mps", 12.0)):
            invalid_reasons.append("human_velocity_out_of_bounds")
        if robot_velocity is not None and abs(robot_velocity) > float(self.config.get("max_robot_speed_mps", 12.0)):
            invalid_reasons.append("robot_velocity_out_of_bounds")
        if sensor_age > max_sensor_age:
            invalid_reasons.append("sensor_stale")

        ttc = None
        required_sep = None
        if distance is not None and closing is not None and closing > 1e-9:
            ttc = distance / closing
        if stopping is not None and margin is not None:
            required_sep = stopping + margin
        derived = {
            "time_to_collision_seconds": ttc,
            "required_separation_distance_m": required_sep,
            "sensor_age_seconds": sensor_age,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        validity = {"valid": not invalid_reasons, "reasons": invalid_reasons, "enabled": True}

        if bool(sample.get("emergency_input", False)):
            return PhysicalKernelResult(SafetyState.EMERGENCY_STOP, "EMERGENCY_INPUT_ACTIVE", 100, now, derived, validity)
        if invalid_reasons:
            reason = "PHYSICAL_SENSOR_STALE" if "sensor_stale" in invalid_reasons else "PHYSICAL_INPUT_INVALID"
            policy = str(self.config.get("invalid_input_policy", "degraded")).lower()
            state = SafetyState.PROTECTIVE_STOP if policy == "protective_stop" else SafetyState.DEGRADED
            return PhysicalKernelResult(state, reason, 70 if state == SafetyState.DEGRADED else 90, now, derived, validity)
        if bool(sample.get("protective_stop_input", False)):
            return PhysicalKernelResult(SafetyState.PROTECTIVE_STOP, "PROTECTIVE_INPUT_ACTIVE", 90, now, derived, validity)
        if distance is not None and distance < min_sep:
            return PhysicalKernelResult(SafetyState.EMERGENCY_STOP, "SEPARATION_MARGIN_LOW", 100, now, derived, validity)
        if ttc is not None and ttc <= critical_ttc:
            return PhysicalKernelResult(SafetyState.EMERGENCY_STOP, "TTC_CRITICAL", 100, now, derived, validity)
        if required_sep is not None and distance is not None and distance < required_sep + stop_margin:
            return PhysicalKernelResult(SafetyState.PROTECTIVE_STOP, "STOPPING_DISTANCE_VIOLATION", 90, now, derived, validity)
        if bool(sample.get("hazard_zone_active", False)) and distance is not None and distance < caution_sep:
            return PhysicalKernelResult(SafetyState.PROTECTIVE_STOP, "HAZARD_ZONE_INTRUSION", 90, now, derived, validity)
        if distance is not None and distance < caution_sep:
            return PhysicalKernelResult(SafetyState.CAUTION, "SEPARATION_MARGIN_LOW", 30, now, derived, validity)
        if ttc is not None and ttc <= caution_ttc:
            return PhysicalKernelResult(SafetyState.CAUTION, "TTC_CAUTION", 30, now, derived, validity)
        return PhysicalKernelResult(SafetyState.NORMAL, "PHYSICAL_CONDITIONS_NORMAL", 0, now, derived, validity)
