"""Adapters for fast and slow ML results used by the safety state machine."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from .physical_kernel import SafetyState


@dataclass(frozen=True)
class FastAdapterResult:
    ready: bool
    timestamp: float
    probability: float | None
    threshold: float
    raw_positive: bool
    persistent_positive: bool
    consecutive_positive_count: int
    latency_ms: float
    model_version: str
    input_quality: str
    reason_code: str
    requested_state: SafetyState


@dataclass(frozen=True)
class SlowAdapterResult:
    ready: bool
    prediction_timestamp: float
    probability_by_horizon: dict[float, float]
    calibrated_probability_by_horizon: dict[float, float]
    threshold_by_horizon: dict[float, float]
    latency_ms: float
    model_version: str
    result_age_seconds: float
    input_quality: str
    reason_code: str
    requested_state: SafetyState
    recommendations: dict[str, Any]


def _valid_probability(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    except Exception:
        return False


class FastDetectorAdapter:
    """Continuous fast-detector policy with hysteresis and persistence."""

    def __init__(self, config: dict[str, Any] | None = None, model_version: str = "fast_policy_v1") -> None:
        self.config = dict(config or {})
        self.model_version = model_version
        self.positive_count = 0
        self.negative_count = 0
        self.persistent_positive = False
        self.last_positive_timestamp: float | None = None

    def update(self, sample: dict[str, Any]) -> FastAdapterResult:
        started = time.perf_counter()
        timestamp = float(sample.get("timestamp", sample.get("prediction_timestamp", 0.0)))
        activation = float(self.config.get("activation_threshold", 0.70))
        release = float(self.config.get("release_threshold", 0.45))
        if release >= activation:
            raise ValueError("fast_policy.release_threshold must be below activation_threshold.")
        pos_required = int(self.config.get("consecutive_positive_required", 3))
        neg_required = int(self.config.get("consecutive_negative_required", 5))
        cooldown = float(self.config.get("cooldown_seconds", 3.0))
        prob = sample.get("probability")
        if not _valid_probability(prob):
            policy = str(self.config.get("invalid_probability_policy", "degraded")).lower()
            state = SafetyState.DEGRADED if policy == "degraded" else SafetyState.NORMAL
            return FastAdapterResult(False, timestamp, None, activation, False, self.persistent_positive, self.positive_count, (time.perf_counter() - started) * 1000.0, self.model_version, "invalid_probability", "FAST_INVALID_PROBABILITY", state)
        probability = float(prob)
        raw_positive = probability >= activation if not self.persistent_positive else probability > release
        if probability >= activation:
            self.positive_count += 1
            self.negative_count = 0
        elif probability <= release:
            self.negative_count += 1
            self.positive_count = 0
        if self.positive_count >= pos_required and not self.persistent_positive:
            self.persistent_positive = True
            self.last_positive_timestamp = timestamp
        if self.persistent_positive and self.negative_count >= neg_required:
            if self.last_positive_timestamp is None or timestamp - self.last_positive_timestamp >= cooldown:
                self.persistent_positive = False
        requested = SafetyState.NORMAL
        reason = "FAST_NORMAL"
        high_duration = float(self.config.get("controlled_stop_after_seconds", 999999.0))
        if self.persistent_positive:
            requested = SafetyState.HIGH_ALERT
            reason = "FAST_PERSISTENT_POSITIVE"
            if self.last_positive_timestamp is not None and timestamp - self.last_positive_timestamp >= high_duration:
                requested = SafetyState.CONTROLLED_STOP
                reason = "FAST_CONTROLLED_STOP_PERSISTENCE"
        return FastAdapterResult(True, timestamp, probability, activation, raw_positive, self.persistent_positive, self.positive_count, (time.perf_counter() - started) * 1000.0, self.model_version, "valid", reason, requested)


class SlowForecasterAdapter:
    """Policy adapter for experimental physiological stress forecasts."""

    def __init__(self, config: dict[str, Any] | None = None, model_version: str = "slow_policy_v1") -> None:
        self.config = dict(config or {})
        self.model_version = model_version
        self.positive_counts: dict[float, int] = {}
        self.negative_counts: dict[float, int] = {}
        self.persistent: dict[float, bool] = {}

    def update(self, sample: dict[str, Any], current_timestamp: float | None = None) -> SlowAdapterResult:
        started = time.perf_counter()
        now = float(current_timestamp if current_timestamp is not None else sample.get("timestamp", sample.get("prediction_timestamp", 0.0)))
        pred_ts = float(sample.get("prediction_timestamp", sample.get("timestamp", now)))
        probs_raw = sample.get("probability_by_horizon", {})
        probs = {float(k): float(v) for k, v in probs_raw.items() if _valid_probability(v)}
        if len(probs) != len(probs_raw):
            policy = str(self.config.get("invalid_probability_policy", "ignore_and_log")).lower()
            state = SafetyState.DEGRADED if policy == "degraded" else SafetyState.NORMAL
            return SlowAdapterResult(False, pred_ts, {}, {}, {}, (time.perf_counter() - started) * 1000.0, self.model_version, now - pred_ts, "invalid_probability", "SLOW_INVALID_PROBABILITY", state, {})
        result_age = max(0.0, now - pred_ts)
        if result_age > float(self.config.get("max_result_age_seconds", 3.0)):
            return SlowAdapterResult(False, pred_ts, probs, probs, {}, (time.perf_counter() - started) * 1000.0, self.model_version, result_age, "stale", "SLOW_RESULT_STALE", SafetyState.NORMAL, {})
        thresholds: dict[float, float] = {}
        requested = SafetyState.NORMAL
        reason = "SLOW_NORMAL"
        recommendations: dict[str, Any] = {}
        for horizon, probability in probs.items():
            block = self.config.get(f"horizon_{int(horizon)}s", self.config.get(f"horizon_{horizon:g}s", {}))
            activation = float(block.get("activation_threshold", 0.75 if horizon <= 5 else 0.70))
            release = float(block.get("release_threshold", 0.50 if horizon <= 5 else 0.45))
            thresholds[horizon] = activation
            if release >= activation:
                raise ValueError("slow_policy release_threshold must be below activation_threshold.")
            if probability >= activation:
                self.positive_counts[horizon] = self.positive_counts.get(horizon, 0) + 1
                self.negative_counts[horizon] = 0
            elif probability <= release:
                self.negative_counts[horizon] = self.negative_counts.get(horizon, 0) + 1
                self.positive_counts[horizon] = 0
            required = int(block.get("consecutive_positive_required", 2 if horizon <= 5 else 3))
            if self.positive_counts.get(horizon, 0) >= required:
                self.persistent[horizon] = True
            if self.persistent.get(horizon, False):
                requested = SafetyState.CAUTION
                if horizon <= 5:
                    reason = "SLOW_5S_PERSISTENT_FORECAST"
                    recommendations.update({"maximum_speed_scale": 0.75, "additional_separation_margin_m": 0.25, "increase_monitoring_sensitivity": True})
                else:
                    reason = "SLOW_30S_PERSISTENT_FORECAST"
                    recommendations.update({"recommended_pause_at_task_boundary": True})
        return SlowAdapterResult(True, pred_ts, probs, probs, thresholds, (time.perf_counter() - started) * 1000.0, self.model_version, result_age, "valid", reason, requested, recommendations)
