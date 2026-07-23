"""Supervisory decision state machine with explicit authority ordering."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .adapters import FastAdapterResult, SlowAdapterResult
from .physical_kernel import PhysicalKernelResult, SafetyState


@dataclass(frozen=True)
class SupervisoryAction:
    action_name: str
    priority: int
    source_layer: str
    reason_code: str
    parameters: dict[str, Any]
    timestamp: float


@dataclass(frozen=True)
class TransitionResult:
    previous_state: SafetyState
    new_state: SafetyState
    requested_action: SupervisoryAction
    trigger_layer: str
    reason_code: str
    timestamp: float
    latched: bool
    transition_allowed: bool
    policy_snapshot: dict[str, Any]
    latency_ms: float


STATE_PRIORITY = {
    SafetyState.NORMAL: 0,
    SafetyState.CAUTION: 10,
    SafetyState.HIGH_ALERT: 20,
    SafetyState.CONTROLLED_STOP: 30,
    SafetyState.DEGRADED: 40,
    SafetyState.PROTECTIVE_STOP: 80,
    SafetyState.EMERGENCY_STOP: 100,
}


def action_for_state(state: SafetyState, layer: str, reason: str, timestamp: float, params: dict[str, Any] | None = None) -> SupervisoryAction:
    action_map = {
        SafetyState.NORMAL: "continue_normal",
        SafetyState.CAUTION: "reduce_speed",
        SafetyState.HIGH_ALERT: "request_operator_attention",
        SafetyState.CONTROLLED_STOP: "controlled_stop",
        SafetyState.DEGRADED: "controlled_stop",
        SafetyState.PROTECTIVE_STOP: "protective_stop",
        SafetyState.EMERGENCY_STOP: "emergency_stop",
    }
    return SupervisoryAction(action_map[state], STATE_PRIORITY[state], layer, reason, params or {}, timestamp)


class SafetyStateMachine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.state = SafetyState.NORMAL
        self.emergency_latched = False
        self.emergency_clear_since: float | None = None
        self.last_transition_timestamp: float | None = None

    def reset_session(self) -> None:
        self.state = SafetyState.NORMAL
        self.emergency_latched = False
        self.emergency_clear_since = None
        self.last_transition_timestamp = None

    def update(
        self,
        physical_kernel_result: PhysicalKernelResult | None,
        fast_detector_result: FastAdapterResult | None,
        slow_forecaster_result: SlowAdapterResult | None,
        current_timestamp: float,
        explicit_reset_command: bool = False,
    ) -> TransitionResult:
        started = time.perf_counter()
        previous = self.state
        policy = dict(self.config)
        emergency_reset_hold = float(self.config.get("emergency_reset_hold_seconds", 1.0))
        cooldown = float(self.config.get("cooldown_seconds", 3.0))

        physical_state = physical_kernel_result.requested_state if physical_kernel_result else SafetyState.NORMAL
        physical_reason = physical_kernel_result.reason_code if physical_kernel_result else "PHYSICAL_NOT_AVAILABLE"
        physical_valid = bool(physical_kernel_result.input_validity.get("valid", True)) if physical_kernel_result else True

        if self.emergency_latched or previous == SafetyState.EMERGENCY_STOP:
            condition_clear = physical_state != SafetyState.EMERGENCY_STOP and physical_valid
            if condition_clear:
                if self.emergency_clear_since is None:
                    self.emergency_clear_since = current_timestamp
            else:
                self.emergency_clear_since = None
            can_reset = bool(explicit_reset_command and condition_clear and self.emergency_clear_since is not None and current_timestamp - self.emergency_clear_since >= emergency_reset_hold)
            if can_reset:
                self.emergency_latched = False
                self.state = SafetyState.NORMAL
                return self._result(previous, self.state, "physical", "EMERGENCY_RESET_ACCEPTED", current_timestamp, False, True, policy, started)
            self.emergency_latched = True
            self.state = SafetyState.EMERGENCY_STOP
            reason = "EMERGENCY_LATCHED_RESET_REQUIRED" if not explicit_reset_command else "EMERGENCY_RESET_REJECTED"
            return self._result(previous, self.state, "physical", reason, current_timestamp, True, False, policy, started)

        candidates: list[tuple[SafetyState, str, str, dict[str, Any]]] = []
        if physical_kernel_result is not None:
            candidates.append((physical_state, "physical", physical_reason, dict(physical_kernel_result.derived_values)))
        if fast_detector_result is not None and fast_detector_result.ready:
            candidates.append((fast_detector_result.requested_state, "fast", fast_detector_result.reason_code, {}))
        elif fast_detector_result is not None and fast_detector_result.requested_state == SafetyState.DEGRADED:
            candidates.append((SafetyState.DEGRADED, "fast", fast_detector_result.reason_code, {}))
        if slow_forecaster_result is not None and slow_forecaster_result.ready:
            candidates.append((slow_forecaster_result.requested_state, "slow", slow_forecaster_result.reason_code, slow_forecaster_result.recommendations))
        elif slow_forecaster_result is not None and slow_forecaster_result.requested_state == SafetyState.DEGRADED:
            candidates.append((SafetyState.DEGRADED, "slow", slow_forecaster_result.reason_code, {}))
        if not candidates:
            candidates.append((SafetyState.NORMAL, "state_machine", "NO_INPUTS_AVAILABLE", {}))

        requested, layer, reason, params = sorted(candidates, key=lambda item: STATE_PRIORITY[item[0]], reverse=True)[0]
        if requested == SafetyState.EMERGENCY_STOP:
            self.emergency_latched = bool(self.config.get("emergency_latching", True))
            self.state = SafetyState.EMERGENCY_STOP
            return self._result(previous, self.state, layer, reason, current_timestamp, self.emergency_latched, True, policy, started, params)

        higher_active = requested in {SafetyState.PROTECTIVE_STOP, SafetyState.DEGRADED, SafetyState.CONTROLLED_STOP, SafetyState.HIGH_ALERT, SafetyState.CAUTION}
        if previous != requested and not higher_active and self.last_transition_timestamp is not None:
            if current_timestamp - self.last_transition_timestamp < cooldown:
                requested, layer, reason, params = previous, "state_machine", "RECOVERY_COOLDOWN_ACTIVE", {}
        if previous in {SafetyState.PROTECTIVE_STOP, SafetyState.DEGRADED, SafetyState.CONTROLLED_STOP, SafetyState.HIGH_ALERT, SafetyState.CAUTION} and STATE_PRIORITY[requested] < STATE_PRIORITY[previous]:
            if self.last_transition_timestamp is not None and current_timestamp - self.last_transition_timestamp < cooldown:
                requested, layer, reason, params = previous, "state_machine", "RECOVERY_COOLDOWN_ACTIVE", {}
        transition_allowed = requested != previous
        if transition_allowed:
            self.last_transition_timestamp = current_timestamp
        self.state = requested
        return self._result(previous, self.state, layer, reason, current_timestamp, False, transition_allowed, policy, started, params)

    def _result(
        self,
        previous: SafetyState,
        new: SafetyState,
        layer: str,
        reason: str,
        timestamp: float,
        latched: bool,
        allowed: bool,
        policy: dict[str, Any],
        started: float,
        params: dict[str, Any] | None = None,
    ) -> TransitionResult:
        return TransitionResult(
            previous,
            new,
            action_for_state(new, layer, reason, timestamp, params),
            layer,
            reason,
            timestamp,
            latched,
            allowed,
            policy,
            (time.perf_counter() - started) * 1000.0,
        )
