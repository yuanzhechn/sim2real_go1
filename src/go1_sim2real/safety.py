"""真机前的动作和状态安全门。"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .types import RobotState


@dataclass(frozen=True)
class SafetyResult:
    allowed: bool
    reason: str = "ok"


class SafetySupervisor:
    def __init__(self, config: dict, default_joint_pos: object) -> None:
        self.config = config
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)
        if self.default_joint_pos.size != 12:
            raise ValueError("default_joint_pos 必须包含 12 个角度")
        self._enabled = not bool(config.get("require_enable_switch", True))
        self._last_action = np.zeros(12, dtype=np.float32)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def reset(self) -> None:
        self._last_action.fill(0.0)

    def validate(self, state: RobotState, action: object) -> SafetyResult:
        now = time.monotonic()
        if state.emergency_stop:
            return SafetyResult(False, "emergency_stop")
        if not state.communication_ok:
            return SafetyResult(False, "communication_error")
        if not self._enabled:
            return SafetyResult(False, "enable_switch_off")
        if state.enable_switch is False:
            return SafetyResult(False, "enable_switch_off")
        if now - state.timestamp > float(self.config.get("stale_state_timeout_s", 0.15)):
            return SafetyResult(False, "state_timeout")
        if state.auxiliary_timestamp is not None and now - state.auxiliary_timestamp > float(
            self.config.get("stale_auxiliary_timeout_s", 0.15)
        ):
            return SafetyResult(False, "auxiliary_state_timeout")
        if abs(state.roll) > float(self.config.get("max_roll_rad", 0.7)):
            return SafetyResult(False, "roll_limit")
        if abs(state.pitch) > float(self.config.get("max_pitch_rad", 0.7)):
            return SafetyResult(False, "pitch_limit")
        if np.max(np.abs(state.joint_pos - self.default_joint_pos)) > float(
            self.config.get("max_joint_error_rad", 1.2)
        ):
            return SafetyResult(False, "joint_error_limit")
        if np.max(np.abs(state.joint_vel)) > float(self.config.get("max_joint_velocity_rad_s", 30.0)):
            return SafetyResult(False, "joint_velocity_limit")
        if state.motor_temperatures is not None and np.max(state.motor_temperatures) > float(
            self.config.get("max_motor_temperature_c", 70.0)
        ):
            return SafetyResult(False, "motor_temperature_limit")
        if state.battery_voltage is not None and state.battery_voltage < float(
            self.config.get("min_battery_voltage_v", 19.0)
        ):
            return SafetyResult(False, "battery_voltage_limit")
        if state.motor_modes is not None and np.any(state.motor_modes == 0x08):
            return SafetyResult(False, "motor_overheat_fault")
        if state.motor_modes is not None:
            allowed_modes = np.asarray(self.config.get("allowed_motor_modes", [0x00, 0x0A]))
            if np.any(~np.isin(state.motor_modes, allowed_modes)):
                return SafetyResult(False, "motor_mode_fault")
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != 12 or not np.all(np.isfinite(action_array)):
            return SafetyResult(False, "invalid_action")
        return SafetyResult(True)

    def filter_action(self, state: RobotState, action: object) -> tuple[np.ndarray, SafetyResult]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        result = self.validate(state, action_array)
        if result.allowed:
            clipped = np.clip(action_array, -1.0, 1.0)
            max_delta = float(self.config.get("max_action_delta", 0.25))
            limited = np.clip(clipped, self._last_action - max_delta, self._last_action + max_delta)
            self._last_action = limited.copy()
            if not np.allclose(limited, clipped):
                return limited, SafetyResult(True, "action_delta_limited")
            return limited, result
        # 归一化零动作为站立目标；发生故障时不继续跟踪策略动作。
        self._last_action.fill(0.0)
        return np.zeros(12, dtype=np.float32), result
