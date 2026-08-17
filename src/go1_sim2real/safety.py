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

    def validate(self, state: RobotState, action: object) -> SafetyResult:
        now = time.monotonic()
        if not self._enabled:
            return SafetyResult(False, "enable_switch_off")
        if now - state.timestamp > float(self.config.get("stale_state_timeout_s", 0.15)):
            return SafetyResult(False, "state_timeout")
        if abs(state.roll) > float(self.config.get("max_roll_rad", 0.7)):
            return SafetyResult(False, "roll_limit")
        if abs(state.pitch) > float(self.config.get("max_pitch_rad", 0.7)):
            return SafetyResult(False, "pitch_limit")
        if np.max(np.abs(state.joint_pos - self.default_joint_pos)) > float(
            self.config.get("max_joint_error_rad", 1.2)
        ):
            return SafetyResult(False, "joint_error_limit")
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
        return np.zeros(12, dtype=np.float32), result
