from __future__ import annotations

import time
import numpy as np

from .transport import RobotState


class SafetySupervisor:
    def __init__(self, config: dict, default_joint_pos):
        self.config = config
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).reshape(12)
        self.enabled = not bool(config.get("require_enable_switch", True))
        self.last_action = np.zeros(12, dtype=np.float32)

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def filter(self, state: RobotState, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        reason = "ok"
        allowed = self.enabled
        if not allowed:
            reason = "enable_switch_off"
        elif time.monotonic() - state.timestamp > float(self.config.get("stale_state_timeout_s", 0.15)):
            allowed, reason = False, "state_timeout"
        elif abs(state.roll) > float(self.config.get("max_roll_rad", 0.7)):
            allowed, reason = False, "roll_limit"
        elif abs(state.pitch) > float(self.config.get("max_pitch_rad", 0.7)):
            allowed, reason = False, "pitch_limit"
        elif action.size != 12 or not np.all(np.isfinite(action)):
            allowed, reason = False, "invalid_action"
        if not allowed:
            return np.zeros(12, dtype=np.float32), reason
        action = np.clip(action, -1.0, 1.0)
        delta = float(self.config.get("max_action_delta", 0.25))
        original = action.copy()
        action = np.clip(action, self.last_action - delta, self.last_action + delta)
        self.last_action = action.copy()
        return action, "action_delta_limited" if np.max(np.abs(action - original)) > 1e-6 else reason
