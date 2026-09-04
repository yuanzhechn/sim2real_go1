from __future__ import annotations

from collections import deque
import numpy as np

from .transport import RobotState


class ObservationBuilder:
    def __init__(self, default_joint_pos, history_length: int = 1, height_scan_clip: float = 1.0):
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).reshape(12)
        self.history_length = int(history_length)
        self.height_scan_clip = float(height_scan_clip)
        self.history = deque(maxlen=self.history_length)

    def reset(self):
        self.history.clear()

    def build(self, state: RobotState, command) -> np.ndarray:
        observation = np.concatenate((state.base_lin_vel, state.base_ang_vel, state.projected_gravity, np.asarray(command, dtype=np.float32).reshape(3), state.joint_pos - self.default_joint_pos, state.joint_vel, state.last_action, np.clip(state.height_scan, -self.height_scan_clip, self.height_scan_clip))).astype(np.float32)
        self.history.append(observation)
        while len(self.history) < self.history_length:
            self.history.appendleft(np.zeros_like(observation))
        return np.concatenate(tuple(self.history))
