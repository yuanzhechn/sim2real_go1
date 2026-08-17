"""Isaac Lab Rough policy 的观测拼接和历史缓存。"""

from __future__ import annotations

from collections import deque

import numpy as np

from .types import RobotState, vector


class Go1ObservationBuilder:
    """生成当前项目训练时的 policy 观测。

    顺序来自 Isaac Lab 的 velocity_env_cfg.py：
    base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
    velocity_commands(3), joint_pos(12), joint_vel(12), actions(12),
    height_scan(187)。
    """

    def __init__(
        self,
        history_length: int = 1,
        height_scan_clip: float = 1.0,
        default_joint_pos: object | None = None,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length 必须大于 0")
        self.history_length = history_length
        self.height_scan_clip = float(height_scan_clip)
        self.default_joint_pos = (
            np.zeros(12, dtype=np.float32)
            if default_joint_pos is None
            else vector(default_joint_pos, 12, "default_joint_pos")
        )
        self._history: deque[np.ndarray] = deque(maxlen=history_length)

    @property
    def observation_dim(self) -> int:
        return 235 * self.history_length

    def reset(self) -> None:
        self._history.clear()

    def build(self, state: RobotState, command: object) -> np.ndarray:
        command_array = vector(command, 3, "velocity_command")
        height_scan = np.clip(state.height_scan, -self.height_scan_clip, self.height_scan_clip)
        observation = np.concatenate(
            (
                state.base_lin_vel,
                state.base_ang_vel,
                state.projected_gravity,
                command_array,
                state.joint_pos - self.default_joint_pos,
                state.joint_vel,
                state.last_action,
                height_scan,
            )
        ).astype(np.float32, copy=False)
        self._history.append(observation)
        while len(self._history) < self.history_length:
            self._history.appendleft(np.zeros_like(observation))
        return np.concatenate(tuple(self._history), axis=0)
