"""Isaac Lab Go1 policy 的可配置观测拼接和历史缓存。"""

from __future__ import annotations

from collections import deque

import numpy as np

from .types import RobotState, vector


class Go1ObservationBuilder:
    """按训练配置的 terms 顺序生成 Rough(235) 或 Flat(48) 观测。"""

    TERM_DIMENSIONS = {
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "velocity_commands": 3,
        "joint_pos": 12,
        "joint_vel": 12,
        "actions": 12,
        "height_scan": 187,
    }
    ROUGH_TERMS = tuple(TERM_DIMENSIONS)

    def __init__(
        self,
        history_length: int = 1,
        height_scan_clip: float = 1.0,
        default_joint_pos: object | None = None,
        terms: object | None = None,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length 必须大于 0")
        self.history_length = history_length
        self.height_scan_clip = float(height_scan_clip)
        self.terms = tuple(self.ROUGH_TERMS if terms is None else terms)
        if not self.terms or len(set(self.terms)) != len(self.terms):
            raise ValueError("observation.terms 不能为空或包含重复项")
        unknown = set(self.terms).difference(self.TERM_DIMENSIONS)
        if unknown:
            raise ValueError(f"未知 observation terms: {sorted(unknown)}")
        self.default_joint_pos = (
            np.zeros(12, dtype=np.float32)
            if default_joint_pos is None
            else vector(default_joint_pos, 12, "default_joint_pos")
        )
        self._history: deque[np.ndarray] = deque(maxlen=history_length)

    @property
    def observation_dim(self) -> int:
        return sum(self.TERM_DIMENSIONS[name] for name in self.terms) * self.history_length

    def reset(self) -> None:
        self._history.clear()

    def build(self, state: RobotState, command: object) -> np.ndarray:
        command_array = vector(command, 3, "velocity_command")
        values = {
            "base_lin_vel": state.base_lin_vel,
            "base_ang_vel": state.base_ang_vel,
            "projected_gravity": state.projected_gravity,
            "velocity_commands": command_array,
            "joint_pos": state.joint_pos - self.default_joint_pos,
            "joint_vel": state.joint_vel,
            "actions": state.last_action,
            "height_scan": np.clip(
                state.height_scan, -self.height_scan_clip, self.height_scan_clip
            ),
        }
        observation = np.concatenate(tuple(values[name] for name in self.terms)).astype(
            np.float32, copy=False
        )
        self._history.append(observation)
        while len(self._history) < self.history_length:
            self._history.appendleft(np.zeros_like(observation))
        return np.concatenate(tuple(self._history), axis=0)
