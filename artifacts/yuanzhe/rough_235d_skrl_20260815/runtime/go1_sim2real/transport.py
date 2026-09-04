from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class RobotState:
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    projected_gravity: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    last_action: np.ndarray
    height_scan: np.ndarray
    roll: float = 0.0
    pitch: float = 0.0
    timestamp: float = 0.0


@dataclass
class DryRunTransport:
    default_joint_pos: np.ndarray

    def __post_init__(self):
        self.default_joint_pos = np.asarray(self.default_joint_pos, dtype=np.float32).reshape(12)
        self.last_action = np.zeros(12, dtype=np.float32)

    def read_state(self) -> RobotState:
        return RobotState(np.zeros(3), np.zeros(3), np.array([0, 0, -1], dtype=np.float32), self.default_joint_pos.copy(), np.zeros(12), self.last_action.copy(), np.zeros(187), timestamp=time.monotonic())

    def send_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).reshape(12).copy()

    def close(self) -> None:
        pass


class UnitreeSdkTransport:
    """必须由用户按实际 Unitree SDK 实现；默认禁止误发控制包。"""

    def __init__(self, **kwargs):
        raise NotImplementedError("请实现 UnitreeSdkTransport 的状态读取和动作发送")
