"""运行层使用的机器人状态和动作类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np


def vector(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} 应为 {size} 个元素，实际为 {array.size}")
    return array


@dataclass
class RobotState:
    """一帧状态；所有角度使用弧度，速度使用 SI 单位。"""

    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    projected_gravity: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    last_action: np.ndarray
    height_scan: np.ndarray
    roll: float = 0.0
    pitch: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.base_lin_vel = vector(self.base_lin_vel, 3, "base_lin_vel")
        self.base_ang_vel = vector(self.base_ang_vel, 3, "base_ang_vel")
        self.projected_gravity = vector(self.projected_gravity, 3, "projected_gravity")
        self.joint_pos = vector(self.joint_pos, 12, "joint_pos")
        self.joint_vel = vector(self.joint_vel, 12, "joint_vel")
        self.last_action = vector(self.last_action, 12, "last_action")
        self.height_scan = vector(self.height_scan, 187, "height_scan")
