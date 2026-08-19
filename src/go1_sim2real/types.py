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
    auxiliary_timestamp: float | None = None
    motor_temperatures: np.ndarray | None = None
    battery_voltage: float | None = None
    motor_modes: np.ndarray | None = None
    enable_switch: bool | None = None
    emergency_stop: bool = False
    communication_ok: bool = True
    remote_axes: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.base_lin_vel = vector(self.base_lin_vel, 3, "base_lin_vel")
        self.base_ang_vel = vector(self.base_ang_vel, 3, "base_ang_vel")
        self.projected_gravity = vector(self.projected_gravity, 3, "projected_gravity")
        self.joint_pos = vector(self.joint_pos, 12, "joint_pos")
        self.joint_vel = vector(self.joint_vel, 12, "joint_vel")
        self.last_action = vector(self.last_action, 12, "last_action")
        self.height_scan = vector(self.height_scan, 187, "height_scan")
        if self.motor_temperatures is not None:
            self.motor_temperatures = vector(self.motor_temperatures, 12, "motor_temperatures")
        if self.motor_modes is not None:
            self.motor_modes = vector(self.motor_modes, 12, "motor_modes")
        if self.remote_axes is not None:
            self.remote_axes = vector(self.remote_axes, 4, "remote_axes")
        scalar_values = (
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.joint_pos,
            self.joint_vel,
            self.last_action,
            self.height_scan,
        )
        if not all(np.all(np.isfinite(value)) for value in scalar_values):
            raise ValueError("RobotState 包含 NaN 或 Inf")
        scalar_state = [self.roll, self.pitch, self.timestamp]
        if self.auxiliary_timestamp is not None:
            scalar_state.append(self.auxiliary_timestamp)
        if self.battery_voltage is not None:
            scalar_state.append(self.battery_voltage)
        if self.remote_axes is not None and not np.all(np.isfinite(self.remote_axes)):
            raise ValueError("RobotState 遥控摇杆包含 NaN 或 Inf")
        if not np.all(np.isfinite(np.asarray(scalar_state, dtype=np.float64))):
            raise ValueError("RobotState 标量状态包含 NaN 或 Inf")
