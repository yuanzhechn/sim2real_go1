"""Unitree 遥控摇杆到训练速度指令的映射。"""

from __future__ import annotations

import numpy as np

from .types import RobotState


class RemoteVelocityCommand:
    """将 ``[lx, ly, rx, ry]`` 映射为 ``[vx, vy, yaw_rate]``。"""

    def __init__(self, config: dict) -> None:
        self.deadband = float(config.get("deadband", 0.08))
        self.scales = np.asarray(config.get("scales", [0.5, 0.3, 0.5]), dtype=np.float32).reshape(3)
        self.signs = np.asarray(config.get("signs", [1.0, -1.0, -1.0]), dtype=np.float32).reshape(3)
        if not 0.0 <= self.deadband < 1.0:
            raise ValueError("remote_control.deadband 必须在 [0, 1) 内")
        if np.any(self.scales < 0.0) or not np.all(np.isfinite(self.scales)):
            raise ValueError("remote_control.scales 必须是有限非负数")
        if np.any(~np.isin(self.signs, (-1.0, 1.0))):
            raise ValueError("remote_control.signs 只能为 -1 或 1")

    def _deadband(self, values: np.ndarray) -> np.ndarray:
        magnitude = np.abs(values)
        scaled = np.maximum(magnitude - self.deadband, 0.0) / (1.0 - self.deadband)
        return np.sign(values) * scaled

    def __call__(self, state: RobotState) -> np.ndarray:
        if state.remote_axes is None:
            raise RuntimeError("LowState 未提供遥控摇杆数据")
        lx, ly, rx, _ry = np.clip(state.remote_axes, -1.0, 1.0)
        raw = np.asarray([ly, lx, rx], dtype=np.float32)
        return self._deadband(raw) * self.scales * self.signs
