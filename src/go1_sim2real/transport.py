"""机器人通信抽象。真实 SDK 接入必须显式实现并经过硬件测试。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Protocol

import numpy as np

from .types import RobotState


class RobotTransport(Protocol):
    def read_state(self) -> RobotState: ...
    def send_action(self, action: np.ndarray) -> None: ...
    def close(self) -> None: ...


@dataclass
class DryRunTransport:
    default_joint_pos: np.ndarray
    last_action: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.default_joint_pos = np.asarray(self.default_joint_pos, dtype=np.float32).reshape(12)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.sent_actions: list[np.ndarray] = []

    def read_state(self) -> RobotState:
        return RobotState(
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.zeros(3),
            projected_gravity=np.array([0.0, 0.0, -1.0]),
            joint_pos=self.default_joint_pos,
            joint_vel=np.zeros(12),
            last_action=self.last_action,
            # Dry-run cannot provide terrain perception. It is only for API tests.
            height_scan=np.zeros(187),
            timestamp=time.monotonic(),
        )

    def send_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).reshape(12).copy()
        self.sent_actions.append(self.last_action.copy())

    def close(self) -> None:
        return None


class JsonlTransport:
    """从一帧一行的 JSONL 状态日志回放，动作只写入内存不发到网络。"""

    def __init__(self, path: str, loop: bool = False) -> None:
        with open(path, "r", encoding="utf-8") as stream:
            self._frames = [json.loads(line) for line in stream if line.strip()]
        if not self._frames:
            raise ValueError(f"状态日志为空: {path}")
        self._index = 0
        self._loop = loop
        self.sent_actions: list[np.ndarray] = []

    def read_state(self) -> RobotState:
        frame = self._frames[self._index]
        if self._index + 1 < len(self._frames):
            self._index += 1
        elif self._loop:
            self._index = 0
        return RobotState(
            base_lin_vel=frame["base_lin_vel"],
            base_ang_vel=frame["base_ang_vel"],
            projected_gravity=frame["projected_gravity"],
            joint_pos=frame["joint_pos"],
            joint_vel=frame["joint_vel"],
            last_action=frame["last_action"],
            height_scan=frame["height_scan"],
            roll=float(frame.get("roll", 0.0)),
            pitch=float(frame.get("pitch", 0.0)),
            timestamp=float(frame.get("timestamp", time.monotonic())),
        )

    def send_action(self, action: np.ndarray) -> None:
        self.sent_actions.append(np.asarray(action, dtype=np.float32).reshape(12).copy())

    def close(self) -> None:
        return None


class UnitreeSdkTransport:
    """Unitree SDK 适配入口，避免在未完成适配时误发控制包。"""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "尚未绑定具体 Unitree Go1 SDK。请在此类中实现状态读取、关节顺序转换和低层动作发送，"
            "并保留上层 SafetySupervisor。"
        )
