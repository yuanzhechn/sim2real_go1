"""机身速度以及可选真实地形扫描的外部状态入口。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time
from typing import Protocol

import numpy as np

from .types import vector


@dataclass(frozen=True)
class AuxiliaryState:
    """统一内部状态；Flat 策略会忽略占位的 height_scan 数组。"""

    base_lin_vel: np.ndarray
    height_scan: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_lin_vel", vector(self.base_lin_vel, 3, "base_lin_vel"))
        object.__setattr__(self, "height_scan", vector(self.height_scan, 187, "height_scan"))
        if not np.isfinite(self.timestamp):
            raise ValueError("auxiliary timestamp 必须是有限值")
        if not np.all(np.isfinite(self.base_lin_vel)) or not np.all(np.isfinite(self.height_scan)):
            raise ValueError("外部速度/高度扫描包含 NaN 或 Inf")


class AuxiliaryStateProvider(Protocol):
    def read(self) -> AuxiliaryState: ...
    def close(self) -> None: ...


class UdpJsonAuxiliaryStateProvider:
    """接收感知/里程计进程发来的一帧一包 JSON UDP 数据。

    Rough 数据格式为 ``{"base_lin_vel": [...], "height_scan": [187 values]}``；
    Flat policy 只要求 ``{"base_lin_vel": [vx, vy, vz]}``。
    时间戳使用本机收到数据包的单调时钟，避免混用不同机器的系统时钟。
    """

    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        timeout_s: float = 0.10,
        require_height_scan: bool = True,
    ) -> None:
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError("auxiliary timeout_s 必须大于 0")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((bind_host, int(bind_port)))
        self._socket.settimeout(self.timeout_s)
        self._latest: AuxiliaryState | None = None
        self.require_height_scan = bool(require_height_scan)

    def _decode(self, payload: bytes) -> AuxiliaryState:
        message = json.loads(payload.decode("utf-8"))
        if self.require_height_scan:
            height_scan = message["height_scan"]
        else:
            # Flat policy 不消费该字段；内部 RobotState 仍保持统一形状。
            height_scan = message.get("height_scan", [0.0] * 187)
        return AuxiliaryState(
            base_lin_vel=message["base_lin_vel"],
            height_scan=height_scan,
            timestamp=time.monotonic(),
        )

    def read(self) -> AuxiliaryState:
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._socket.settimeout(remaining)
            try:
                payload, _ = self._socket.recvfrom(65535)
            except socket.timeout:
                break
            try:
                self._latest = self._decode(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"外部感知 UDP 数据无效: {exc}") from exc
            # Drain queued packets without waiting so the newest frame is used.
            self._socket.setblocking(False)
            try:
                while True:
                    payload, _ = self._socket.recvfrom(65535)
                    self._latest = self._decode(payload)
            except BlockingIOError:
                pass
            finally:
                self._socket.setblocking(True)
            break
        if self._latest is None:
            raise TimeoutError("未收到外部辅助状态数据")
        if time.monotonic() - self._latest.timestamp > self.timeout_s:
            raise TimeoutError("外部辅助状态数据已超时")
        return self._latest

    def close(self) -> None:
        self._socket.close()
