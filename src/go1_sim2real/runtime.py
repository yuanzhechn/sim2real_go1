"""策略控制循环。"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .observation import Go1ObservationBuilder
from .safety import SafetySupervisor
from .transport import RobotTransport
from .types import RobotState


def run_control_loop(
    transport: RobotTransport,
    policy: Callable[[object], object],
    observation_builder: Go1ObservationBuilder,
    safety: SafetySupervisor,
    command: object,
    control_dt: float,
    steps: int | None = None,
    action_clip: float = 1.0,
    on_step: Callable[[int, str, np.ndarray, RobotState], None] | None = None,
) -> None:
    if control_dt <= 0:
        raise ValueError("control_dt 必须大于 0")
    command_array = np.asarray(command, dtype=np.float32).reshape(-1)
    if command_array.size != 3 or not np.all(np.isfinite(command_array)):
        raise ValueError("速度指令必须是有限的 3 维向量")
    observation_builder.reset()
    # TorchScript/BLAS 首次调用可能包含明显的懒加载开销。必须在读取带时效性的
    # LowState 之前预热，否则第一帧会被安全层正确判为 state_timeout。
    warmup_action = np.asarray(
        policy(np.zeros(observation_builder.observation_dim, dtype=np.float32)), dtype=np.float32
    ).reshape(-1)
    if warmup_action.size != 12 or not np.all(np.isfinite(warmup_action)):
        raise ValueError("策略预热输出必须是有限的 12 维向量")
    step = 0
    next_tick = time.monotonic()
    try:
        while steps is None or step < steps:
            state = transport.read_state()
            observation = observation_builder.build(state, command_array)
            action = np.asarray(policy(observation), dtype=np.float32).reshape(-1)
            if action.size != 12 or not np.all(np.isfinite(action)):
                raise ValueError(f"策略动作必须是有限的 12 维向量，实际维度为 {action.size}")
            action = np.clip(action, -action_clip, action_clip)
            filtered_action, result = safety.filter_action(state, action)
            if result.allowed:
                transport.send_action(filtered_action)
            else:
                enter_safe_mode = getattr(transport, "enter_safe_mode", None)
                if enter_safe_mode is not None:
                    enter_safe_mode(result.reason)
                else:
                    transport.send_action(filtered_action)
            if on_step is not None:
                on_step(step, result.reason, filtered_action, state)
            step += 1
            next_tick += control_dt
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        transport.close()
