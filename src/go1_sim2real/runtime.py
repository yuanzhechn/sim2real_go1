"""策略控制循环。"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .observation import Go1ObservationBuilder
from .safety import SafetySupervisor
from .transport import RobotTransport


def run_control_loop(
    transport: RobotTransport,
    policy: Callable[[object], object],
    observation_builder: Go1ObservationBuilder,
    safety: SafetySupervisor,
    command: object,
    control_dt: float,
    steps: int | None = None,
    action_clip: float = 1.0,
    on_step: Callable[[int, str, np.ndarray], None] | None = None,
) -> None:
    observation_builder.reset()
    step = 0
    next_tick = time.monotonic()
    try:
        while steps is None or step < steps:
            state = transport.read_state()
            observation = observation_builder.build(state, command)
            action = np.asarray(policy(observation), dtype=np.float32).reshape(-1)
            if action.size != 12:
                raise ValueError(f"策略动作应为 12 维，实际为 {action.size}")
            action = np.clip(action, -action_clip, action_clip)
            filtered_action, result = safety.filter_action(state, action)
            transport.send_action(filtered_action)
            if on_step is not None:
                on_step(step, result.reason, filtered_action)
            step += 1
            next_tick += control_dt
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        transport.close()
