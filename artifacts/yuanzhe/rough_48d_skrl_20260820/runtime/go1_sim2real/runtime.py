from __future__ import annotations

import time
import numpy as np


def run_loop(transport, policy, observations, safety, command, control_dt: float, steps: int = 100, action_clip: float = 1.0):
    observations.reset()
    try:
        for step in range(steps):
            state = transport.read_state()
            value = observations.build(state, command)
            action = np.asarray(policy(value), dtype=np.float32).reshape(12)
            action, reason = safety.filter(state, np.clip(action, -action_clip, action_clip))
            transport.send_action(action)
            if step < 5 or step % 50 == 0:
                print(f"step={step:06d} safety={reason} action_max={np.max(np.abs(action)):.3f}")
            time.sleep(control_dt)
    finally:
        transport.close()
