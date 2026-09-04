import time

import numpy as np

from go1_sim2real.observation import Go1ObservationBuilder
from go1_sim2real.runtime import run_control_loop
from go1_sim2real.safety import SafetySupervisor
from go1_sim2real.types import RobotState


class FaultTransport:
    def __init__(self):
        self.safe_reasons = []
        self.sent = []
        self.finished = False
        self.closed = False

    def read_state(self):
        return RobotState(
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.zeros(3),
            projected_gravity=[0, 0, -1],
            joint_pos=np.zeros(12),
            joint_vel=np.zeros(12),
            last_action=np.zeros(12),
            height_scan=np.zeros(187),
            timestamp=time.monotonic(),
            enable_switch=False,
        )

    def send_action(self, action):
        self.sent.append(np.asarray(action))

    def enter_safe_mode(self, reason):
        self.safe_reasons.append(reason)

    def finish_policy(self):
        self.finished = True

    def close(self):
        self.closed = True


def test_safety_fault_stops_immediately_and_skips_finish_transition():
    transport = FaultTransport()
    builder = Go1ObservationBuilder(
        default_joint_pos=np.zeros(12),
        terms=[
            "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
            "joint_pos", "joint_vel", "actions",
        ],
    )
    safety = SafetySupervisor({"require_enable_switch": True}, np.zeros(12))
    safety.set_enabled(True)
    calls = []

    run_control_loop(
        transport=transport,
        policy=lambda _observation: np.zeros(12),
        observation_builder=builder,
        safety=safety,
        command=np.zeros(3),
        control_dt=0.001,
        steps=100,
        on_step=lambda step, reason, *_rest: calls.append((step, reason)),
    )

    assert calls == [(0, "enable_switch_off")]
    assert transport.safe_reasons == ["enable_switch_off"]
    assert not transport.finished
    assert transport.closed
