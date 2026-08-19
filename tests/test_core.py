import numpy as np

from go1_sim2real.observation import Go1ObservationBuilder
from go1_sim2real.safety import SafetySupervisor
from go1_sim2real.transport import DryRunTransport


def test_observation_shape_and_order():
    pose = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
    state = DryRunTransport(pose).read_state()
    obs = Go1ObservationBuilder(default_joint_pos=pose).build(state, [0.2, 0.0, 0.1])
    assert obs.shape == (235,)
    np.testing.assert_allclose(obs[0:3], 0.0)
    np.testing.assert_allclose(obs[9:12], [0.2, 0.0, 0.1])
    np.testing.assert_allclose(obs[12:24], 0.0)


def test_flat_observation_omits_height_scan():
    pose = np.zeros(12, dtype=np.float32)
    state = DryRunTransport(pose).read_state()
    terms = [
        "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
        "joint_pos", "joint_vel", "actions",
    ]
    obs = Go1ObservationBuilder(default_joint_pos=pose, terms=terms).build(state, [0.1, 0.0, 0.0])
    assert obs.shape == (48,)
    np.testing.assert_allclose(obs[9:12], [0.1, 0.0, 0.0])


def test_safety_rejects_disabled_and_large_action():
    pose = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
    state = DryRunTransport(pose).read_state()
    safety = SafetySupervisor({"require_enable_switch": True}, pose)
    _, result = safety.filter_action(state, np.ones(12))
    assert not result.allowed
    assert result.reason == "enable_switch_off"


def test_safety_slews_enabled_action():
    pose = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
    state = DryRunTransport(pose).read_state()
    safety = SafetySupervisor({"require_enable_switch": True, "max_action_delta": 0.25}, pose)
    safety.set_enabled(True)
    action, result = safety.filter_action(state, np.ones(12))
    assert result.allowed
    assert result.reason == "action_delta_limited"
    np.testing.assert_allclose(action, 0.25)
