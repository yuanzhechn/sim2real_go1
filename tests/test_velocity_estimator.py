import numpy as np

from go1_sim2real.velocity_estimator import Go1ContactVelocityEstimator


def test_analytic_jacobian_matches_finite_difference():
    q = np.asarray([0.12, 0.8, -1.5], dtype=np.float32)
    _foot, jacobian = Go1ContactVelocityEstimator.foot_position_and_jacobian(0, q)
    epsilon = 1e-4
    numeric = np.empty((3, 3), dtype=np.float32)
    for joint in range(3):
        plus = q.copy()
        minus = q.copy()
        plus[joint] += epsilon
        minus[joint] -= epsilon
        p_plus, _ = Go1ContactVelocityEstimator.foot_position_and_jacobian(0, plus)
        p_minus, _ = Go1ContactVelocityEstimator.foot_position_and_jacobian(0, minus)
        numeric[:, joint] = (p_plus - p_minus) / (2 * epsilon)
    np.testing.assert_allclose(jacobian, numeric, atol=3e-4)


def test_stationary_contact_recovers_known_body_velocity():
    estimator = Go1ContactVelocityEstimator(filter_alpha=1.0, contact_threshold=5.0)
    q = np.asarray(
        [[0.1, 0.8, -1.5], [-0.1, 0.8, -1.5], [0.1, 1.0, -1.5], [-0.1, 1.0, -1.5]],
        dtype=np.float32,
    )
    omega = np.asarray([0.1, -0.05, 0.2], dtype=np.float32)
    expected = np.asarray([0.4, -0.1, 0.05], dtype=np.float32)
    dq = np.empty_like(q)
    for leg in range(4):
        foot, jacobian = estimator.foot_position_and_jacobian(leg, q[leg])
        relative_velocity = -expected - np.cross(omega, foot)
        dq[leg] = np.linalg.pinv(jacobian) @ relative_velocity
    q_group_major = np.concatenate((q[:, 0], q[:, 1], q[:, 2]))
    dq_group_major = np.concatenate((dq[:, 0], dq[:, 1], dq[:, 2]))
    result = estimator.update(
        q_group_major, dq_group_major, omega, [20, 20, 20, 20]
    )
    np.testing.assert_allclose(result, expected, atol=1e-4)
    assert estimator.valid
    assert estimator.contact_count == 4


def test_suspended_zero_estimated_force_is_invalid():
    estimator = Go1ContactVelocityEstimator(contact_threshold=5.0)
    result = estimator.update(np.zeros(12), np.zeros(12), np.zeros(3), np.zeros(4))
    np.testing.assert_allclose(result, 0.0)
    assert not estimator.valid
    assert estimator.contact_count == 0


def test_zero_force_fallback_uses_robust_four_leg_velocity():
    estimator = Go1ContactVelocityEstimator(
        filter_alpha=1.0, contact_threshold=5.0, allow_force_fallback=True
    )
    q = np.asarray(
        [[0.1, 0.8, -1.5], [-0.1, 0.8, -1.5],
         [0.1, 1.0, -1.5], [-0.1, 1.0, -1.5]], dtype=np.float32
    )
    omega = np.asarray([0.1, -0.05, 0.2], dtype=np.float32)
    expected = np.asarray([0.4, -0.1, 0.05], dtype=np.float32)
    dq = np.empty_like(q)
    for leg in range(4):
        foot, jacobian = estimator.foot_position_and_jacobian(leg, q[leg])
        dq[leg] = np.linalg.pinv(jacobian) @ (
            -expected - np.cross(omega, foot)
        )
    result = estimator.update(
        np.concatenate((q[:, 0], q[:, 1], q[:, 2])),
        np.concatenate((dq[:, 0], dq[:, 1], dq[:, 2])),
        omega,
        np.zeros(4),
    )

    np.testing.assert_allclose(result, expected, atol=1e-4)
    assert estimator.valid
    assert estimator.contact_count == 0


def test_default_pose_base_height_matches_go1_geometry():
    q = np.asarray(
        [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0,
         -1.5, -1.5, -1.5, -1.5], dtype=np.float32
    )
    height = Go1ContactVelocityEstimator.estimate_base_height(q, foot_radius=0.02)
    assert 0.30 < height < 0.33
