import numpy as np

from go1_sim2real.action import Go1JointPositionMapper


def test_joint_position_mapping():
    mapper = Go1JointPositionMapper(np.zeros(12), action_scale=0.25, hip_scale_reduction=0.5)
    target = mapper.to_joint_target(np.ones(12))
    np.testing.assert_allclose(target[[0, 3, 6, 9]], 0.125)
    np.testing.assert_allclose(target[[1, 2, 4, 5, 7, 8, 10, 11]], 0.25)
