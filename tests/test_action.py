import numpy as np

from go1_sim2real.action import Go1JointPositionMapper


def test_joint_position_mapping():
    mapper = Go1JointPositionMapper(np.zeros(12), action_scale=0.25, hip_scale_reduction=0.5)
    target = mapper.to_joint_target(np.ones(12))
    np.testing.assert_allclose(target[:4], 0.125)
    np.testing.assert_allclose(target[4:], 0.25)
