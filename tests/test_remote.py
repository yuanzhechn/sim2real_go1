import numpy as np

from go1_sim2real.remote import RemoteVelocityCommand
from go1_sim2real.transport import DryRunTransport


def test_remote_velocity_mapping_and_deadband():
    state = DryRunTransport(np.zeros(12)).read_state()
    state.remote_axes = np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    mapper = RemoteVelocityCommand(
        {"deadband": 0.1, "scales": [0.5, 0.3, 0.8], "signs": [1, -1, -1]}
    )
    np.testing.assert_allclose(mapper(state), [0.5, -0.3, -0.8])
    state.remote_axes = np.asarray([0.05, -0.05, 0.05, 1.0], dtype=np.float32)
    np.testing.assert_allclose(mapper(state), 0.0)
