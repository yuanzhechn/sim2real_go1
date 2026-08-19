import numpy as np

from go1_sim2real.sensors import UdpJsonAuxiliaryStateProvider


def test_flat_auxiliary_packet_only_requires_base_velocity():
    provider = UdpJsonAuxiliaryStateProvider(
        "127.0.0.1", 0, require_height_scan=False
    )
    try:
        state = provider._decode(b'{"base_lin_vel":[0.1,-0.2,0.0]}')
    finally:
        provider.close()
    np.testing.assert_allclose(state.base_lin_vel, [0.1, -0.2, 0.0])
    np.testing.assert_allclose(state.height_scan, 0.0)
