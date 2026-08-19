from types import SimpleNamespace
import struct
import time

import numpy as np

from go1_sim2real.sensors import AuxiliaryState
from go1_sim2real.transport import UnitreeSdkTransport


class FakeMotor:
    def __init__(self, q=0.0, dq=0.0):
        self.mode = 0x0A
        self.q = q
        self.dq = dq
        self.temperature = 35
        self.Kp = 0.0
        self.Kd = 0.0
        self.tau = 0.0


class FakeLowState:
    def __init__(self):
        self.tick = 1
        self.motorState = [FakeMotor(float(index), index / 10) for index in range(20)]
        self.imu = SimpleNamespace(
            quaternion=[1.0, 0.0, 0.0, 0.0],
            gyroscope=[0.1, 0.2, 0.3],
            rpy=[0.0, 0.0, 0.0],
        )
        self.bms = SimpleNamespace(cell_vol=[2500] * 10)
        self.wirelessRemote = [0, 0, 1 << 5, 0] + [0] * 36


class FakeLowCmd:
    def __init__(self):
        self.motorCmd = [FakeMotor() for _ in range(20)]


class FakeUdp:
    def __init__(self, *args):
        self.send_count = 0

    def InitCmdData(self, cmd):
        self.cmd = cmd

    def Recv(self):
        return 0

    def GetRecv(self, state):
        return None

    def SetSend(self, cmd):
        self.cmd = cmd

    def Send(self):
        self.send_count += 1
        return 0


class FakeSdk:
    LOWLEVEL = 0xFF
    UDP = FakeUdp
    LowCmd = FakeLowCmd
    LowState = FakeLowState


class BootstrapUdp(FakeUdp):
    def Recv(self):
        return 0 if self.send_count else -1


class BootstrapSdk(FakeSdk):
    UDP = BootstrapUdp


class FakeAuxiliary:
    def read(self):
        return AuxiliaryState(np.array([0.4, 0.0, 0.0]), np.arange(187) / 100, time.monotonic())

    def close(self):
        pass


def make_config():
    policy_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    sdk_names = policy_names[3:6] + policy_names[0:3] + policy_names[9:12] + policy_names[6:9]
    return {
        "observation": {"require_height_scan": True},
        "robot": {
            "joint_names": policy_names,
            "default_joint_pos": [0.0] * 12,
            "action_scale": 0.25,
            "control_dt": 0.02,
        },
        "safety": {"stand_on_fault": True},
        "transport": {
            "sdk_joint_names": sdk_names,
            "joint_directions": [1.0] * 12,
            "joint_offsets": [0.0] * 12,
            "kp": list(range(1, 13)),
            "kd": [0.5] * 12,
            "torque_ff": [0.0] * 12,
            "joint_lower_limits": [-20.0] * 12,
            "joint_upper_limits": [20.0] * 12,
            "enable_button": "L2",
            "emergency_stop_button": "B",
        },
    }


def test_unitree_state_reorders_joints_and_projects_gravity():
    transport = UnitreeSdkTransport(make_config(), FakeSdk, FakeAuxiliary())
    state = transport.read_state()
    np.testing.assert_allclose(state.joint_pos, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
    np.testing.assert_allclose(state.projected_gravity, [0, 0, -1])
    np.testing.assert_allclose(state.base_lin_vel, [0.4, 0, 0])
    # 启动时即使 L2 已经按下也不能自动使能，必须先松开再重新按下。
    assert not state.enable_switch
    assert not state.emergency_stop
    transport.close()


def test_unitree_action_reorders_targets_and_gains():
    transport = UnitreeSdkTransport(make_config(), FakeSdk, FakeAuxiliary())
    transport.read_state()
    transport.send_action(np.ones(12, dtype=np.float32))
    np.testing.assert_allclose([m.q for m in transport._cmd.motorCmd[:12]], 0.25)
    assert [m.Kp for m in transport._cmd.motorCmd[:12]] == [4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9]
    transport.close()


def test_read_only_transport_never_sends_even_on_close():
    transport = UnitreeSdkTransport(
        make_config(), FakeSdk, FakeAuxiliary(), allow_commands=False
    )
    transport.read_state()
    transport.close()
    assert transport._udp.send_count == 0


def test_command_transport_bootstraps_low_state_with_passive_packet():
    transport = UnitreeSdkTransport(make_config(), BootstrapSdk, FakeAuxiliary())
    transport.read_state()
    assert transport._udp.send_count == 1
    assert all(m.mode == 0x00 for m in transport._cmd.motorCmd[:12])
    assert all(m.Kp == 0.0 and m.Kd == 0.0 and m.tau == 0.0 for m in transport._cmd.motorCmd[:12])
    transport.close()


def test_remote_l2_toggles_enable_and_b_latches_emergency_stop():
    transport = UnitreeSdkTransport(make_config(), FakeSdk, FakeAuxiliary())
    assert not transport.read_state().enable_switch

    transport._state.wirelessRemote = [0, 0, 0, 0] + [0] * 36
    assert not transport.read_state().enable_switch
    transport._state.wirelessRemote = [0, 0, 1 << 5, 0] + [0] * 36
    assert transport.read_state().enable_switch
    assert transport.read_state().enable_switch

    transport._state.wirelessRemote = [0, 0, 0, 0] + [0] * 36
    transport.read_state()
    transport._state.wirelessRemote = [0, 0, 1 << 5, 0] + [0] * 36
    assert not transport.read_state().enable_switch

    transport._state.wirelessRemote = [0, 0, 0, 1 << 1] + [0] * 36
    stopped = transport.read_state()
    assert stopped.emergency_stop
    assert not stopped.enable_switch
    transport._state.wirelessRemote = [0, 0, 0, 0] + [0] * 36
    assert transport.read_state().emergency_stop
    transport.close()


def test_program_mode_waits_for_centered_remote_then_auto_enables():
    config = make_config()
    config["transport"].update(
        {
            "enable_switch_mode": "program",
            "remote_startup_center_frames": 2,
            "remote_control": {"deadband": 0.08},
        }
    )
    transport = UnitreeSdkTransport(config, FakeSdk, FakeAuxiliary())
    raw = bytearray(40)
    struct.pack_into("<fffff", raw, 4, 0.0, 0.0, 0.0, 0.0, 0.4)
    transport._state.wirelessRemote = list(raw)
    assert not transport.read_state().enable_switch
    raw = bytearray(40)
    transport._state.wirelessRemote = list(raw)
    assert not transport.read_state().enable_switch
    enabled = transport.read_state()
    assert enabled.enable_switch
    np.testing.assert_allclose(enabled.remote_axes, 0.0)
    transport.close()
