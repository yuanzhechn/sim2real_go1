"""Closed-loop startup against a lagging plant; no SDK or network."""
from types import SimpleNamespace
import numpy as np
import pytest
from go1_sim2real.transport import UnitreeSdkTransport


def plant(monkeypatch, jam=False, estop=False, disturbance=False):
    clock=[0.]
    monkeypatch.setattr('go1_sim2real.transport.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('go1_sim2real.transport.time.sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    transport=UnitreeSdkTransport.__new__(UnitreeSdkTransport)
    transport._allow_commands=True;transport._passive_only=False
    transport.config=dict(startup_transition_s=16.,startup_control_dt=.02,startup_kp=10.,startup_kd=1.,
        startup_fixed_cadence=True,startup_tracking_pause_rad=.35,startup_tracking_resume_rad=.30,
        startup_max_duration_s=28.,startup_max_tracking_error=.45)
    transport.safety_config={};transport._default_joint_pos=np.full(12,1.2)
    transport.policy_joint_names=[str(i) for i in range(12)]
    transport._last_action=np.ones(12)
    actual=np.zeros(12);last=np.zeros(12);sent=[]
    def read():
        frozen=jam or 4.<=clock[0]<8.
        if not frozen:actual[:]+=.08*(last-actual)
        if disturbance and clock[0]>5.:actual[:]=-1.
        return SimpleNamespace(joint_pos=actual.copy(),joint_vel=np.zeros(12),
            emergency_stop=estop and clock[0]>5.,remote_axes=np.zeros(4),battery_voltage=24.,
            motor_temperatures=np.full(12,35.),contact_count=4,base_lin_vel=np.zeros(3))
    def send(target,**_):
        sent.append((clock[0],target.copy(),actual.copy()));last[:]=target
    transport.read_state=read;transport._send_joint_target=send
    return transport,sent,clock


def test_transient_stall_holds_then_resumes_without_target_jump(monkeypatch):
    transport,sent,clock=plant(monkeypatch)
    transport.prepare_for_policy()
    targets=np.array([row[1] for row in sent])
    errors=np.array([np.max(np.abs(row[1]-row[2])) for row in sent])
    assert 16.<clock[0]<28.
    assert np.count_nonzero(np.all(np.diff(targets,axis=0)==0,axis=1))>10
    assert np.max(errors)<=.35
    assert np.max(np.diff(targets,axis=0))<.003
    np.testing.assert_allclose(targets[-1],1.2)


def test_permanent_stall_times_out_without_completing_startup(monkeypatch):
    transport,sent,clock=plant(monkeypatch,jam=True)
    with pytest.raises(RuntimeError,match='超时'):transport.prepare_for_policy()
    assert clock[0]<=28.01
    assert max(np.max(row[1]) for row in sent)<=.35
    np.testing.assert_allclose(transport._last_action,1.)


def test_estop_remains_active_while_progress_is_paused(monkeypatch):
    transport,_,_=plant(monkeypatch,estop=True)
    with pytest.raises(RuntimeError,match='急停'):transport.prepare_for_policy()


def test_unexpected_motion_still_hits_hard_tracking_guard(monkeypatch):
    transport,sent,_=plant(monkeypatch,disturbance=True)
    with pytest.raises(RuntimeError,match='未跟随'):transport.prepare_for_policy()
    assert all(np.max(np.abs(target-actual))<.45 for _,target,actual in sent)
