import importlib.util
from pathlib import Path

import numpy as np
import pytest
from go1_sim2real.transport import DryRunTransport

path=Path(__file__).resolve().parents[1]/'tools/hardware/test_19499_stand.py'
spec=importlib.util.spec_from_file_location('stand_test_tool',path)
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def state():
    s=DryRunTransport(np.zeros(12)).read_state()
    s.battery_voltage=24.;s.motor_temperatures=np.full(12,40.)
    return s


def test_actual_low_battery_is_rejected():
    s=state();s.battery_voltage=20.608
    with pytest.raises(RuntimeError,match='Battery'):tool.validate_protected_state(s)


def test_no_battery_data_is_rejected():
    s=state();s.battery_voltage=None
    with pytest.raises(RuntimeError,match='Battery'):tool.validate_protected_state(s)


@pytest.mark.parametrize('fault,reason', [('emergency_stop','emergency'),('stale','Stale'),('tilt','Posture'),('hot','temperature'),('fast','velocity')])
def test_other_guards_still_apply(fault,reason):
    s=state()
    if fault=='emergency_stop':s.emergency_stop=True
    elif fault=='stale':s.timestamp-=1
    elif fault=='tilt':s.roll=.5
    elif fault=='hot':s.motor_temperatures[0]=65
    else:s.joint_vel[0]=7
    with pytest.raises(RuntimeError,match=reason):tool.validate_protected_state(s)


def test_good_state_passes():
    tool.validate_protected_state(state())


def test_calibrated_gain_cannot_exceed_known_limit():
    assert tool.calibrated_startup_gains([20]*12)==[20.]*12
    with pytest.raises(RuntimeError):tool.calibrated_startup_gains([30]*12)


def test_actual_latency_fault_still_rejected():
    with pytest.raises(RuntimeError,match='20ms'):tool.enforce_control_budget(22.818442)
    tool.enforce_control_budget(14.2)


def test_nonfinite_latency_rejected():
    with pytest.raises(RuntimeError):tool.enforce_control_budget(float('nan'))


def test_folded_knees_are_distinguished_from_extended_stance():
    standing=np.array([.1,-.1,.1,-.1]+[.75]*4+[-1.5]*4)
    folded=np.array([.3,-.3,.3,-.3]+[1.19]*4+[-2.77]*4)
    _,clear=tool.support_pose_geometry(standing,[0,0,-1])
    _,low=tool.support_pose_geometry(folded,[0,0,-1])
    assert min(clear)>.15
    assert max(low)<.07


def test_invalid_support_geometry_cannot_pass_clearance_guard():
    with pytest.raises(ValueError):tool.support_pose_geometry(np.zeros(12),[0,0,0])
    with pytest.raises(ValueError):tool.support_pose_geometry(np.full(12,np.nan),[0,0,-1])


def test_requested_effort_checks_position_and_damping_together():
    s=state();s.joint_pos=np.zeros(12);s.joint_vel=np.zeros(12)
    tool.enforce_pd_effort(np.full(12,.45),s,np.full(12,35.),np.ones(12),np.zeros(12))
    s.joint_vel=np.full(12,-6.)
    with pytest.raises(RuntimeError,match='20Nm'):
        tool.enforce_pd_effort(np.full(12,.45),s,np.full(12,35.),np.ones(12),np.zeros(12))
    with pytest.raises(RuntimeError):
        tool.enforce_pd_effort(np.full(12,np.nan),s,np.full(12,35.),np.ones(12),np.zeros(12))


def test_walking_profile_settles_ramps_and_returns_to_zero():
    assert tool.walking_profile(0,.05)==([0.,0.,0.],.2)
    assert tool.walking_profile(4,.05)==([.025,0.,0.],.35)
    assert tool.walking_profile(7,.05)==([.05,0.,0.],.5)
    assert tool.walking_profile(11,.05)==([.025,0.,0.],.35)
    assert tool.walking_profile(13,.05)==([0.,0.,0.],.2)
    assert tool.walking_profile(7,0)==([0.,0.,0.],.2)
    assert tool.walking_profile(7,.05,1.)==([.05,0.,0.],1.)
    assert tool.walking_profile(13,.05,1.)==([0.,0.,0.],.2)
    with pytest.raises(ValueError):tool.walking_profile(7,.05,2.)


@pytest.mark.parametrize('fault',['velocity_invalid','too_fast','no_contact','travel','lateral','heading'])
def test_walking_faults_stop_motion(fault):
    s=state();s.base_lin_vel=np.zeros(3);s.contact_count=4
    displacement=np.zeros(3);yaw=0.
    if fault=='velocity_invalid':s.base_lin_vel_valid=False
    elif fault=='too_fast':s.base_lin_vel[0]=.4
    elif fault=='no_contact':s.contact_count=0
    elif fault=='travel':displacement[0]=.31
    elif fault=='lateral':displacement[1]=.11
    else:yaw=.26
    with pytest.raises(RuntimeError):tool.validate_walking_state(s,displacement,yaw)
