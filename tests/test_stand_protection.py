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
