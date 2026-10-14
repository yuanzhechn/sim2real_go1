#!/usr/bin/env python3
"""Bounded, protected 19499 commissioning. Never marks the model validated."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from go1_sim2real.bundle import load_bundle, validate_bundle_runtime_config
from go1_sim2real.config import load_config, validate_hardware_config
from go1_sim2real.observation import Go1ObservationBuilder
from go1_sim2real.policy import TorchScriptPolicy
from go1_sim2real.safety import SafetySupervisor
from go1_sim2real.transport import UnitreeSdkTransport, DryRunTransport


def validate_protected_state(state, now=None):
    now=time.monotonic() if now is None else now
    if state.emergency_stop: raise RuntimeError('B emergency stop')
    if abs(state.roll)>.45 or abs(state.pitch)>.45: raise RuntimeError('Posture limit')
    if now-state.timestamp>.15: raise RuntimeError('Stale robot state')
    if state.battery_voltage is None or state.battery_voltage<21:
        raise RuntimeError('Battery must be >=21V; observed: '+str(state.battery_voltage))
    if state.motor_temperatures is None or np.max(state.motor_temperatures)>60:
        raise RuntimeError('Motor temperature limit')
    if np.max(np.abs(state.joint_vel))>6:
        raise RuntimeError('Joint velocity limit')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-hardware',action='store_true')
    parser.add_argument('--protected',action='store_true')
    parser.add_argument('--diagnostic-only',action='store_true',help='Passive LowState initialization only; no joint targets')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.diagnostic_only and not args.enable_hardware:
        parser.error('Passive diagnostics require --enable-hardware and --protected')
    if args.enable_hardware and not args.protected:
        parser.error('Hardware test requires confirmed protective sling and remote emergency stop')
    config=load_config(ROOT/'config/robots/unitree5151_19499.yaml')
    original=load_config(ROOT/'config/robots/unitree5151.yaml')
    validate_hardware_config(original)
    bundle=load_bundle(ROOT/'artifacts/yuanzhe/collective_harder_19499_1072d_20261014')
    validate_bundle_runtime_config(bundle,config)
    if bundle.manifest['checkpoint_sha256']!='119461a7d5a665f42fddefe48732bcfa932461a22bc256c1308009ddca9907d4':
        raise RuntimeError('Wrong checkpoint')
    if hashlib.sha256((bundle.root/'checkpoint/model_19499.pt').read_bytes()).hexdigest()!=bundle.manifest['checkpoint_sha256']:
        raise RuntimeError('Checkpoint integrity failed')
    # Use the separately verified robot interface; do not copy validation flags to this model.
    if config.transport.get('base_hardware_calibration_id')!=original.transport['calibration_id']:
        raise RuntimeError('Robot calibration provenance mismatch')
    for key in ('sdk_python_path','target_ip','target_port','sdk_joint_names','joint_directions',
                'joint_offsets','joint_lower_limits','joint_upper_limits','kp','kd','torque_ff'):
        if config.transport[key]!=original.transport[key]:
            raise RuntimeError('Hardware calibration changed: '+key)
    if config.observation['height_map_mode']!='unknown':
        raise RuntimeError('This protected test expects the explicit unknown map contract')
    policy=None if args.diagnostic_only else TorchScriptPolicy(bundle.policy_path,cpu_threads=config.policy['cpu_threads'],freeze_for_inference=True)
    builder=Go1ObservationBuilder(default_joint_pos=config.robot['default_joint_pos'],terms=config.observation['terms'],height_map_mode='unknown')
    if policy is not None:
        for _ in range(50): policy(np.zeros(1072,dtype=np.float32))
    raw=copy.deepcopy(config.raw)
    raw['safety'].update(max_roll_rad=.45,max_pitch_rad=.45,max_joint_velocity_rad_s=6.,
                         max_action_delta=.05,runtime_action_limit=.2,require_base_lin_vel_valid=False)
    safety=SafetySupervisor(raw['safety'],raw['robot']['default_joint_pos']);safety.set_enabled(True)
    report=dict(model=19499,hardware=args.enable_hardware,protected=args.protected,command=[0,0,0],
                action_limit=.2,model_validated=False,frames=[],result='not_started')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    transport=None;restore=False
    control=ROOT/'tools/hardware/sport_mode_control.py'
    try:
        if args.enable_hardware:
            subprocess.run(['sudo','-n','/usr/bin/python3',str(control),'status'],check=True)
            restore=True
            subprocess.run(['sudo','-n','/usr/bin/python3',str(control),'stop'],check=True)
            transport=UnitreeSdkTransport(raw,require_auxiliary=True)
            real_read=transport.read_state
            # Warm state decoding while sending no joint-position targets.
            report['passive_state_warmup']=[]
            for _ in range(3):
                start=time.monotonic();sample=real_read()
                report['passive_state_warmup'].append(dict(read_ms=(time.monotonic()-start)*1000,
                    age_ms=(time.monotonic()-sample.timestamp)*1000,tick=transport._last_tick,
                    battery_voltage=sample.battery_voltage,
                    cell_vol_mV=[int(v) for v in transport._state.bms.cell_vol],
                    soc=int(transport._state.bms.SOC),roll=sample.roll,pitch=sample.pitch,
                    max_motor_temperature=float(np.max(sample.motor_temperatures))))
                if sample.emergency_stop:raise RuntimeError('B emergency stop during passive warmup')
                time.sleep(.02)
            if args.diagnostic_only:
                report['result']='passive_diagnostics_completed'
                return
            deadline=time.monotonic()+30.
            def guarded_read():
                state=real_read()
                if time.monotonic()>deadline: raise RuntimeError('Protected test wall-time limit')
                validate_protected_state(state)
                return state
            transport.read_state=guarded_read
            state=transport.read_state()
            report['initial_q']=state.joint_pos.tolist()
            report['initial_roll_pitch']=[state.roll,state.pitch]
            report['battery_voltage']=state.battery_voltage
            # Existing verified low-gain transition; no gain increase on an untested model.
            transport.prepare_for_policy(duration=5.)
            report['startup_completed']=True
        else:
            transport=DryRunTransport(np.array(raw['robot']['default_joint_pos']))
        started=time.monotonic();next_tick=started
        for step in range(150 if args.enable_hardware else 3):
            if time.monotonic()-started>3.:break
            cycle_start=time.monotonic()
            state=transport.read_state()
            observation=builder.build(state,[0,0,0])
            action=np.asarray(policy(observation),dtype=np.float32)
            action=np.clip(action,-.2,.2)
            filtered,result=safety.filter_action(state,action)
            compute_ms=(time.monotonic()-cycle_start)*1000
            record=dict(step=step,compute_ms=compute_ms,safety=result.reason,roll=state.roll,pitch=state.pitch,
                        action=filtered.tolist(),joint_pos=state.joint_pos.tolist(),contact_count=state.contact_count)
            report['frames'].append(record)
            if not result.allowed:raise RuntimeError('Safety fault: '+result.reason)
            # No command from an over-budget policy cycle reaches the motors.
            if args.enable_hardware and compute_ms>20:raise RuntimeError('20ms control budget exceeded: %.2fms'%compute_ms)
            transport.send_action(filtered)
            print(json.dumps(record),flush=True)
            next_tick+=.02;time.sleep(max(0,next_tick-time.monotonic()))
        report['result']='short_test_completed'
    except BaseException as exc:
        report['result']='aborted';report['reason']=str(exc)
        print('TEST_ABORTED:',str(exc),flush=True)
        raise
    finally:
        try:
            if transport is not None:transport.close()
        finally:
            try:
                if restore:
                    subprocess.run(['sudo','-n','/usr/bin/python3',str(control),'start'],check=True)
            finally:
                args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
