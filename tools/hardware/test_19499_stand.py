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
from go1_sim2real.policy import TorchScriptPolicy, NumpyElevationPolicy
from go1_sim2real.velocity_estimator import Go1ContactVelocityEstimator
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


def calibrated_startup_gains(values):
    gains=np.asarray(values,dtype=np.float32)
    if gains.shape!=(12,) or not np.all(np.isfinite(gains)) or np.any(gains<=0) or np.any(gains>20):
        raise RuntimeError('Calibrated startup gain must be 12 values in (0,20]')
    return gains.tolist()


def enforce_control_budget(elapsed_ms):
    if not np.isfinite(elapsed_ms) or elapsed_ms<0 or elapsed_ms>20:
        raise RuntimeError('20ms control budget exceeded: %.2fms'%elapsed_ms)


def enforce_pd_effort(target, state, kp, kd, torque_ff):
    """Bound the requested PD effort; SDK protection still applies afterward."""
    effort=np.asarray(kp)*(np.asarray(target)-state.joint_pos)-np.asarray(kd)*state.joint_vel+np.asarray(torque_ff)
    if effort.shape!=(12,) or not np.isfinite(effort).all() or np.max(np.abs(effort))>20.:
        raise RuntimeError('Protected requested PD effort exceeds 20Nm or is invalid')
    return effort


def support_pose_geometry(joint_pos, projected_gravity):
    """Flat-floor estimate only; cannot replace operator confirmation of knee clearance."""
    q=Go1ContactVelocityEstimator.policy_to_leg_matrix(joint_pos)
    gravity=np.asarray(projected_gravity,dtype=np.float32)
    if not np.isfinite(q).all() or gravity.shape!=(3,) or not np.isfinite(gravity).all() or not .95 <= np.linalg.norm(gravity) <= 1.05:
        raise ValueError('Support geometry requires finite joints and unit projected gravity')
    feet=[];knees=[]
    for leg in range(4):
        q0,q1,_=q[leg]
        rx=Go1ContactVelocityEstimator._rx(float(q0))
        hip=np.array([Go1ContactVelocityEstimator.HIP_X[leg],Go1ContactVelocityEstimator.HIP_Y[leg],0.])
        thigh=hip+rx@np.array([0.,Go1ContactVelocityEstimator.SIDE[leg]*.08,0.])
        knee=thigh+rx@Go1ContactVelocityEstimator._ry(float(q1))@np.array([0.,0.,-.213])
        foot,_=Go1ContactVelocityEstimator.foot_position_and_jacobian(leg,q[leg])
        feet.append(float(foot@gravity));knees.append(float(knee@gravity))
    height=float(np.median(feet)+.02)
    return height,[height-z for z in knees]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-hardware',action='store_true')
    parser.add_argument('--protected',action='store_true')
    parser.add_argument('--diagnostic-only',action='store_true',help='Passive LowState initialization only; no joint targets')
    parser.add_argument('--startup-duration',type=float,choices=(5.,10.),default=5.)
    parser.add_argument('--startup-calibrated-gain',action='store_true',help='Use existing robot calibrated Kp for the startup transition')
    parser.add_argument('--ground-support-startup',action='store_true',help='Use existing ground-start profile: front Kp20, rear Kp30')
    parser.add_argument('--ground-support-hold',action='store_true',help='Maintain rear Kp30 through actor takeover, with joint error guard')
    parser.add_argument('--rear-support-kp',type=float,choices=(30.,35.,40.),default=30.,help='Protected rear gain; requested PD effort is bounded to 20Nm')
    parser.add_argument('--backend',choices=('torchscript','numpy'),default='torchscript')
    parser.add_argument('--hold-seconds',type=int,choices=(3,30),default=3)
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
    policy=None if args.diagnostic_only else (NumpyElevationPolicy(bundle.policy_path) if args.backend=='numpy' else TorchScriptPolicy(bundle.policy_path,cpu_threads=config.policy['cpu_threads'],freeze_for_inference=True))
    builder=Go1ObservationBuilder(default_joint_pos=config.robot['default_joint_pos'],terms=config.observation['terms'],height_map_mode='unknown')
    if policy is not None:
        for _ in range(50): policy(np.zeros(1072,dtype=np.float32))
    raw=copy.deepcopy(config.raw)
    if args.startup_calibrated_gain:
        raw['transport']['startup_kp']=calibrated_startup_gains(original.transport['kp'])
        raw['transport']['startup_kp_limit']=20.
    if args.ground_support_startup:
        raw['transport']['startup_kp']=[args.rear_support_kp if n.startswith(('RL_','RR_')) else 20. for n in config.robot['joint_names']]
        raw['transport']['startup_kp_limit']=args.rear_support_kp
    if args.ground_support_hold:
        if not args.ground_support_startup:parser.error('Support hold requires the same support startup profile')
        raw['transport']['kp']=[args.rear_support_kp if n.startswith(('RL_','RR_')) else 20. for n in config.robot['joint_names']]
    raw['safety'].update(max_roll_rad=.45,max_pitch_rad=.45,max_joint_velocity_rad_s=6.,
                         max_action_delta=.05,runtime_action_limit=.2,require_base_lin_vel_valid=False)
    if args.ground_support_hold:raw['safety']['max_joint_error_rad']=.45
    safety=SafetySupervisor(raw['safety'],raw['robot']['default_joint_pos']);safety.set_enabled(True)
    report=dict(model=19499,hardware=args.enable_hardware,protected=args.protected,command=[0,0,0],
                action_limit=.2,model_validated=False,frames=[],result='not_started',
                calibrated_startup_gain=args.startup_calibrated_gain,startup_duration=args.startup_duration,
                backend=args.backend,hold_seconds=args.hold_seconds,deadline_misses=0,ground_support_startup=args.ground_support_startup,ground_support_hold=args.ground_support_hold,rear_support_kp=args.rear_support_kp,requested_pd_effort_limit_nm=20.)
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
            report['startup_telemetry']=[]
            read_count=0
            latest_state=None
            def guarded_read():
                nonlocal read_count,latest_state
                state=real_read()
                if time.monotonic()>deadline: raise RuntimeError('Protected test wall-time limit')
                validate_protected_state(state)
                if read_count % 50 == 0:
                    motors=list(transport._state.motorState)[:12]
                    commands=list(transport._cmd.motorCmd)[:12]
                    row=dict(read=read_count,tick=transport._last_tick,
                        command_level=int(transport._cmd.levelFlag),state_level=int(transport._state.levelFlag),
                        command_modes=[int(m.mode) for m in commands],state_modes=[int(m.mode) for m in motors],
                        command_q=[float(m.q) for m in commands],command_kp=[float(m.Kp) for m in commands],
                        measured_q=[float(m.q) for m in motors],measured_tau=[float(m.tauEst) for m in motors])
                    report['startup_telemetry'].append(row)
                    print('STARTUP_PACKET '+json.dumps(row),flush=True)
                read_count+=1
                latest_state=state
                return state
            transport.read_state=guarded_read
            real_send_target=transport._send_joint_target
            def guarded_send_target(target,kp=None,kd=None,torque_ff=None):
                if latest_state is None:raise RuntimeError('No fresh state for requested PD effort check')
                validate_protected_state(latest_state)
                enforce_pd_effort(target,latest_state,
                    transport._kp if kp is None else kp,
                    transport._kd if kd is None else kd,
                    transport._torque_ff if torque_ff is None else torque_ff)
                return real_send_target(target,kp=kp,kd=kd,torque_ff=torque_ff)
            transport._send_joint_target=guarded_send_target
            state=transport.read_state()
            report['initial_q']=state.joint_pos.tolist()
            report['initial_roll_pitch']=[state.roll,state.pitch]
            report['battery_voltage']=state.battery_voltage
            # Use verified interface gains; keep target, tracking and all other guards.
            transport.prepare_for_policy(duration=args.startup_duration)
            report['startup_completed']=True
            deadline=time.monotonic()+args.hold_seconds+5.
        else:
            transport=DryRunTransport(np.array(raw['robot']['default_joint_pos']))
        started=time.monotonic();next_tick=started
        last_safe=np.zeros(12,dtype=np.float32);last_good=started;misses=0
        for step in range(args.hold_seconds*50 if args.enable_hardware else 3):
            if time.monotonic()-started>args.hold_seconds:break
            cycle_start=time.monotonic()
            state=transport.read_state()
            observation=builder.build(state,[0,0,0])
            action=np.asarray(policy(observation),dtype=np.float32)
            action=np.clip(action,-.2,.2)
            filtered,result=safety.filter_action(state,action)
            compute_ms=(time.monotonic()-cycle_start)*1000
            record=dict(step=step,compute_ms=compute_ms,safety=result.reason,roll=state.roll,pitch=state.pitch,
                        action=filtered.tolist(),joint_pos=state.joint_pos.tolist(),contact_count=state.contact_count,action_sent=False,elapsed_s=time.monotonic()-started)
            report['frames'].append(record)
            if not result.allowed:raise RuntimeError('Safety fault: '+result.reason)
            # No command from an over-budget policy cycle reaches the motors.
            if args.enable_hardware and args.hold_seconds==3:
                enforce_control_budget(compute_ms)
            elif args.enable_hardware and compute_ms>20:
                misses+=1;report['deadline_misses']+=1
                record['late_action_discarded']=True
                if misses>=3 or time.monotonic()-last_good>.1:
                    raise RuntimeError('Repeated missed deadlines or safe target older than 100ms')
                safety._last_action=last_safe.copy()
                filtered=last_safe.copy()
                record['action']=filtered.tolist()
            else:
                misses=0;last_good=time.monotonic()
            transport.send_action(filtered)
            last_safe=filtered.copy()
            record["action_sent"]=True
            if step%50==0:
                height,knee_heights=support_pose_geometry(state.joint_pos,state.projected_gravity)
                record["flat_ground_knee_height_estimates_m"]=knee_heights
                record['flat_ground_height_estimate_m']=height
                report['latest']=record
                if args.enable_hardware and args.hold_seconds>3 and height<.22:
                    raise RuntimeError('Estimated support pose too low to claim standing')
                if args.enable_hardware and args.ground_support_hold and step>=100 and min(knee_heights)<.07:
                    raise RuntimeError('Rear/other knee estimated clearance below 7cm; standing not achieved')
                print('STAND_STATUS '+json.dumps(record),flush=True)
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
