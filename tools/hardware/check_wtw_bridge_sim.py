#!/usr/bin/env python3
"""Bounded WTW inference/LCM test. Always starts --simulate, never a robot UDP connection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from go1_sim2real.wtw import WtwInterface,SDK_ORDER


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=ROOT.parent/'walk-these-ways')
    parser.add_argument('--build',type=Path,default=ROOT/'build/wtw_bridge')
    parser.add_argument('--policy-dir',type=Path,default=ROOT/'logs/reference_audit/wtw_0e7236b')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.upstream))
    sys.path.insert(0,str(args.build/'generated'))
    import lcm
    from go1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
    from go1_bridge_status_lcmt import go1_bridge_status_lcmt
    config=json.loads((args.policy_dir/'interface.json').read_text())
    adapter=WtwInterface(config)
    torch.set_num_threads(1)
    policy=torch.jit.load(str(args.policy_dir/'wtw_student.ts'),map_location='cpu').eval()
    with torch.inference_mode():
        for _ in range(30):policy(torch.zeros(1,2100))
    prefix='GO1_WTW_TEST_'
    url='udpm://239.255.76.67:7669?ttl=0'
    channel=lcm.LCM(url);latest={}
    def receive(_channel,data):latest['status']=go1_bridge_status_lcmt.decode(data)
    channel.subscribe(prefix+'status',receive).set_queue_capacity(1)
    def pump(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:channel.handle_timeout(1)
    def coherent():
        return 'status' in latest and abs(time.time()*1e6-latest['status'].timestamp_us)<100000
    report={'simulated':True,'hardware_validated':False,'frames':[]}
    child=subprocess.Popen([str(args.build/'go1_wtw_bridge'),'--simulate','--seconds','8','--lcm-url',url],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+3
        while not coherent() and time.monotonic()<deadline:pump(.01)
        if not coherent():raise RuntimeError('No coherent simulated bridge state')
        if not latest['status'].simulated:raise RuntimeError('Refusing a non-simulated bridge')
        history=np.zeros(2100,dtype=np.float32);previous=np.zeros(12);next_tick=time.monotonic()
        with torch.inference_mode():
            for index in range(100):
                if not coherent():pump(.012)
                status=latest['status']
                if status.fault not in (go1_bridge_status_lcmt.FAULT_NONE,go1_bridge_status_lcmt.FAULT_NO_COMMAND):raise RuntimeError(f'Simulator safety fault {status.fault}')
                start=time.monotonic();action=policy(torch.from_numpy(history[None,:])).numpy()[0]
                action=np.clip(action,-.5,.5)
                action=np.clip(action,previous-.05,previous+.05)
                packet=pd_tau_targets_lcmt();packet.id=index+1;packet.timestamp_us=time.time_ns()//1000
                packet.q_des=adapter.sdk_target(action).tolist();packet.kp=[20.]*12;packet.kd=[.5]*12
                channel.publish(prefix+'pd_plustau_targets',packet.encode())
                previous=action.copy();adapter.set_action(action)
                next_tick+=.02;pump(max(0,next_tick-time.monotonic()))
                if not coherent():pump(.012)
                if not coherent():raise RuntimeError('State messages lost coherence')
                body=legs=latest['status']
                roll,pitch=body.rpy[:2]
                gravity=[np.sin(pitch),-np.sin(roll)*np.cos(pitch),-np.cos(roll)*np.cos(pitch)]
                _,history=adapter.observe(gravity,np.asarray(legs.q)[SDK_ORDER],np.asarray(legs.dq)[SDK_ORDER],adapter.default_commands())
                report['frames'].append(dict(index=index,cycle_ms=(time.monotonic()-start)*1000,
                    bridge_cycles=latest['status'].cycles,bridge_fault=latest['status'].fault))
        pump(.18)
        report['watchdog_fault']=latest['status'].fault
        if report['watchdog_fault']!=go1_bridge_status_lcmt.FAULT_STALE_COMMAND:raise RuntimeError('100ms command watchdog did not latch')
        # A later valid command must not restart control after a latched watchdog.
        packet.id=101;packet.timestamp_us=time.time_ns()//1000
        channel.publish(prefix+'pd_plustau_targets',packet.encode());pump(.04)
        report['watchdog_still_latched']=latest['status'].fault==go1_bridge_status_lcmt.FAULT_STALE_COMMAND
        if not report['watchdog_still_latched']:raise RuntimeError('Fault unexpectedly reset')
        report['result']='simulated_pipeline_and_watchdog_passed'
    except BaseException as error:
        report['result']='failed';report['reason']=str(error);raise
    finally:
        child.terminate()
        stdout,stderr=child.communicate(timeout=3)
        report['bridge_stdout']=stdout;report['bridge_stderr']=stderr
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))


if __name__=='__main__':main()
