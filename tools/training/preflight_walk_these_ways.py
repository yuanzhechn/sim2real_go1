#!/usr/bin/env python3
"""Offline export of the upstream WTW student; never imports a robot transport.

Architecture: Improbable-AI/walk-these-ways, go1_gym_learn/ppo_cse/actor_critic.py.
Keep this baseline separate from model_19499 and its immutable bundle.
"""
import argparse
import collections
import hashlib
import io
import importlib
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ConfigReader(pickle.Unpickler):
    """Only decode upstream configuration arrays/tensors, mapping tensors to CPU."""
    def find_class(self, module, name):
        if (module,name)==('torch.storage','_load_from_bytes'):
            return lambda data:torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        multiarray=importlib.import_module('numpy._core.multiarray' if hasattr(np,'_core') else 'numpy.core.multiarray')
        allowed={
            ('numpy','dtype'):np.dtype,('numpy','ndarray'):np.ndarray,
            ('numpy.core.multiarray','scalar'):multiarray.scalar,
            ('numpy.core.multiarray','_reconstruct'):multiarray._reconstruct,
            ('torch._utils','_rebuild_tensor_v2'):torch._utils._rebuild_tensor_v2,
            ('collections','OrderedDict'):collections.OrderedDict,
        }
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported configuration global: {module}.{name}')
        return allowed[module,name]


def mlp(weights,prefix):
    indices=sorted(int(k.split('.')[1]) for k in weights if k.startswith(prefix+'.') and k.endswith('.weight'))
    if indices!=list(range(0,2*len(indices),2)) or not indices:
        raise ValueError('Unexpected upstream MLP architecture: '+prefix)
    layers=[]
    for n,index in enumerate(indices):
        weight=weights[f'{prefix}.{index}.weight'];bias=weights[f'{prefix}.{index}.bias']
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError('Nonfinite upstream parameters')
        layer=nn.Linear(weight.shape[1],weight.shape[0])
        layer.load_state_dict({'weight':weight,'bias':bias},strict=True)
        layers.append(layer)
        if n<len(indices)-1:layers.append(nn.ELU())
    return nn.Sequential(*layers)


class Student(nn.Module):
    def __init__(self, adaptation, body):
        super().__init__();self.adaptation_module=adaptation;self.actor_body=body

    def forward(self, history):
        latent=self.adaptation_module(history)
        return self.actor_body(torch.cat((history,latent),dim=-1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[3]/'walk-these-ways')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.source_root.resolve()
    run=root/'runs/gait-conditioned-agility/pretrain-v0/train/025417.456545'
    checkpoint=run/'checkpoints/ac_weights_last.pt'
    with (run/'parameters.pkl').open('rb') as file:config=ConfigReader(file).load()['Cfg']
    env=config['env'];control=config['control']
    if config.get('terrain',{}).get('measure_heights',False):
        raise ValueError('This baseline has terrain observations; camera-free contract not established')
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    torch.set_num_threads(1)
    model=Student(mlp(weights,'adaptation_module'),mlp(weights,'actor_body')).eval()
    history_dim=env['num_observations']*env['num_observation_history']
    if history_dim!=model.adaptation_module[0].in_features:
        raise ValueError('Configuration/checkpoint history mismatch')
    scripted=torch.jit.script(model)
    generator=torch.Generator().manual_seed(19499)
    samples=torch.randn(32,history_dim,generator=generator)*.1
    with torch.inference_mode():
        reference=model(samples);actual=scripted(samples)
        max_error=float((reference-actual).abs().max())
        if not torch.isfinite(actual).all() or max_error>1e-5:raise ValueError('Export parity failure')
        sample=samples[:1]
        for _ in range(50):scripted(sample)
        times=[]
        for _ in range(300):
            start=time.perf_counter();scripted(sample);times.append((time.perf_counter()-start)*1000)
    args.output.mkdir(parents=True,exist_ok=True)
    policy=args.output/'wtw_student.ts';scripted.save(str(policy))
    interface=dict(env={key:env[key] for key in ('num_observations','num_observation_history','observe_two_prev_actions','observe_clock_inputs','observe_vel')},
        obs_scales=config['obs_scales'],control=control,normalization=config['normalization'],
        commands={'num_commands':config['commands']['num_commands'],'pacing_offset':config['commands'].get('pacing_offset',False)},
        init_state={'default_joint_angles':config['init_state']['default_joint_angles']},
        control_dt=config['sim']['dt']*control['decimation'])
    (args.output/'interface.json').write_text(json.dumps(interface,indent=2)+'\n')
    report=dict(source_url='https://github.com/Improbable-AI/walk-these-ways',
        source_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        observation_dim=env['num_observations'],history_length=env['num_observation_history'],
        history_dim=history_dim,latent_dim=model.adaptation_module[-1].out_features,
        action_dim=model.actor_body[-1].out_features,command_dim=config['commands']['num_commands'],
        observe_clock_inputs=env.get('observe_clock_inputs'),observe_vel=env.get('observe_vel'),
        observe_two_prev_actions=env.get('observe_two_prev_actions'),
        action_scale=control['action_scale'],hip_scale_reduction=control['hip_scale_reduction'],
        stiffness=control['stiffness'],damping=control['damping'],
        default_joint_angles=config['init_state']['default_joint_angles'],
        control_dt=config['sim']['dt']*control['decimation'],
        max_export_error=max_error,mean_inference_ms=float(np.mean(times)),p99_inference_ms=float(np.percentile(times,99)),
        camera_required_by_actor=False,hardware_validated=False,
        note='Offline inference only; requires WTW history, command/clock and joint-order adapters. Not model_19499.')
    (args.output/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
