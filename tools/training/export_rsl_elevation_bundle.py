#!/usr/bin/env python3
"""Export the exact unnormalized 1072-D RSL-RL ELU actor into a new bundle."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from go1_sim2real.elevation import HEIGHT_MAP_SPEC


class ElevationActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1072, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)


def load_actor(checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['actor_state_dict']
    actor = ElevationActor()
    expected = set(actor.state_dict()) | {'distribution.std_param'}
    if set(state) != expected:
        raise ValueError('Unsupported actor keys (normalization/recurrent/architecture changes must not be dropped)')
    if state['distribution.std_param'].shape != (12,):
        raise ValueError('Unexpected distribution dimension')
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint weights')
    actor.load_state_dict({k: v for k, v in state.items() if k.startswith('mlp.')}, strict=True)
    return actor.eval(), state


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--robot-config', type=Path, required=True)
    parser.add_argument('--training-spec', type=Path, required=True, help='Verified training observation/action contract, not hardware defaults')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing bundle')
    torch.set_num_threads(4)
    actor, state = load_actor(args.checkpoint)
    config = yaml.safe_load(args.robot_config.read_text())
    spec = yaml.safe_load(args.training_spec.read_text())
    if spec['policy']['observation_dim'] != 1072 or spec['policy']['hidden_layers'] != [512,256,128]:
        raise ValueError('Training spec does not match supported actor architecture')
    if spec['observation']['height_map'] != HEIGHT_MAP_SPEC:
        raise ValueError('Training elevation contract differs')
    config['robot'] = spec['robot']
    config['observation'] = dict(spec['observation'], height_map_mode='unknown')
    config['policy'] = spec['policy']
    config['transport']['base_hardware_calibration_id'] = config['transport']['calibration_id']
    config['transport']['hardware_validated'] = False
    config['transport']['calibration_id'] = ''
    config['transport']['blind_deployment_validated'] = False
    # Keep the existing calibrated hardware parameters; model gate is separate.
    for name in ('policy','checkpoint','config','training','validation'):
        (args.output/name).mkdir(parents=True)
    policy_path=args.output/'policy/policy.ts'
    torch.jit.script(actor).save(str(policy_path))
    shutil.copy2(args.checkpoint,args.output/'checkpoint/model_19499.pt')
    for name in ('adaptive_leg_profile.json','continuation_settings.json','healthy_gait_profile.json','posture_profile.json'):
        shutil.copy2(args.checkpoint.parent/name,args.output/'training'/name)
    shutil.copy2(args.training_spec, args.output/'training/actor_contract.yaml')
    config_path=args.output/'config/deployment.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    # Reference uses functional operations directly on checkpoint tensors.
    generator=torch.Generator().manual_seed(19499)
    obs=torch.randn(32,1072,generator=generator)*.1
    obs[:,6:9]=torch.tensor([0.,0.,-1.])
    obs[:16,48:]=0 # no-camera / all cells unknown
    obs[16:,48:560]=-.34
    obs[16:,560:]=1
    with torch.inference_mode():
        reference=obs
        for index in (0,2,4,6):
            reference=F.linear(reference,state[f'mlp.{index}.weight'],state[f'mlp.{index}.bias'])
            if index != 6: reference=F.elu(reference)
        actual=torch.jit.load(str(policy_path))(obs)
    max_error=float((reference-actual).abs().max())
    if not torch.allclose(reference, actual, rtol=1e-6, atol=1e-6):
        raise ValueError('Export changed deterministic actor output')
    np.savez(args.output/'validation/parity.npz',observations=obs.numpy(),actions=reference.numpy())
    (args.output/'validation/export_report.json').write_text(json.dumps(dict(
        checkpoint_sha256=sha(args.checkpoint),max_abs_error=max_error,samples=32,
        torch_version=torch.__version__,camera_missing_samples=16,
        note='Numerical parity only; not locomotion or hardware qualification.'),indent=2)+'\n')
    manifest=dict(framework='rsl_rl',checkpoint='checkpoint/model_19499.pt',
                  checkpoint_sha256=sha(args.checkpoint),policy='policy/policy.ts',policy_sha256=sha(policy_path),
                  config='config/deployment.yaml',observation_dim=1072,action_dim=12,
                  hidden_layers=[512,256,128],action_scale=.25,height_scan_dim=0,height_map_dim=1024,
                  recurrent=False,obs_normalization=False,source_iteration=19499,
                  deployment_status='offline_validated_hardware_pending')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    print('export_reference_max_error=',max_error)


if __name__=='__main__':
    main()
