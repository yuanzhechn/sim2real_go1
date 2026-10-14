#!/usr/bin/env python3
"""Validate a bundle and benchmark CPU inference without importing robot SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from go1_sim2real.bundle import load_bundle,validate_bundle_runtime_config
from go1_sim2real.config import load_config
from go1_sim2real.policy import TorchScriptPolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    bundle=load_bundle(args.bundle);config=load_config(args.config)
    validate_bundle_runtime_config(bundle,config)
    checkpoint=(bundle.root/bundle.manifest['checkpoint']).resolve()
    checkpoint.relative_to(bundle.root)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==bundle.manifest['checkpoint_sha256'], 'Checkpoint SHA256 mismatch'
    fixture=np.load(bundle.root/'validation/parity.npz')
    threads=int(config.policy.get('cpu_threads',4))
    frozen=bool(config.policy.get('freeze_for_inference',False))
    observations=fixture['observations']
    references=fixture['actions']
    policy=TorchScriptPolicy(bundle.policy_path,cpu_threads=threads,freeze_for_inference=frozen)
    actual=np.stack([policy(obs).numpy() for obs in observations])
    np.testing.assert_allclose(actual,references,atol=2e-6,rtol=1e-5)
    for _ in range(50): policy(observations[0])
    durations=[]
    for index in range(500):
        start=time.perf_counter()
        policy(observations[index%len(observations)])
        durations.append((time.perf_counter()-start)*1000)
    period=float(config.robot['control_dt'])*1000
    report=dict(torch_version=torch.__version__,cpu_threads=threads,freeze_for_inference=frozen,samples=len(actual),
                checkpoint_sha256=bundle.manifest['checkpoint_sha256'],
                policy_sha256=bundle.manifest['policy_sha256'],
                max_abs_error=float(np.max(np.abs(actual-references))),
                inference_ms_mean=float(np.mean(durations)),inference_ms_p95=float(np.percentile(durations,95)),
                inference_ms_p99=float(np.percentile(durations,99)),inference_ms_max=float(np.max(durations)),
                control_period_ms=period,
                inference_p99_within_period=bool(np.percentile(durations,99)<period),
                hardware_started=False,
                note='CPU inference only, not full control-loop latency or blind locomotion qualification.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
