from pathlib import Path
import json
import time

import numpy as np
import pytest
import yaml

from go1_sim2real.elevation import elevation_observation
from go1_sim2real.observation import Go1ObservationBuilder
from go1_sim2real.transport import DryRunTransport, JsonlTransport
from go1_sim2real.config import load_config, validate_hardware_config
from go1_sim2real.bundle import load_bundle, validate_bundle_runtime_config
from go1_sim2real.sensors import UdpJsonAuxiliaryStateProvider
from go1_sim2real.safety import SafetySupervisor

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'artifacts/yuanzhe/collective_harder_19499_1072d_20261014'
CONFIG = ROOT/'config/robots/unitree5151_19499.yaml'
TERMS = list(Go1ObservationBuilder.ROUGH_TERMS[:-1])+['terrain_height_map']


def test_unknown_map_is_zero_with_invalid_mask():
    state=DryRunTransport(np.zeros(12)).read_state()
    obs=Go1ObservationBuilder(terms=TERMS).build(state,[.1,.2,.3])
    assert obs.shape==(1072,)
    np.testing.assert_allclose(obs[9:12],[.1,.2,.3])
    np.testing.assert_array_equal(obs[48:],0)
    assert Go1ObservationBuilder().observation_dim==235


def test_map_clips_heights_and_clears_invalid_values():
    heights=np.full(512,-.34); validity=np.ones(512)
    heights[0]=3; validity[1]=0
    obs=elevation_observation(heights,validity)
    np.testing.assert_allclose(obs[:3],[1,0,-.34])
    np.testing.assert_array_equal(obs[512:515],[1,0,1])


@pytest.mark.parametrize('bad', [np.full(512,.5),np.full(512,np.nan)])
def test_map_rejects_invalid_validity(bad):
    with pytest.raises(ValueError): elevation_observation(np.zeros(512),bad)


def test_map_rejects_wrong_shape():
    with pytest.raises(ValueError): elevation_observation(np.zeros(187),np.zeros(512))


def test_udp_map_cannot_silently_omit_fields():
    provider=UdpJsonAuxiliaryStateProvider('127.0.0.1',0,require_height_scan=False,require_height_map=True)
    try:
        with pytest.raises(KeyError): provider._decode(b'{"base_lin_vel":[0,0,0]}')
        message=dict(base_lin_vel=[.1,0,0],terrain_heights=[-.34]*512,terrain_validity=[1]*512)
        frame=provider._decode(json.dumps(message).encode())
        np.testing.assert_allclose(frame.terrain_heights,-.34)
        np.testing.assert_array_equal(frame.terrain_validity,1)
    finally: provider.close()


def test_replay_retains_elevation(tmp_path):
    frame=dict(base_lin_vel=[0]*3,base_ang_vel=[0]*3,projected_gravity=[0,0,-1],
               joint_pos=[0]*12,joint_vel=[0]*12,last_action=[0]*12,height_scan=[0]*187,
               terrain_heights=[-.34]*512,terrain_validity=[1]*512)
    path=tmp_path/'state.jsonl'; path.write_text(json.dumps(frame)+'\n')
    state=JsonlTransport(str(path)).read_state()
    obs=Go1ObservationBuilder(terms=TERMS).build(state,[0,0,0])
    np.testing.assert_allclose(obs[48:560],-.34)
    np.testing.assert_array_equal(obs[560:],1)


def test_required_map_faults_when_all_cells_unknown():
    state=DryRunTransport(np.zeros(12)).read_state()
    safety=SafetySupervisor(dict(require_enable_switch=False,require_height_map_valid=True),np.zeros(12))
    assert safety.validate(state,np.zeros(12)).reason=='height_map_unavailable'
    state.terrain_validity[:]=1
    assert safety.validate(state,np.zeros(12)).allowed


def test_19499_bundle_matches_and_refuses_unvalidated_hardware():
    bundle=load_bundle(BUNDLE); config=load_config(CONFIG)
    validate_bundle_runtime_config(bundle,config)
    np.testing.assert_allclose(config.robot['default_joint_pos'][4:8], .75)
    assert bundle.manifest['checkpoint_sha256']=='119461a7d5a665f42fddefe48732bcfa932461a22bc256c1308009ddca9907d4'
    with pytest.raises(ValueError,match='hardware_validated'): validate_hardware_config(config)
    config.transport['hardware_validated']=True
    config.transport['calibration_id']='test-only'
    with pytest.raises(ValueError,match='blind locomotion'): validate_hardware_config(config)


@pytest.mark.parametrize("freeze", [False, True])
def test_exported_actor_matches_checkpoint_reference(freeze):
    import torch
    fixture=np.load(BUNDLE/'validation/parity.npz')
    from go1_sim2real.policy import TorchScriptPolicy
    actor=TorchScriptPolicy(BUNDLE/'policy/policy.ts', cpu_threads=4, freeze_for_inference=freeze)
    actions=np.stack([actor(obs).numpy() for obs in fixture['observations']])
    np.testing.assert_allclose(actions,fixture['actions'],rtol=1e-5,atol=2e-6)


def test_grid_mismatch_is_rejected(tmp_path):
    config=yaml.safe_load(CONFIG.read_text()); config['observation']['height_map']['resolution_m']=.2
    path=tmp_path/'bad.yaml'; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='grid contract'): load_config(path)


def test_unknown_mode_never_claims_a_received_map_is_valid():
    state=DryRunTransport(np.zeros(12)).read_state()
    state.terrain_heights[:]=-.34
    state.terrain_validity[:]=1
    obs=Go1ObservationBuilder(terms=TERMS,height_map_mode="unknown").build(state,[0,0,0])
    np.testing.assert_array_equal(obs[48:],0)
