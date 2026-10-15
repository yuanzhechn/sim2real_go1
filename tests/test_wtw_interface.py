import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from go1_sim2real.wtw import WtwInterface,SDK_ORDER


@pytest.mark.filterwarnings('ignore:__array_wrap__:DeprecationWarning')
def test_interface_matches_pinned_upstream_without_robot_or_lcm(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    upstream=root.parent/'walk-these-ways'
    if not upstream.is_dir():pytest.skip('Optional sibling WTW clone is absent')
    spec=importlib.util.spec_from_file_location('wtw_preflight',root/'tools/training/preflight_walk_these_ways.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    path=upstream/'runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/parameters.pkl'
    with path.open('rb') as file:config=exporter.ConfigReader(file).load()['Cfg']
    config['control_dt']=config['control']['decimation']*config['sim']['dt']
    adapter=WtwInterface(config)
    packets=[]
    class FakeLCM:
        def publish(self,channel,data):packets.append((channel,data))
    monkeypatch.setitem(sys.modules,'lcm',types.SimpleNamespace(LCM=lambda *_:FakeLCM()))
    monkeypatch.setitem(sys.modules,'cv2',types.SimpleNamespace())
    monkeypatch.syspath_prepend(str(upstream))
    def load(name,relative):
        spec=importlib.util.spec_from_file_location(name,upstream/relative)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    module=load('wtw_agent_reference','go1_gym_deploy/envs/lcm_agent.py')
    history_module=load('wtw_history_reference','go1_gym_deploy/envs/history_wrapper.py')
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    state={'q':adapter.default.copy(),'dq':np.zeros(12),'g':np.array([0.,0.,-1.]),'cmd':adapter.default_commands(.1)}
    class Estimator:
        joint_idxs=SDK_ORDER
        def get_gravity_vector(self):return state['g']
        def get_dof_pos(self):return state['q']
        def get_dof_vel(self):return state['dq']
        def get_body_linear_vel(self):return np.zeros(3)
        def get_body_angular_vel(self):return np.zeros(3)
        def get_contact_state(self):return np.ones(4)
        def get_camera_front(self):return None
        get_camera_bottom=get_camera_front;get_camera_rear=get_camera_front
        get_camera_left=get_camera_front;get_camera_right=get_camera_front
    class Commands:
        def get_command(self,*args,**kwargs):return state['cmd'],False
    agent=module.LCMAgent(config,Estimator(),Commands())
    history=history_module.HistoryWrapper(agent)
    assert not torch.count_nonzero(history.reset()['obs_history'])
    random=np.random.default_rng(42)
    for _ in range(50):
        action=random.uniform(-1,1,12).astype(np.float32)
        state['q']=adapter.default+random.normal(0,.05,12)
        state['dq']=random.normal(0,.1,12)
        expected,_,_,_=history.step(torch.from_numpy(action[None,:]))
        adapter.set_action(action)
        observation,actual=adapter.observe(state['g'],state['q'],state['dq'],state['cmd'])
        np.testing.assert_allclose(observation,expected['obs'].numpy()[0],atol=2e-6,rtol=1e-6)
        np.testing.assert_allclose(actual,expected['obs_history'].numpy()[0],atol=2e-6,rtol=1e-6)
        packet=module.pd_tau_targets_lcmt.decode(packets[-1][1])
        np.testing.assert_allclose(adapter.sdk_target(action),packet.q_des,atol=1e-7)
    with pytest.raises(ValueError):adapter.set_action(np.full(12,np.nan))
    with pytest.raises(ValueError):adapter.observe([0,0,-1],state['q'],state['dq'],np.zeros(3))
