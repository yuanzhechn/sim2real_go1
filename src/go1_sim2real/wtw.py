"""Independent adapter for the pinned WTW Go1 baseline's observation contract.

Reference: Improbable-AI/walk-these-ways 0e7236b, LCMAgent and HistoryWrapper.
No upstream code or weights are vendored. No network or robot side effects.
"""
from __future__ import annotations
import numpy as np

JOINT_NAMES=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
SDK_ORDER=np.array([3,4,5,0,1,2,9,10,11,6,7,8])


class WtwInterface:
    def __init__(self, config):
        self.config=config;env=config['env'];self.dt=float(config['control_dt'])
        if (env['num_observations'],env['num_observation_history'])!=(70,30):
            raise ValueError('Only the pinned 70x30 WTW baseline is supported')
        if env['observe_vel'] or not env['observe_two_prev_actions'] or not env['observe_clock_inputs']:
            raise ValueError('Unsupported WTW observation contract')
        if config['commands']['num_commands']!=15 or config['commands'].get('pacing_offset',False):
            raise ValueError('Unsupported WTW gait command contract')
        self.default=np.array([config['init_state']['default_joint_angles'][name] for name in JOINT_NAMES])
        scale=config['obs_scales']
        self.command_scale=np.array([scale['lin_vel'],scale['lin_vel'],scale['ang_vel'],scale['body_height_cmd'],1,1,1,1,1,
            scale['footswing_height_cmd'],scale['body_pitch_cmd'],scale['body_roll_cmd'],scale['stance_width_cmd'],scale['stance_length_cmd'],scale['aux_reward_cmd']])
        self.reset()

    def reset(self):
        self.actions=np.zeros(12);self.previous_actions=np.zeros(12)
        self.clock=np.zeros(4);self.phase=0.;self.history=np.zeros((30,70),dtype=np.float32)

    def set_action(self, action):
        action=np.asarray(action)
        if action.shape!=(12,) or not np.isfinite(action).all():raise ValueError('Invalid WTW action')
        self.previous_actions=self.actions.copy()
        self.actions=np.clip(action,-self.config['normalization']['clip_actions'],self.config['normalization']['clip_actions']).copy()

    def observe(self, gravity, positions, velocities, commands):
        gravity=np.asarray(gravity);positions=np.asarray(positions);velocities=np.asarray(velocities);commands=np.asarray(commands)
        if gravity.shape!=(3,) or positions.shape!=(12,) or velocities.shape!=(12,) or commands.shape!=(15,):
            raise ValueError('Invalid WTW observation dimensions')
        if not all(np.isfinite(v).all() for v in (gravity,positions,velocities,commands)):
            raise ValueError('Nonfinite WTW observation')
        scale=self.config['obs_scales']
        observation=np.concatenate((gravity,commands*self.command_scale,(positions-self.default)*scale['dof_pos'],
            velocities*scale['dof_vel'],self.actions,self.previous_actions,self.clock)).astype(np.float32)
        self.history[:-1]=self.history[1:];self.history[-1]=observation
        # Upstream forms the observation before advancing its gait clock in step().
        self.phase=(self.phase+self.dt*commands[4])%1.
        phase,offset,bound=commands[5:8]
        feet=self.phase+np.array([phase+offset+bound,offset,bound,phase])
        self.clock=np.sin(2*np.pi*feet)
        return observation,self.history.reshape(-1).copy()

    def sdk_target(self, action):
        action=np.asarray(action,dtype=np.float64)
        if action.shape!=(12,) or not np.isfinite(action).all():raise ValueError('Invalid WTW target action')
        delta=action*float(self.config['control']['action_scale'])
        delta[[0,3,6,9]]*=float(self.config['control']['hip_scale_reduction'])
        return (self.default+delta)[SDK_ORDER]

    @staticmethod
    def default_commands(forward=0.):
        return np.array([forward,0,0,0,3,.5,0,0,.5,.08,0,0,.25,.4,0],dtype=np.float64)
