#!/usr/bin/env python3
"""Print the resolved Isaac Lab Go1 policy I/O ordering on the training host."""

from __future__ import annotations

from isaaclab.app import AppLauncher


app = AppLauncher(headless=True).app

import gymnasium as gym
import numpy as np
import torch

import Multi_UnitreeGo1.tasks  # noqa: F401, E402
from Multi_UnitreeGo1.tasks.manager_based.multi_unitreego1.multi_unitreego1_env_cfg import (  # noqa: E402
    MultiUnitreego1RoughEnvCfg_PLAY,
)


def main() -> None:
    cfg = MultiUnitreego1RoughEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    env = gym.make("Template-Multi-Unitreego1-Rough-Play-v0", cfg=cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    action_term = unwrapped.action_manager.get_term("joint_pos")
    print("RESOLVED_JOINT_NAMES=" + repr(list(robot.joint_names)), flush=True)
    print("ACTION_JOINT_IDS=" + repr(action_term._joint_ids), flush=True)
    print("ACTION_JOINT_NAMES=" + repr(list(action_term._joint_names)), flush=True)
    print("OBS_TERM_NAMES=" + repr(unwrapped.observation_manager.active_terms["policy"]), flush=True)
    print("OBS_TERM_DIMS=" + repr(unwrapped.observation_manager.group_obs_term_dim["policy"]), flush=True)

    policy = torch.jit.load("/tmp/go1_rough_policy_local.ts", map_location=unwrapped.device).eval()
    command_term = unwrapped.command_manager.get_term("base_velocity")
    action_samples = []
    velocity_samples = []
    with torch.inference_mode():
        for step in range(160):
            vx = 0.0 if step < 80 else 0.25
            command_term.vel_command_b[:, 0] = vx
            command_term.vel_command_b[:, 1:] = 0.0
            observations = unwrapped.observation_manager.compute()["policy"]
            actions = policy(observations)
            env.step(actions)
            action_samples.append(actions[0].cpu().numpy())
            velocity_samples.append(robot.data.root_lin_vel_b[0].cpu().numpy())
    actions_np = np.asarray(action_samples)
    velocity_np = np.asarray(velocity_samples)
    for label, sample_slice in (("ZERO", slice(20, 80)), ("FORWARD", slice(100, 160))):
        print(f"{label}_ACTION_MIN=" + repr(actions_np[sample_slice].min(axis=0).tolist()), flush=True)
        print(f"{label}_ACTION_MAX=" + repr(actions_np[sample_slice].max(axis=0).tolist()), flush=True)
        print(f"{label}_MAX_ABS=" + repr(float(np.abs(actions_np[sample_slice]).max())), flush=True)
        print(f"{label}_MEAN_BASE_VEL=" + repr(velocity_np[sample_slice].mean(axis=0).tolist()), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
