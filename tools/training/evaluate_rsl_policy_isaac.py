#!/usr/bin/env python3
"""Evaluate an exported 235-D Go1 policy in Isaac Lab without a training runner."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--vx", type=float, default=0.15)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument(
    "--asset-root", default="/shared_data/Assets/Isaac/5.1/Isaac"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import (  # noqa: E402
    UnitreeGo1RoughEnvCfg_PLAY,
)


def main() -> None:
    cfg = UnitreeGo1RoughEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.visual_material = None
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.vx, args.vx)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    asset_root = args.asset_root.rstrip("/")
    cfg.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.spawn.usd_path = f"{asset_root}/Robots/Unitree/Go1/go1.usd"
    cfg.scene.robot.actuators["base_legs"].network_file = (
        f"{asset_root}/IsaacLab/ActuatorNets/Unitree/unitree_go1.pt"
    )
    cfg.scene.sky_light.spawn.texture_file = (
        f"{asset_root}/Materials/Textures/Skies/PolyHaven/"
        "kloofendal_43d_clear_puresky_4k.hdr"
    )

    env = gym.make("Isaac-Velocity-Rough-Unitree-Go1-Play-v0", cfg=cfg)
    robot = env.unwrapped.scene["robot"]
    policy = torch.jit.load(args.policy, map_location=env.unwrapped.device).eval()
    observation, _ = env.reset()
    actions = []
    velocities = []
    positions = []
    terminations = 0
    with torch.inference_mode():
        for _ in range(args.steps):
            action = policy(observation["policy"])
            observation, _reward, terminated, truncated, _info = env.step(action)
            actions.append(action[0].cpu().numpy())
            velocities.append(robot.data.root_lin_vel_b[0].cpu().numpy())
            positions.append(robot.data.root_pos_w[0].cpu().numpy())
            terminations += int(bool(torch.any(terminated | truncated).item()))

    action_array = np.asarray(actions)
    velocity_array = np.asarray(velocities)
    position_array = np.asarray(positions)
    tail = slice(max(0, args.steps // 2), args.steps)
    print("JOINT_NAMES=" + repr(list(robot.joint_names)), flush=True)
    print("ACTION_MIN=" + repr(action_array.min(axis=0).tolist()), flush=True)
    print("ACTION_MAX=" + repr(action_array.max(axis=0).tolist()), flush=True)
    print("ACTION_MAX_ABS=" + repr(float(np.abs(action_array).max())), flush=True)
    print("MEAN_BASE_VEL_TAIL=" + repr(velocity_array[tail].mean(axis=0).tolist()), flush=True)
    print("POSITION_DELTA=" + repr((position_array[-1] - position_array[0]).tolist()), flush=True)
    print("TERMINATIONS=" + repr(terminations), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
