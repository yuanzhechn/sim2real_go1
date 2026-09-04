#!/usr/bin/env python3
"""吊带保护且四足着地时，保持原厂站姿并诊断足力传感器。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go1_sim2real.config import load_config
from go1_sim2real.transport import UnitreeSdkTransport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config/my_go1.yaml"))
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--ground-supported", action="store_true")
    parser.add_argument("--enable-hardware-motion", action="store_true")
    args = parser.parse_args()
    if not args.ground_supported or not args.enable_hardware_motion:
        raise SystemExit(
            "必须确认四足接触地面且吊带继续承重，并添加 "
            "--ground-supported --enable-hardware-motion"
        )
    if not 3.0 <= args.hold_seconds <= 15.0:
        raise SystemExit("--hold-seconds 必须在 [3,15] 秒内")

    loaded = load_config(args.config)
    auxiliary = loaded.transport.get("auxiliary_state", {})
    if auxiliary.get("mode") != "kinematic_contact":
        raise SystemExit("该标定要求 auxiliary_state.mode=kinematic_contact")
    config = deepcopy(loaded.raw)
    config["transport"]["kp"] = [20.0] * 12
    config["transport"]["kd"] = [1.0] * 12
    config["transport"]["torque_ff"] = [0.0] * 12
    dt = float(config["robot"]["control_dt"])
    deadband = float(config["transport"].get("remote_control", {}).get("deadband", 0.08))
    samples = []
    transport = UnitreeSdkTransport(config, require_auxiliary=True, allow_commands=True)
    try:
        initial_state = transport.read_state()
        if initial_state.emergency_stop:
            raise RuntimeError("B 急停触发")
        if initial_state.remote_axes is None or np.max(
            np.abs(initial_state.remote_axes[[0, 1, 2]])
        ) > deadband:
            raise RuntimeError("标定开始时遥控摇杆必须回中")
        default = np.asarray(config["robot"]["default_joint_pos"], dtype=np.float32)
        action_scale = float(config["robot"]["action_scale"])
        initial_joint_pos = initial_state.joint_pos.copy()
        hold_action = (initial_joint_pos - default) / action_scale
        hold_steps = int(round(args.hold_seconds / dt))
        print(
            "足力诊断开始：Kp=20，仅保持切换瞬间原厂站姿；"
            "保持吊带承重且不要移动机器人",
            flush=True,
        )
        print("hold_q=" + np.array2string(initial_joint_pos, precision=3), flush=True)
        for step in range(hold_steps):
            state = transport.read_state()
            if state.emergency_stop:
                raise RuntimeError("B 急停触发")
            if state.remote_axes is None or np.max(np.abs(state.remote_axes[[0, 1, 2]])) > deadband:
                raise RuntimeError("标定期间遥控摇杆必须回中")
            if np.max(np.abs(state.joint_vel)) > 3.0:
                raise RuntimeError("标定期间关节速度超过 3 rad/s")
            tracking_error = float(np.max(np.abs(state.joint_pos - initial_joint_pos)))
            if step > 10 and tracking_error > 0.35:
                raise RuntimeError(f"原厂站姿保持误差过大: {tracking_error:.3f} rad")
            transport.send_action(hold_action)
            samples.append(
                (
                    int(state.contact_count or 0),
                    state.foot_forces.copy(),
                    state.foot_forces_estimated.copy(),
                    state.base_lin_vel.copy(),
                    bool(state.base_lin_vel_valid),
                )
            )
            if step % 25 == 0:
                print(
                    f"step={step:04d} contacts={state.contact_count} "
                    f"force_raw={np.array2string(state.foot_forces, precision=1)} "
                    f"force_est={np.array2string(state.foot_forces_estimated, precision=1)} "
                    f"hold_error={tracking_error:.3f}",
                    flush=True,
                )
            time.sleep(dt)

        contacts = np.asarray([item[0] for item in samples])
        force_raw = np.asarray([item[1] for item in samples])
        force_est = np.asarray([item[2] for item in samples])
        velocities = np.asarray([item[3] for item in samples])
        valid = np.asarray([item[4] for item in samples])
        valid_ratio = float(np.mean(valid))
        contact_ratio = float(np.mean(contacts >= 2))
        velocity_rms = np.sqrt(np.mean(velocities * velocities, axis=0))
        print(
            f"标定统计: valid_ratio={valid_ratio:.3f} contact_ratio={contact_ratio:.3f} "
            f"velocity_rms={np.array2string(velocity_rms, precision=4)} "
            f"force_raw_mean={np.array2string(force_raw.mean(axis=0), precision=1)} "
            f"force_raw_std={np.array2string(force_raw.std(axis=0), precision=2)} "
            f"force_est_mean={np.array2string(force_est.mean(axis=0), precision=1)}",
            flush=True,
        )
        if valid_ratio < 0.9 or contact_ratio < 0.9:
            raise RuntimeError("可靠接触比例不足 90%，不得进入落地策略测试")
        if np.max(velocity_rms) > 0.15:
            raise RuntimeError("静止速度估计漂移超过 0.15 m/s，不得进入落地策略测试")
        print("四足接触与静止速度估计标定通过", flush=True)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
