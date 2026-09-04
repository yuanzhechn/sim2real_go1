#!/usr/bin/env python3
"""吊挂状态下以低增益从当前关节角平滑移动到策略默认姿态。"""

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
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--max-dq", type=float, default=6.0)
    parser.add_argument("--suspended", action="store_true")
    parser.add_argument("--enable-hardware-motion", action="store_true")
    args = parser.parse_args()
    if not args.suspended or not args.enable_hardware_motion:
        raise SystemExit("主动标定必须同时添加 --suspended --enable-hardware-motion")
    if args.duration < 3.0 or args.hold < 0 or not (0 < args.kp <= 10) or not (0 < args.kd <= 2):
        raise SystemExit("要求 duration>=3、0<=hold、0<Kp<=10、0<Kd<=2")

    loaded = load_config(args.config)
    config = deepcopy(loaded.raw)
    config["transport"]["kp"] = [float(args.kp)] * 12
    config["transport"]["kd"] = [float(args.kd)] * 12
    config["transport"]["torque_ff"] = [0.0] * 12
    default = np.asarray(config["robot"]["default_joint_pos"], dtype=np.float32)
    scale = float(config["robot"]["action_scale"])
    dt = float(config["robot"]["control_dt"])
    deadband = float(config["transport"].get("remote_control", {}).get("deadband", 0.08))

    transport = UnitreeSdkTransport(config, require_auxiliary=False, allow_commands=True)
    try:
        state = transport.read_state()
        if state.emergency_stop:
            raise RuntimeError("B 急停已触发")
        if np.max(np.abs(state.remote_axes[[0, 1, 2]])) > deadband:
            raise RuntimeError("遥控摇杆未回中")
        if state.battery_voltage is None or state.battery_voltage < 21.0:
            raise RuntimeError(f"电池电压不适合主动标定: {state.battery_voltage}")
        if np.max(state.motor_temperatures) > 60:
            raise RuntimeError("电机温度过高")

        initial = state.joint_pos.copy()
        initial_action = (initial - default) / scale
        total_steps = max(1, int(round(args.duration / dt)))
        hold_steps = max(0, int(round(args.hold / dt)))
        print("主动标定开始: low-gain Kp=%.1f Kd=%.1f" % (args.kp, args.kd))
        print("initial_q=" + np.array2string(initial, precision=3))
        print("target_q =" + np.array2string(default, precision=3))

        for step in range(total_steps + hold_steps):
            state = transport.read_state()
            if state.emergency_stop:
                raise RuntimeError("B 急停触发")
            if np.max(np.abs(state.remote_axes[[0, 1, 2]])) > deadband:
                raise RuntimeError("遥控摇杆离开中心，停止标定")
            dq_max = float(np.max(np.abs(state.joint_vel)))
            if dq_max > args.max_dq:
                raise RuntimeError(f"关节速度过高: {dq_max:.3f} rad/s")
            if np.max(state.motor_temperatures) > 60:
                raise RuntimeError("电机温度超过 60°C")

            if step < total_steps:
                ratio = min(1.0, (step + 1) / float(total_steps))
                smooth = ratio * ratio * (3.0 - 2.0 * ratio)
                action = initial_action * (1.0 - smooth)
            else:
                action = np.zeros(12, dtype=np.float32)
            intended = default + action * scale
            tracking_error = float(np.max(np.abs(state.joint_pos - intended)))
            if step > 10 and tracking_error > 0.45:
                raise RuntimeError(f"关节未跟随目标: error={tracking_error:.3f} rad")
            transport.send_action(action)
            if step % 25 == 0 or step == total_steps - 1:
                print(
                    "step=%03d alpha=%.3f dq_max=%.3f err=%.3f q=%s"
                    % (step, min(1.0, (step + 1) / float(total_steps)), dq_max,
                       tracking_error, np.array2string(state.joint_pos, precision=2))
                )
            time.sleep(dt)
        print("低增益默认姿态标定完成")
    finally:
        transport.close()


if __name__ == "__main__":
    main()
