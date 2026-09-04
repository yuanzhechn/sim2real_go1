#!/usr/bin/env python3
"""只读检查 Go1 低层状态；不会发送策略位置命令。"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--config", default=str(ROOT / "config/go1_rough.yaml"))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument(
        "--enable-hardware-read",
        action="store_true",
        help="确认已吊挂/断开执行器并只读取 LowState",
    )
    parser.add_argument(
        "--passive-bootstrap",
        action="store_true",
        help="sport 已停止且机器人已吊挂时，发送电机失能/零力矩包建立 LowState 回传",
    )
    parser.add_argument(
        "--expect-estop",
        action="store_true",
        help="在读取窗口内等待并验证 B 急停映射；仅可与 --passive-bootstrap 一起使用",
    )
    parser.add_argument(
        "--dump-remote-changes",
        action="store_true",
        help="遥控器原始数据变化时打印按键掩码和前 24 字节，用于真机按键诊断",
    )
    args = parser.parse_args()
    if not args.enable_hardware_read:
        raise SystemExit("必须显式添加 --enable-hardware-read")
    if args.expect_estop and not args.passive_bootstrap:
        raise SystemExit("--expect-estop 必须与 --passive-bootstrap 一起使用")
    config = load_config(args.config)
    transport = UnitreeSdkTransport(
        config.raw,
        require_auxiliary=False,
        allow_commands=args.passive_bootstrap,
        passive_only=args.passive_bootstrap,
    )
    try:
        estop_seen = False
        previous_remote: bytes | None = None
        for step in range(args.steps):
            state = transport.read_state()
            if args.dump_remote_changes:
                remote = bytes(int(value) & 0xFF for value in transport._state.wirelessRemote)
                if remote != previous_remote:
                    buttons = remote[2] | (remote[3] << 8) if len(remote) >= 4 else 0
                    print(
                        f"remote_change step={step:05d} buttons=0x{buttons:04x} "
                        f"raw24={remote[:24].hex()}",
                        flush=True,
                    )
                    previous_remote = remote
            if step % 25 == 0:
                foot_force = np.asarray(
                    getattr(transport._state, "footForce", [0, 0, 0, 0]),
                    dtype=np.float32,
                )
                foot_force_est = np.asarray(
                    getattr(transport._state, "footForceEst", [0, 0, 0, 0]),
                    dtype=np.float32,
                )
                print(
                    f"step={step:05d} roll={state.roll:+.3f} pitch={state.pitch:+.3f} "
                    f"q={np.array2string(state.joint_pos, precision=3)} "
                    f"dq_max={np.max(np.abs(state.joint_vel)):.3f} "
                    f"temp_max={np.max(state.motor_temperatures):.0f} "
                    f"battery={state.battery_voltage} "
                    f"foot_force={np.array2string(foot_force, precision=0)} "
                    f"foot_est={np.array2string(foot_force_est, precision=0)} "
                    f"axes={np.array2string(state.remote_axes, precision=2)} "
                    f"enable={state.enable_switch} estop={state.emergency_stop}"
                )
            if args.expect_estop and state.emergency_stop:
                print(f"B 急停映射验证通过: step={step}")
                estop_seen = True
                break
            time.sleep(float(config.robot["control_dt"]))
        if args.expect_estop and not estop_seen:
            raise RuntimeError("读取窗口内没有检测到 B 急停")
    finally:
        transport.close()


if __name__ == "__main__":
    main()
