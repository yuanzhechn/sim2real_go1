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
    args = parser.parse_args()
    if not args.enable_hardware_read:
        raise SystemExit("必须显式添加 --enable-hardware-read")
    config = load_config(args.config)
    transport = UnitreeSdkTransport(
        config.raw, require_auxiliary=False, allow_commands=False
    )
    try:
        for step in range(args.steps):
            state = transport.read_state()
            if step % 25 == 0:
                print(
                    f"step={step:05d} roll={state.roll:+.3f} pitch={state.pitch:+.3f} "
                    f"q={np.array2string(state.joint_pos, precision=3)} "
                    f"dq_max={np.max(np.abs(state.joint_vel)):.3f} "
                    f"temp_max={np.max(state.motor_temperatures):.0f} "
                    f"battery={state.battery_voltage} enable={state.enable_switch} estop={state.emergency_stop}"
                )
            time.sleep(float(config.robot["control_dt"]))
    finally:
        transport.close()


if __name__ == "__main__":
    main()
