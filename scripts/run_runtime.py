#!/usr/bin/env python3
"""运行 sim2real 控制层；默认只使用 dry-run。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go1_sim2real.config import load_config
from go1_sim2real.observation import Go1ObservationBuilder
from go1_sim2real.policy import TorchScriptPolicy
from go1_sim2real.runtime import run_control_loop
from go1_sim2real.safety import SafetySupervisor
from go1_sim2real.transport import DryRunTransport, JsonlTransport, UnitreeSdkTransport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--config", default=str(ROOT / "config/go1_rough.yaml"))
    parser.add_argument("--command", nargs=3, type=float, default=[0.2, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="使用安全的虚拟 transport")
    parser.add_argument("--state-jsonl", help="从 JSONL 状态日志回放，不连接真机")
    parser.add_argument(
        "--enable-hardware",
        action="store_true",
        help="显式确认启用真机控制；仅在完成 SDK 适配和吊挂测试后使用",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.state_jsonl and not args.dry_run:
        args.dry_run = True
    if not args.dry_run and config.transport.get("mode") != "unitree_sdk":
        raise SystemExit("未显式指定 --dry-run，且配置 transport.mode 不是 unitree_sdk；为安全起见已停止")
    if not args.dry_run and not args.enable_hardware:
        raise SystemExit("真机模式必须显式添加 --enable-hardware；首次测试请使用 --dry-run")

    default_pos = np.asarray(config.robot["default_joint_pos"], dtype=np.float32)
    if args.state_jsonl:
        transport = JsonlTransport(args.state_jsonl)
    elif args.dry_run:
        transport = DryRunTransport(default_pos)
    else:
        transport = UnitreeSdkTransport(config=config.raw)
    policy = TorchScriptPolicy(args.policy)
    builder = Go1ObservationBuilder(
        history_length=int(config.observation.get("history_length", 1)),
        height_scan_clip=float(config.observation.get("height_scan_clip", 1.0)),
        default_joint_pos=default_pos,
    )
    safety = SafetySupervisor(config.safety, default_pos)
    # dry-run 不需要遥控器使能；真机必须由硬件适配层在确认后调用 set_enabled(True)。
    if args.dry_run:
        safety.set_enabled(True)

    def report(step: int, reason: str, action: np.ndarray) -> None:
        if step < 5 or step % 50 == 0:
            print(f"step={step:06d} safety={reason} action_max={np.max(np.abs(action)):.3f}")

    run_control_loop(
        transport=transport,
        policy=policy,
        observation_builder=builder,
        safety=safety,
        command=args.command,
        control_dt=float(config.robot["control_dt"]),
        steps=args.steps,
        action_clip=float(config.policy.get("action_clip", 1.0)),
        on_step=report,
    )


if __name__ == "__main__":
    main()
