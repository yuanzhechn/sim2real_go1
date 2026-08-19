#!/usr/bin/env python3
"""运行 sim2real 控制层；默认只使用 dry-run。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go1_sim2real.config import load_config, validate_hardware_config
from go1_sim2real.bundle import load_bundle, validate_bundle_runtime_config
from go1_sim2real.observation import Go1ObservationBuilder
from go1_sim2real.policy import TorchScriptPolicy
from go1_sim2real.remote import RemoteVelocityCommand
from go1_sim2real.runtime import run_control_loop
from go1_sim2real.safety import SafetySupervisor
from go1_sim2real.transport import DryRunTransport, JsonlTransport, UnitreeSdkTransport


def main() -> None:
    parser = argparse.ArgumentParser()
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--policy", help="直接加载 TorchScript 策略")
    model_source.add_argument("--bundle", help="加载并完整校验训练侧 sim2real bundle")
    parser.add_argument("--config", default=str(ROOT / "config/go1_rough.yaml"))
    parser.add_argument("--command", nargs=3, type=float, default=[0.2, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    parser.add_argument(
        "--command-source",
        choices=("auto", "fixed", "remote"),
        default="auto",
        help="auto: dry-run 使用 --command，真机使用遥控摇杆",
    )
    parser.add_argument("--steps", type=int, default=100, help="控制步数；0 表示持续运行")
    parser.add_argument("--dry-run", action="store_true", help="使用安全的虚拟 transport")
    parser.add_argument("--state-jsonl", help="从 JSONL 状态日志回放，不连接真机")
    parser.add_argument("--log-jsonl", help="记录每步状态、安全结果和动作（真机强烈建议启用）")
    parser.add_argument(
        "--enable-hardware",
        action="store_true",
        help="显式确认启用真机控制；仅在完成 SDK 适配和吊挂测试后使用",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只校验配置、bundle 和模型签名，不连接机器人",
    )
    parser.add_argument(
        "--manage-sport-mode",
        action="store_true",
        help="模型预热后停止原厂 sport mode，并在退出时恢复",
    )
    args = parser.parse_args()
    if args.steps < 0:
        raise SystemExit("--steps 不能为负数；使用 0 表示持续运行")
    if args.manage_sport_mode and args.dry_run:
        raise SystemExit("--manage-sport-mode 不能与 --dry-run 同时使用")

    config = load_config(args.config)
    policy_path = args.policy
    if args.bundle:
        try:
            bundle = load_bundle(args.bundle)
            validate_bundle_runtime_config(bundle, config)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"bundle 校验失败: {exc}") from exc
        policy_path = str(bundle.policy_path)
        print(
            "bundle 校验通过: "
            f"policy_sha256={bundle.manifest['policy_sha256']} "
            f"obs={bundle.manifest['observation_dim']} action={bundle.manifest['action_dim']}"
        )
    if args.state_jsonl and not args.dry_run:
        args.dry_run = True
    if not args.dry_run and config.transport.get("mode") != "unitree_sdk":
        raise SystemExit("未显式指定 --dry-run，且配置 transport.mode 不是 unitree_sdk；为安全起见已停止")
    if not args.dry_run and not args.enable_hardware:
        raise SystemExit("真机模式必须显式添加 --enable-hardware；首次测试请使用 --dry-run")
    if not args.dry_run:
        try:
            validate_hardware_config(config)
        except ValueError as exc:
            raise SystemExit(f"真机配置未通过安全校验: {exc}") from exc

    policy = TorchScriptPolicy(policy_path)
    default_pos = np.asarray(config.robot["default_joint_pos"], dtype=np.float32)
    builder = Go1ObservationBuilder(
        history_length=int(config.observation.get("history_length", 1)),
        height_scan_clip=float(config.observation.get("height_scan_clip", 1.0)),
        default_joint_pos=default_pos,
        terms=config.observation.get("terms"),
    )
    # 在打开真机 UDP 和建立低层状态通道前完成模型加载/签名检查。PyTorch 首次
    # 加载在 ARM64 上可能耗时数秒，不能让硬件在这段时间处于半初始化状态。
    probe_action = np.asarray(
        policy(np.zeros(builder.observation_dim, dtype=np.float32)), dtype=np.float32
    ).reshape(-1)
    if probe_action.size != int(config.policy["action_dim"]) or not np.all(np.isfinite(probe_action)):
        raise SystemExit(
            f"策略签名无效: 期望 {config.policy['action_dim']} 个有限动作，实际 {probe_action.size}"
        )
    if args.preflight_only:
        print("preflight 校验通过；未连接机器人")
        return

    safety = SafetySupervisor(config.safety, default_pos)
    # 真机的逐帧遥控器状态仍由 SafetySupervisor 检查；这里仅打开软件总门。
    safety.set_enabled(True)

    use_remote_command = args.command_source == "remote" or (
        args.command_source == "auto" and not args.dry_run
    )
    max_command = float(config.safety.get("max_command_velocity", 1.0))
    if not use_remote_command and np.max(np.abs(np.asarray(args.command, dtype=np.float32))) > max_command:
        raise SystemExit(f"速度指令超过 safety.max_command_velocity={max_command}")
    command_provider = (
        RemoteVelocityCommand(config.transport.get("remote_control", {}))
        if use_remote_command
        else None
    )

    log_stream = open(args.log_jsonl, "a", encoding="utf-8", buffering=1) if args.log_jsonl else None

    def report(step: int, reason: str, action: np.ndarray, state, command: np.ndarray) -> None:
        if step < 5 or step % 50 == 0:
            print(
                f"step={step:06d} safety={reason} "
                f"command={np.array2string(command, precision=2)} "
                f"action_max={np.max(np.abs(action)):.3f}"
            )
        if log_stream is not None:
            record = {
                "monotonic_time": time.monotonic(),
                "step": step,
                "safety": reason,
                "command": command.tolist(),
                "action": action.tolist(),
                "base_lin_vel": state.base_lin_vel.tolist(),
                "base_ang_vel": state.base_ang_vel.tolist(),
                "projected_gravity": state.projected_gravity.tolist(),
                "joint_pos": state.joint_pos.tolist(),
                "joint_vel": state.joint_vel.tolist(),
                "height_scan": state.height_scan.tolist(),
                "roll": state.roll,
                "pitch": state.pitch,
                "battery_voltage": state.battery_voltage,
                "motor_temperatures": None if state.motor_temperatures is None else state.motor_temperatures.tolist(),
                "enable_switch": state.enable_switch,
                "emergency_stop": state.emergency_stop,
            }
            log_stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    sport_restore_required = bool(args.manage_sport_mode)
    transport = None
    try:
        if args.manage_sport_mode:
            subprocess.run(
                [
                    "sudo", "-n", "/usr/bin/python3",
                    str(ROOT / "scripts" / "sport_mode_control.py"), "stop",
                ],
                check=True,
            )
        if args.state_jsonl:
            transport = JsonlTransport(args.state_jsonl)
        elif args.dry_run:
            transport = DryRunTransport(default_pos)
        else:
            transport = UnitreeSdkTransport(config=config.raw)
        run_control_loop(
            transport=transport,
            policy=policy,
            observation_builder=builder,
            safety=safety,
            command=args.command,
            control_dt=float(config.robot["control_dt"]),
            steps=None if args.steps == 0 else args.steps,
            action_clip=float(config.policy.get("action_clip", 1.0)),
            on_step=report,
            command_provider=command_provider,
        )
    finally:
        if log_stream is not None:
            log_stream.close()
        if sport_restore_required:
            print("正在恢复原厂 sport mode...")
            subprocess.run(
                [
                    "sudo", "-n", "/usr/bin/python3",
                    str(ROOT / "scripts" / "sport_mode_control.py"), "start",
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
