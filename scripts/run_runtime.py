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
    parser.add_argument(
        "--suspended-test",
        action="store_true",
        help="仅限吊挂测试：固定零速度输入并暂用零 base_lin_vel；禁止落地运行",
    )
    parser.add_argument(
        "--suspended-remote-test",
        action="store_true",
        help="仅与 --suspended-test 配合：允许吊挂状态用遥控器给策略速度指令",
    )
    parser.add_argument(
        "--ground-zero-velocity-test",
        action="store_true",
        help="吊带保护的短时落地试走：仅临时以零值代替未标定的 base_lin_vel",
    )
    parser.add_argument(
        "--ground-remote-control",
        action="store_true",
        help="落地持续运动模式：模型起立后由遥控器给速度指令，直到 B/Ctrl+C 退出",
    )
    parser.add_argument(
        "--remote-scale",
        type=float,
        default=1.0,
        help="遥控速度额外缩放，必须在 (0,1]，首次吊挂测试建议 0.5",
    )
    parser.add_argument(
        "--action-limit",
        type=float,
        help="仅允许缩小 policy.action_clip，用于分阶段吊挂测试",
    )
    args = parser.parse_args()
    if args.steps < 0:
        raise SystemExit("--steps 不能为负数；使用 0 表示持续运行")
    if args.manage_sport_mode and args.dry_run:
        raise SystemExit("--manage-sport-mode 不能与 --dry-run 同时使用")
    if args.suspended_test:
        if args.dry_run or not args.enable_hardware:
            raise SystemExit("--suspended-test 必须与真机 --enable-hardware 一起使用")
        max_suspended_steps = 1500 if args.suspended_remote_test else 500
        if args.steps <= 0 or args.steps > max_suspended_steps:
            raise SystemExit(
                f"--suspended-test 要求 1<=--steps<={max_suspended_steps}"
            )
        if args.suspended_remote_test:
            if args.command_source != "remote":
                raise SystemExit("--suspended-remote-test 要求 --command-source remote")
        elif args.command_source == "remote" or np.max(np.abs(np.asarray(args.command))) > 0:
            raise SystemExit("--suspended-test 默认只允许 --command-source fixed --command 0 0 0")
    elif args.suspended_remote_test:
        raise SystemExit("--suspended-remote-test 必须与 --suspended-test 一起使用")
    if args.ground_zero_velocity_test and args.ground_remote_control:
        raise SystemExit("两种地面测试模式不能同时使用")
    if args.ground_zero_velocity_test:
        command = np.asarray(args.command, dtype=np.float32)
        if args.suspended_test or args.dry_run or not args.enable_hardware:
            raise SystemExit("--ground-zero-velocity-test 只允许独立用于真机")
        if not args.manage_sport_mode:
            raise SystemExit("地面试走必须添加 --manage-sport-mode")
        if args.command_source != "fixed":
            raise SystemExit("首次地面试走只允许 --command-source fixed")
        if args.steps <= 0 or args.steps > 500:
            raise SystemExit("地面试走要求 1<=--steps<=500")
        if not (0.0 <= command[0] <= 0.15) or np.max(np.abs(command[1:])) > 0:
            raise SystemExit("地面试走只允许 vx 在 [0,0.15]，vy=wz=0")
    if args.ground_remote_control:
        if args.suspended_test or args.dry_run or not args.enable_hardware:
            raise SystemExit("--ground-remote-control 只允许独立用于真机")
        if not args.manage_sport_mode:
            raise SystemExit("地面运动模式必须添加 --manage-sport-mode")
        if args.command_source != "remote":
            raise SystemExit("地面运动模式要求 --command-source remote")
        if args.steps != 0:
            raise SystemExit("地面运动模式要求 --steps 0 持续运行")
    if not (0.0 < args.remote_scale <= 1.0):
        raise SystemExit("--remote-scale 必须在 (0,1] 内")

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
    ground_policy_mode = args.ground_zero_velocity_test or args.ground_remote_control
    if ground_policy_mode:
        # 只绕过当前尚未完成的线速度输入；保留急停、姿态、关节、温度、
        # 电压、通信时效和动作限幅。实测完全坐下时 Kp=20 无法抬起后躯，
        # 因而一次接管内先以增强后腿增益起立，再直接进入短时策略测试。
        config.safety["require_base_lin_vel_valid"] = False
        ground_kp = [
            30.0 if name.startswith(("RL_", "RR_")) else 20.0
            for name in config.robot["joint_names"]
        ]
        config.transport["startup_kp"] = ground_kp
        config.transport["startup_kp_limit"] = 30.0
        config.transport["startup_transition_s"] = 10.0
        config.transport["startup_max_tracking_error"] = 1.25

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

    configured_action_clip_value = config.policy.get("action_clip")
    configured_action_clip = (
        None if configured_action_clip_value is None else float(configured_action_clip_value)
    )
    maximum_action_limit = float(
        config.safety.get(
            "max_deployment_action_limit",
            1.0 if configured_action_clip is None else configured_action_clip,
        )
    )
    deployment_action_limit = float(
        config.safety.get(
            "deployment_action_limit",
            1.0 if configured_action_clip is None else configured_action_clip,
        )
    )
    action_clip = deployment_action_limit if args.action_limit is None else float(args.action_limit)
    if not (0.0 < action_clip <= maximum_action_limit):
        raise SystemExit(
            f"--action-limit 必须在 (0, safety.max_deployment_action_limit={maximum_action_limit}] 内"
        )
    if configured_action_clip is not None and action_clip > configured_action_clip:
        raise SystemExit(
            f"--action-limit 不能超过训练动作裁剪值 {configured_action_clip}"
        )
    ground_action_limit = 1.5 if args.ground_remote_control else 1.0
    if ground_policy_mode and action_clip > ground_action_limit:
        raise SystemExit(
            f"当前地面运动模式的动作上限不得超过 {ground_action_limit}"
        )

    config.safety["runtime_action_limit"] = action_clip
    safety = SafetySupervisor(config.safety, default_pos)
    # 真机的逐帧遥控器状态仍由 SafetySupervisor 检查；这里仅打开软件总门。
    safety.set_enabled(True)

    use_remote_command = args.command_source == "remote" or (
        args.command_source == "auto" and not args.dry_run
    )
    max_command = float(config.safety.get("max_command_velocity", 1.0))
    if not use_remote_command and np.max(np.abs(np.asarray(args.command, dtype=np.float32))) > max_command:
        raise SystemExit(f"速度指令超过 safety.max_command_velocity={max_command}")
    if use_remote_command:
        remote_config = dict(config.transport.get("remote_control", {}))
        remote_config["scales"] = (
            np.asarray(remote_config.get("scales", [0.5, 0.3, 0.5]), dtype=np.float32)
            * args.remote_scale
        ).tolist()
        command_provider = RemoteVelocityCommand(remote_config)
    else:
        command_provider = None

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
                "foot_forces_estimated": None if state.foot_forces_estimated is None else state.foot_forces_estimated.tolist(),
                "foot_forces": None if state.foot_forces is None else state.foot_forces.tolist(),
                "contact_count": state.contact_count,
                "base_lin_vel_valid": state.base_lin_vel_valid,
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
            require_height_scan = bool(
                config.observation.get("require_height_scan", False)
            )
            transport = UnitreeSdkTransport(
                config=config.raw,
                require_auxiliary=not (
                    args.suspended_test
                    or (ground_policy_mode and not require_height_scan)
                ),
            )
        run_control_loop(
            transport=transport,
            policy=policy,
            observation_builder=builder,
            safety=safety,
            command=args.command,
            control_dt=float(config.robot["control_dt"]),
            steps=None if args.steps == 0 else args.steps,
            action_clip=action_clip,
            on_step=report,
            command_provider=command_provider,
            # 地面承重状态下强拉回默认关节角会造成倾倒和位置环震动；
            # close() 先发阻尼包，随后 finally 恢复原厂 Sport。
            finish_transition_on_normal_exit=not ground_policy_mode,
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
