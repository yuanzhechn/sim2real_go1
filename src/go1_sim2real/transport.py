"""机器人通信抽象。真实 SDK 接入必须显式实现并经过硬件测试。"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import struct
import sys
import time
from typing import Any, Protocol

import numpy as np

from .types import RobotState
from .action import Go1JointPositionMapper
from .sensors import AuxiliaryStateProvider, UdpJsonAuxiliaryStateProvider
from .velocity_estimator import Go1ContactVelocityEstimator


class RobotTransport(Protocol):
    def read_state(self) -> RobotState: ...
    def send_action(self, action: np.ndarray) -> None: ...
    def close(self) -> None: ...


@dataclass
class DryRunTransport:
    default_joint_pos: np.ndarray
    last_action: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.default_joint_pos = np.asarray(self.default_joint_pos, dtype=np.float32).reshape(12)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.sent_actions: list[np.ndarray] = []

    def read_state(self) -> RobotState:
        return RobotState(
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.zeros(3),
            projected_gravity=np.array([0.0, 0.0, -1.0]),
            joint_pos=self.default_joint_pos,
            joint_vel=np.zeros(12),
            last_action=self.last_action,
            # Dry-run cannot provide terrain perception. It is only for API tests.
            height_scan=np.zeros(187),
            timestamp=time.monotonic(),
        )

    def send_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).reshape(12).copy()
        self.sent_actions.append(self.last_action.copy())

    def close(self) -> None:
        return None

    def enter_safe_mode(self, reason: str) -> None:
        self.send_action(np.zeros(12, dtype=np.float32))


class JsonlTransport:
    """从一帧一行的 JSONL 状态日志回放，动作只写入内存不发到网络。"""

    def __init__(self, path: str, loop: bool = False) -> None:
        with open(path, "r", encoding="utf-8") as stream:
            self._frames = [json.loads(line) for line in stream if line.strip()]
        if not self._frames:
            raise ValueError(f"状态日志为空: {path}")
        self._index = 0
        self._loop = loop
        self.sent_actions: list[np.ndarray] = []

    def read_state(self) -> RobotState:
        frame = self._frames[self._index]
        if self._index + 1 < len(self._frames):
            self._index += 1
        elif self._loop:
            self._index = 0
        return RobotState(
            base_lin_vel=frame["base_lin_vel"],
            base_ang_vel=frame["base_ang_vel"],
            projected_gravity=frame["projected_gravity"],
            joint_pos=frame["joint_pos"],
            joint_vel=frame["joint_vel"],
            last_action=frame["last_action"],
            height_scan=frame["height_scan"],
            roll=float(frame.get("roll", 0.0)),
            pitch=float(frame.get("pitch", 0.0)),
            timestamp=float(frame.get("timestamp", time.monotonic())),
        )

    def send_action(self, action: np.ndarray) -> None:
        self.sent_actions.append(np.asarray(action, dtype=np.float32).reshape(12).copy())

    def close(self) -> None:
        return None

    def enter_safe_mode(self, reason: str) -> None:
        self.send_action(np.zeros(12, dtype=np.float32))


class UnitreeSdkTransport:
    """Unitree ``unitree_legged_sdk`` Go1 低层 UDP Python 适配器。

    Python 扩展模块在官方 SDK 中名为 ``robot_interface``。模块在构造时才导入，
    因而 dry-run/回放不依赖 SDK。测试可通过 ``sdk_module`` 和
    ``auxiliary_provider`` 注入无网络后端。
    """

    def __init__(
        self,
        config: dict[str, Any],
        sdk_module: Any | None = None,
        auxiliary_provider: AuxiliaryStateProvider | None = None,
        require_auxiliary: bool = True,
        allow_commands: bool = True,
        passive_only: bool = False,
    ) -> None:
        sdk_was_injected = sdk_module is not None
        robot = config["robot"]
        self.observation_config = config["observation"]
        self.config = config["transport"]
        self.safety_config = config["safety"]
        self.policy_joint_names = list(robot["joint_names"])
        self.sdk_joint_names = list(self.config["sdk_joint_names"])
        if sorted(self.policy_joint_names) != sorted(self.sdk_joint_names):
            raise ValueError("policy joint_names 与 transport.sdk_joint_names 必须是同一组关节")
        self._policy_to_sdk = np.asarray(
            [self.sdk_joint_names.index(name) for name in self.policy_joint_names], dtype=np.int64
        )
        self._directions = np.asarray(self.config["joint_directions"], dtype=np.float32).reshape(12)
        self._offsets = np.asarray(self.config["joint_offsets"], dtype=np.float32).reshape(12)
        self._kp = np.asarray(self.config["kp"], dtype=np.float32).reshape(12)
        self._kd = np.asarray(self.config["kd"], dtype=np.float32).reshape(12)
        self._torque_ff = np.asarray(self.config["torque_ff"], dtype=np.float32).reshape(12)
        self._joint_lower = np.asarray(self.config["joint_lower_limits"], dtype=np.float32).reshape(12)
        self._joint_upper = np.asarray(self.config["joint_upper_limits"], dtype=np.float32).reshape(12)
        numeric_arrays = (
            self._directions, self._offsets, self._kp, self._kd, self._torque_ff,
            self._joint_lower, self._joint_upper,
        )
        if not all(np.all(np.isfinite(values)) for values in numeric_arrays):
            raise ValueError("真机标定数组包含 NaN 或 Inf")
        if np.any(self._kp < 0) or np.any(self._kd < 0):
            raise ValueError("Kp/Kd 不能为负数")
        if np.any(self._joint_lower >= self._joint_upper):
            raise ValueError("关节下限必须小于上限")
        self._mapper = Go1JointPositionMapper(
            robot["default_joint_pos"], action_scale=float(robot["action_scale"])
        )
        self._default_joint_pos = np.asarray(robot["default_joint_pos"], dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._last_state_timestamp = 0.0
        self._last_tick: int | None = None
        self._closed = False
        self._have_state = False
        self._allow_commands = bool(allow_commands)
        self._passive_only = bool(passive_only)
        if self._passive_only and not self._allow_commands:
            raise ValueError("passive_only 需要 allow_commands=True 以建立 LowState 回传")
        self._bootstrap_sent = False
        self._enable_latched = False
        self._emergency_stop_latched = False
        self._previous_enable_pressed: bool | None = None
        self._velocity_estimator: Go1ContactVelocityEstimator | None = None

        if (
            self._allow_commands
            and not sdk_was_injected
            and bool(self.config.get("reject_sport_mode_process", True))
        ):
            self._assert_sport_mode_stopped()

        if sdk_module is None:
            module_path = str(self.config.get("sdk_python_path", "")).strip()
            if module_path:
                resolved = str(Path(module_path).expanduser().resolve())
                if resolved not in sys.path:
                    sys.path.insert(0, resolved)
            module_name = str(self.config.get("sdk_module", "robot_interface"))
            try:
                sdk_module = importlib.import_module(module_name)
            except ImportError as exc:
                raise RuntimeError(
                    f"无法导入 {module_name}；请按 unitree_legged_sdk 文档构建 Python wrapper，"
                    "并设置 transport.sdk_python_path"
                ) from exc
        self._sdk = sdk_module
        low_level = int(getattr(self._sdk, "LOWLEVEL", 0xFF))
        local_port = int(self.config.get("local_port", 8090))
        target_ip = str(self.config.get("target_ip", "192.168.123.10"))
        target_port = int(self.config.get("target_port", 8007))
        try:
            self._udp = self._sdk.UDP(low_level, local_port, target_ip, target_port)
        except TypeError as exc:
            raise RuntimeError(
                "robot_interface.UDP 构造签名不兼容；需要 Go1 unitree_legged_sdk v3.8.x Python wrapper"
            ) from exc
        disconnect_watchdog = getattr(self._udp, "SetDisconnectTime", None)
        if disconnect_watchdog is not None:
            disconnect_watchdog(
                float(robot["control_dt"]), float(self.config.get("watchdog_timeout_s", 0.10))
            )
        self._cmd = self._sdk.LowCmd()
        self._state = self._sdk.LowState()
        init_cmd = getattr(self._udp, "InitCmdData", None)
        if init_cmd is None:
            init_cmd = getattr(self._udp, "initCommunicationLowCmdData", None)
        if init_cmd is None:
            raise RuntimeError("SDK UDP 对象缺少低层命令初始化方法")
        init_cmd(self._cmd)
        self._sdk_safety = None
        if hasattr(self._sdk, "Safety") and hasattr(self._sdk, "LeggedType"):
            self._sdk_safety = self._sdk.Safety(self._sdk.LeggedType.Go1)

        if auxiliary_provider is None and require_auxiliary:
            auxiliary = self.config.get("auxiliary_state", {})
            auxiliary_mode = auxiliary.get("mode")
            if auxiliary_mode == "kinematic_contact":
                self._velocity_estimator = Go1ContactVelocityEstimator(
                    contact_threshold=float(auxiliary.get("contact_threshold", 5.0)),
                    min_contacts=int(auxiliary.get("min_contacts", 2)),
                    filter_alpha=float(auxiliary.get("filter_alpha", 0.2)),
                    max_speed=float(auxiliary.get("max_speed", 3.0)),
                    max_no_contact_frames=int(auxiliary.get("max_no_contact_frames", 5)),
                )
            elif auxiliary_mode != "udp_json":
                raise ValueError(
                    "策略必须提供 auxiliary_state.mode=udp_json 或 kinematic_contact"
                )
            else:
                auxiliary_provider = UdpJsonAuxiliaryStateProvider(
                    bind_host=str(auxiliary.get("bind_host", "127.0.0.1")),
                    bind_port=int(auxiliary["bind_port"]),
                    timeout_s=float(auxiliary.get("timeout_s", 0.10)),
                    require_height_scan=bool(
                        self.observation_config.get("require_height_scan", True)
                    ),
                )
        self._auxiliary_provider = auxiliary_provider

    @staticmethod
    def _assert_sport_mode_stopped() -> None:
        for process_dir in Path("/proc").glob("[0-9]*"):
            try:
                command = process_dir.joinpath("cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if any(Path(value.decode("utf-8", errors="ignore")).name == "Legged_sport" for value in command):
                raise RuntimeError(
                    "检测到 Legged_sport 正在运行；禁止与低层电机控制并发。"
                    "请按 Unitree 流程安全停止运动服务后再启动真机模式"
                )

    @staticmethod
    def _projected_gravity(quaternion_wxyz: object) -> np.ndarray:
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-6 or not np.isfinite(norm):
            raise RuntimeError("IMU 四元数无效")
        w, x, y, z = quaternion / norm
        return np.asarray(
            (
                -2.0 * (x * z - w * y),
                -2.0 * (y * z + w * x),
                -(1.0 - 2.0 * (x * x + y * y)),
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _remote_buttons(remote: object) -> int:
        raw = bytes(int(value) & 0xFF for value in remote)
        if len(raw) < 4:
            return 0
        return raw[2] | (raw[3] << 8)

    def _button_pressed(self, button_name: str) -> bool:
        bits = {
            "R1": 0, "L1": 1, "start": 2, "select": 3,
            "R2": 4, "L2": 5, "F1": 6, "F2": 7,
            "A": 8, "B": 9, "X": 10, "Y": 11,
            "up": 12, "right": 13, "down": 14, "left": 15,
        }
        if button_name not in bits:
            raise ValueError(f"未知遥控器按键: {button_name}")
        return bool(self._remote_buttons(self._state.wirelessRemote) & (1 << bits[button_name]))

    def _remote_axes(self) -> np.ndarray:
        raw = bytes(int(value) & 0xFF for value in self._state.wirelessRemote)
        if len(raw) < 24:
            raise RuntimeError("LowState.wirelessRemote 少于 24 字节")
        # xRockerBtnDataStruct: head[2], buttons(uint16), lx, rx, ry, analog_L2, ly.
        lx, rx, ry, _analog_l2, ly = struct.unpack_from("<fffff", raw, 4)
        axes = np.asarray([lx, ly, rx, ry], dtype=np.float32)
        if not np.all(np.isfinite(axes)) or np.max(np.abs(axes)) > 1.5:
            raise RuntimeError(f"遥控摇杆数据无效: {axes.tolist()}")
        return np.clip(axes, -1.0, 1.0)

    def _remote_safety_state(self, remote_axes: np.ndarray) -> tuple[bool, bool]:
        enable_pressed = self._button_pressed(str(self.config.get("enable_button", "L2")))
        emergency_pressed = self._button_pressed(
            str(self.config.get("emergency_stop_button", "B"))
        )
        if emergency_pressed:
            self._emergency_stop_latched = True
            self._enable_latched = False

        switch_mode = str(self.config.get("enable_switch_mode", "toggle"))
        if switch_mode == "hold":
            enabled = enable_pressed and not self._emergency_stop_latched
        elif switch_mode == "program":
            # 程序接管期间无需功能键；但必须先观察到摇杆连续回中，防止带着
            # 非零速度指令启动。B 急停一旦锁存，只能重启程序清除。
            deadband = float(self.config.get("remote_control", {}).get("deadband", 0.08))
            centered = bool(np.max(np.abs(remote_axes[[0, 1, 2]])) <= deadband)
            if not hasattr(self, "_remote_centered_frames"):
                self._remote_centered_frames = 0
            self._remote_centered_frames = self._remote_centered_frames + 1 if centered else 0
            required = max(1, int(self.config.get("remote_startup_center_frames", 10)))
            if self._remote_centered_frames >= required and not self._emergency_stop_latched:
                self._enable_latched = True
            enabled = self._enable_latched and not self._emergency_stop_latched
        elif switch_mode == "toggle":
            # 首帧只记录按键状态，避免程序启动时已经压住 L2 导致意外使能。
            if self._previous_enable_pressed is not None:
                rising_edge = enable_pressed and not self._previous_enable_pressed
                if rising_edge and not self._emergency_stop_latched:
                    self._enable_latched = not self._enable_latched
            enabled = self._enable_latched and not self._emergency_stop_latched
        else:
            raise ValueError(f"未知 enable_switch_mode: {switch_mode}")
        self._previous_enable_pressed = enable_pressed
        return enabled, self._emergency_stop_latched

    def _sdk_to_policy(self, values: object) -> np.ndarray:
        sdk_values = np.asarray(values, dtype=np.float32).reshape(12)
        return sdk_values[self._policy_to_sdk] * self._directions + self._offsets

    def _policy_to_sdk_values(self, values: object) -> np.ndarray:
        policy_values = np.asarray(values, dtype=np.float32).reshape(12)
        sdk_values = np.empty(12, dtype=np.float32)
        sdk_values[self._policy_to_sdk] = (policy_values - self._offsets) * self._directions
        return sdk_values

    def _policy_vector_to_sdk(self, values: object, signed: bool = False) -> np.ndarray:
        policy_values = np.asarray(values, dtype=np.float32).reshape(12)
        sdk_values = np.empty(12, dtype=np.float32)
        if signed:
            policy_values = policy_values * self._directions
        sdk_values[self._policy_to_sdk] = policy_values
        return sdk_values

    def read_state(self) -> RobotState:
        if self._closed:
            raise RuntimeError("transport 已关闭")
        receive_deadline = time.monotonic() + float(self.config.get("state_timeout_s", 0.10))
        recv_result = self._udp.Recv()
        while isinstance(recv_result, int) and recv_result < 0:
            if self._allow_commands:
                self._send_passive_bootstrap()
            if time.monotonic() >= receive_deadline:
                mode = "已发送被动初始化包" if self._allow_commands else "只读模式未发送初始化包"
                raise ConnectionError(f"Unitree UDP Recv 超时 ({mode}): {recv_result}")
            time.sleep(float(self.config.get("bootstrap_interval_s", 0.002)))
            recv_result = self._udp.Recv()
        self._udp.GetRecv(self._state)
        now = time.monotonic()
        motor_states = list(self._state.motorState)[:12]
        if len(motor_states) != 12:
            raise RuntimeError("LowState.motorState 少于 12 个电机")
        imu = self._state.imu
        quaternion = np.asarray(imu.quaternion, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) < 0.5:
            raise ConnectionError("尚未收到有效的 Unitree LowState")
        tick = int(getattr(self._state, "tick", 0))
        if self._last_tick is None or tick != self._last_tick:
            self._last_tick = tick
            self._last_state_timestamp = now
            self._have_state = True

        sdk_pos = [float(motor.q) for motor in motor_states]
        sdk_vel = [float(motor.dq) for motor in motor_states]
        positions = self._sdk_to_policy(sdk_pos)
        velocities = self._sdk_to_policy(sdk_vel) - self._offsets
        rpy = np.asarray(imu.rpy, dtype=np.float32).reshape(3)
        rpy_signs = np.asarray(self.config.get("rpy_signs", [1, 1, 1]), dtype=np.float32).reshape(3)
        angular_velocity = np.asarray(imu.gyroscope, dtype=np.float32).reshape(3)
        angular_velocity *= np.asarray(
            self.config.get("angular_velocity_signs", [1, 1, 1]), dtype=np.float32
        ).reshape(3)
        force_sdk = np.asarray(
            getattr(self._state, "footForce", (0, 0, 0, 0)), dtype=np.float32
        )
        force_est_sdk = np.asarray(
            getattr(self._state, "footForceEst", (0, 0, 0, 0)), dtype=np.float32
        )
        # Unitree: FR,FL,RR,RL；policy: FL,FR,RL,RR。
        force_order = np.asarray((1, 0, 3, 2), dtype=np.int64)
        foot_forces = force_sdk[force_order]
        foot_forces_estimated = force_est_sdk[force_order]
        base_lin_vel_valid = True
        contact_count = None
        if self._velocity_estimator is not None:
            base_lin_vel = self._velocity_estimator.update(
                positions, velocities, angular_velocity, foot_forces_estimated
            )
            base_lin_vel_valid = self._velocity_estimator.valid
            contact_count = self._velocity_estimator.contact_count
            from .sensors import AuxiliaryState

            auxiliary = AuxiliaryState(base_lin_vel, np.zeros(187), now)
        elif self._auxiliary_provider is None:
            from .sensors import AuxiliaryState

            auxiliary = AuxiliaryState(np.zeros(3), np.zeros(187), now)
        else:
            auxiliary = self._auxiliary_provider.read()
        cell_voltage = np.asarray(getattr(self._state.bms, "cell_vol", ()), dtype=np.float32)
        battery_voltage = float(np.sum(cell_voltage) / 1000.0) if np.any(cell_voltage > 0) else None
        remote_axes = self._remote_axes()
        enable_switch, emergency_stop = self._remote_safety_state(remote_axes)
        result = RobotState(
            base_lin_vel=auxiliary.base_lin_vel,
            base_ang_vel=angular_velocity,
            projected_gravity=self._projected_gravity(quaternion),
            joint_pos=positions,
            joint_vel=velocities,
            last_action=self._last_action,
            height_scan=auxiliary.height_scan,
            roll=float(rpy[0] * rpy_signs[0]),
            pitch=float(rpy[1] * rpy_signs[1]),
            timestamp=self._last_state_timestamp,
            auxiliary_timestamp=auxiliary.timestamp,
            motor_temperatures=[float(motor.temperature) for motor in motor_states],
            battery_voltage=battery_voltage,
            motor_modes=[float(motor.mode) for motor in motor_states],
            enable_switch=enable_switch,
            emergency_stop=emergency_stop,
            communication_ok=self._have_state,
            remote_axes=remote_axes,
            foot_forces=foot_forces,
            foot_forces_estimated=foot_forces_estimated,
            contact_count=contact_count,
            base_lin_vel_valid=base_lin_vel_valid,
        )
        return result

    def _send_passive_bootstrap(self) -> None:
        """用零力矩、失能电机包建立 LowState 回传通道。

        Unitree 低层 UDP 是请求/响应式接口，新本地端口在首次 Send 前不会收到
        LowState。此操作只允许在显式命令模式且 sport mode 已被启动门阻止后发生。
        """

        if not self._allow_commands:
            return
        for index in range(12):
            motor = self._cmd.motorCmd[index]
            motor.mode = 0x00
            motor.q = 2.146e9
            motor.dq = 16000.0
            motor.Kp = 0.0
            motor.Kd = 0.0
            motor.tau = 0.0
        self._udp.SetSend(self._cmd)
        send_result = self._udp.Send()
        if isinstance(send_result, int) and send_result < 0:
            raise ConnectionError(f"Unitree UDP 被动初始化发送失败: {send_result}")
        self._bootstrap_sent = True

    def _apply_sdk_safety(self) -> None:
        if self._sdk_safety is None:
            return
        self._sdk_safety.PositionLimit(self._cmd)
        power_factor = int(self.config.get("power_protect_level", 1))
        result = self._sdk_safety.PowerProtect(self._cmd, self._state, power_factor)
        if isinstance(result, int) and result < 0:
            raise RuntimeError(f"Unitree SDK PowerProtect 拒绝命令: {result}")

    def _send_joint_target(
        self,
        target: np.ndarray,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        torque_ff: np.ndarray | None = None,
    ) -> None:
        target = np.asarray(target, dtype=np.float32).reshape(12)
        if not np.all(np.isfinite(target)):
            raise ValueError("关节目标包含 NaN 或 Inf")
        target = np.clip(target, self._joint_lower, self._joint_upper)
        kp = self._kp if kp is None else np.asarray(kp, dtype=np.float32).reshape(12)
        kd = self._kd if kd is None else np.asarray(kd, dtype=np.float32).reshape(12)
        torque_ff = (
            self._torque_ff
            if torque_ff is None
            else np.asarray(torque_ff, dtype=np.float32).reshape(12)
        )
        sdk_target = self._policy_to_sdk_values(target)
        sdk_kp = self._policy_vector_to_sdk(kp)
        sdk_kd = self._policy_vector_to_sdk(kd)
        sdk_tau = self._policy_vector_to_sdk(torque_ff, signed=True)
        for index in range(12):
            motor = self._cmd.motorCmd[index]
            motor.mode = 0x0A
            motor.q = float(sdk_target[index])
            motor.dq = 0.0
            motor.Kp = float(sdk_kp[index])
            motor.Kd = float(sdk_kd[index])
            motor.tau = float(sdk_tau[index])
        self._apply_sdk_safety()
        self._udp.SetSend(self._cmd)
        send_result = self._udp.Send()
        if isinstance(send_result, int) and send_result < 0:
            raise ConnectionError(f"Unitree UDP Send 失败: {send_result}")

    def prepare_for_policy(
        self,
        duration: float | None = None,
        label: str = "策略启动过渡",
    ) -> None:
        """吊挂/站立启动时，从当前关节角平滑移动到策略默认姿态。"""

        if not self._allow_commands or self._passive_only:
            raise PermissionError("当前 transport 不允许策略启动过渡")
        duration = float(
            self.config.get("startup_transition_s", 5.0)
            if duration is None
            else duration
        )
        if duration < 3.0:
            raise ValueError("transport.startup_transition_s 必须至少为 3 秒")
        dt = float(self.config.get("startup_control_dt", 0.02))
        kp_config = self.config.get("startup_kp", 10.0)
        kd_config = self.config.get("startup_kd", 1.0)
        kp = np.broadcast_to(np.asarray(kp_config, dtype=np.float32), (12,)).copy()
        kd = np.broadcast_to(np.asarray(kd_config, dtype=np.float32), (12,)).copy()
        kp_limit = float(self.config.get("startup_kp_limit", 10.0))
        if not (
            np.all(np.isfinite(kp))
            and np.all(np.isfinite(kd))
            and np.all(kp > 0.0)
            and np.all(kp <= kp_limit)
            and kp_limit <= 40.0
            and np.all(kd > 0.0)
            and np.all(kd <= 2.0)
        ):
            raise ValueError("启动过渡要求每关节 0<Kp<=startup_kp_limit<=40 且 0<Kd<=2")
        zero_tau = np.zeros(12, dtype=np.float32)
        deadband = float(self.config.get("remote_control", {}).get("deadband", 0.08))

        initial_state = self.read_state()
        initial = initial_state.joint_pos.copy()
        steps = max(1, int(round(duration / dt)))
        print(
            f"{label}: duration={duration:.1f}s "
            f"Kp={float(np.min(kp)):.1f}..{float(np.max(kp)):.1f} "
            f"Kd={float(np.min(kd)):.1f}..{float(np.max(kd)):.1f}",
            flush=True,
        )
        print("initial_q=" + np.array2string(initial, precision=3), flush=True)
        print(
            "default_q=" + np.array2string(self._default_joint_pos, precision=3),
            flush=True,
        )
        for step in range(steps):
            state = self.read_state()
            if state.emergency_stop:
                raise RuntimeError("B 急停触发")
            if state.remote_axes is None or np.max(np.abs(state.remote_axes[[0, 1, 2]])) > deadband:
                raise RuntimeError("启动过渡期间遥控摇杆必须回中")
            if state.battery_voltage is not None and state.battery_voltage < float(
                self.safety_config.get("min_battery_voltage_v", 19.0)
            ):
                raise RuntimeError(f"电池电压过低: {state.battery_voltage}")
            if state.motor_temperatures is not None and np.max(state.motor_temperatures) > float(
                self.safety_config.get("max_motor_temperature_c", 70.0)
            ):
                raise RuntimeError("电机温度过高")
            max_dq = float(np.max(np.abs(state.joint_vel)))
            if max_dq > min(6.0, float(self.safety_config.get("max_joint_velocity_rad_s", 25.0))):
                raise RuntimeError(f"启动过渡关节速度过高: {max_dq:.3f} rad/s")
            ratio = (step + 1) / float(steps)
            smooth = ratio * ratio * (3.0 - 2.0 * ratio)
            target = initial * (1.0 - smooth) + self._default_joint_pos * smooth
            tracking_error = float(np.max(np.abs(state.joint_pos - target)))
            if step > 10 and tracking_error > float(
                self.config.get("startup_max_tracking_error", 0.45)
            ):
                errors = state.joint_pos - target
                worst = int(np.argmax(np.abs(errors)))
                raise RuntimeError(
                    f"启动过渡关节未跟随目标: error={tracking_error:.3f} rad "
                    f"joint={self.policy_joint_names[worst]} "
                    f"q={state.joint_pos[worst]:.3f} target={target[worst]:.3f}"
                )
            self._send_joint_target(target, kp=kp, kd=kd, torque_ff=zero_tau)
            if step % 50 == 0 or step == steps - 1:
                print(
                    f"startup_step={step:04d} alpha={smooth:.3f} "
                    f"dq_max={max_dq:.3f} tracking_error={tracking_error:.3f} "
                    f"contacts={state.contact_count} "
                    f"base_lin_vel={np.array2string(state.base_lin_vel, precision=3)}",
                    flush=True,
                )
            time.sleep(dt)
        self._last_action.fill(0.0)

    def finish_policy(self) -> None:
        """正常结束策略时低增益回到默认姿态，减少与 Sport 模式交接的冲击。"""

        self.prepare_for_policy(
            duration=float(self.config.get("shutdown_transition_s", 3.0)),
            label="策略退出过渡",
        )

    def send_action(self, action: np.ndarray) -> None:
        if not self._allow_commands:
            raise PermissionError("此 transport 以只读模式打开，拒绝发送命令")
        if self._passive_only:
            raise PermissionError("此 transport 仅允许电机失能的被动初始化包")
        if self._closed:
            raise RuntimeError("transport 已关闭")
        if not self._have_state:
            raise RuntimeError("尚未收到有效 LowState，拒绝发送电机命令")
        normalized = np.asarray(action, dtype=np.float32).reshape(12)
        if not np.all(np.isfinite(normalized)):
            raise ValueError("动作包含 NaN 或 Inf")
        target = self._mapper.to_joint_target(normalized)
        self._send_joint_target(target)
        self._last_action = normalized.copy()

    def _send_damping(self) -> None:
        if not self._have_state or not self._allow_commands:
            return
        damping_kd = float(self.config.get("fault_damping_kd", 1.0))
        for index in range(12):
            motor = self._cmd.motorCmd[index]
            motor.mode = 0x0A
            motor.q = 2.146e9
            motor.dq = 0.0
            motor.Kp = 0.0
            motor.Kd = damping_kd
            motor.tau = 0.0
        self._udp.SetSend(self._cmd)
        self._udp.Send()

    def enter_safe_mode(self, reason: str) -> None:
        # 通信、急停、过热和姿态故障不应继续以位置环强制站立。
        force_damping = {
            "emergency_stop", "communication_error", "state_timeout",
            "auxiliary_state_timeout", "motor_temperature_limit", "motor_overheat_fault",
            "enable_switch_off", "roll_limit", "pitch_limit", "joint_error_limit",
            "joint_velocity_limit", "motor_mode_fault",
            "base_lin_vel_invalid",
        }
        if bool(self.safety_config.get("stand_on_fault", True)) and reason not in force_damping:
            self.send_action(np.zeros(12, dtype=np.float32))
        else:
            self._send_damping()
        self._last_action.fill(0.0)

    def close(self) -> None:
        if self._closed:
            return
        try:
            repeat = max(1, int(self.config.get("shutdown_damping_packets", 5)))
            for _ in range(repeat):
                if self._passive_only:
                    self._send_passive_bootstrap()
                else:
                    self._send_damping()
        finally:
            self._closed = True
            if self._auxiliary_provider is not None:
                self._auxiliary_provider.close()
