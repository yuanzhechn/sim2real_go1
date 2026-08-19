"""配置读取。YAML 只在运行时读取，不把硬件参数写死在代码中。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RuntimeConfig:
    raw: dict[str, Any]

    @property
    def robot(self) -> dict[str, Any]:
        return self.raw["robot"]

    @property
    def observation(self) -> dict[str, Any]:
        return self.raw["observation"]

    @property
    def policy(self) -> dict[str, Any]:
        return self.raw["policy"]

    @property
    def safety(self) -> dict[str, Any]:
        return self.raw["safety"]

    @property
    def transport(self) -> dict[str, Any]:
        return self.raw["transport"]


def load_config(path: str | Path) -> RuntimeConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("缺少 PyYAML，请在 Isaac Lab Python 中安装本项目") from exc
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件不是 YAML 对象: {config_path}")
    config = RuntimeConfig(raw)
    _validate_config(config)
    return config


def _validate_config(config: RuntimeConfig) -> None:
    required = {"robot", "observation", "policy", "safety", "transport"}
    missing = required.difference(config.raw)
    if missing:
        raise ValueError(f"配置缺少顶层字段: {sorted(missing)}")
    if int(config.robot.get("joint_count", 0)) != 12:
        raise ValueError("Go1 joint_count 必须为 12")
    if len(config.robot.get("joint_names", ())) != 12:
        raise ValueError("robot.joint_names 必须包含 12 项")
    if len(config.robot.get("default_joint_pos", ())) != 12:
        raise ValueError("robot.default_joint_pos 必须包含 12 项")
    if float(config.robot.get("control_dt", 0.0)) <= 0:
        raise ValueError("robot.control_dt 必须大于 0")
    dimensions = config.observation.get("dimensions", ())
    history_length = int(config.observation.get("history_length", 1))
    if history_length < 1:
        raise ValueError("observation.history_length 必须大于 0")
    if sum(int(value) for value in dimensions) * history_length != int(
        config.policy.get("observation_dim", -1)
    ):
        raise ValueError("observation.dimensions/history_length 与 policy.observation_dim 不一致")
    if int(config.policy.get("action_dim", 0)) != 12:
        raise ValueError("policy.action_dim 必须为 12")


def validate_hardware_config(config: RuntimeConfig) -> None:
    """拒绝未标定或会改变 Rough 策略输入分布的真机配置。"""

    transport = config.transport
    if transport.get("mode") != "unitree_sdk":
        raise ValueError("transport.mode 必须为 unitree_sdk")
    if not bool(transport.get("hardware_validated", False)):
        raise ValueError("transport.hardware_validated 尚未设为 true；必须先完成吊挂标定")
    if not str(transport.get("calibration_id", "")).strip():
        raise ValueError("真机配置必须填写 calibration_id 以记录本次标定")
    required_arrays = {
        "sdk_joint_names": 12,
        "joint_directions": 12,
        "joint_offsets": 12,
        "kp": 12,
        "kd": 12,
        "torque_ff": 12,
        "joint_lower_limits": 12,
        "joint_upper_limits": 12,
    }
    for key, length in required_arrays.items():
        if len(transport.get(key, ())) != length:
            raise ValueError(f"transport.{key} 必须包含 {length} 项")
    if any(float(value) not in (-1.0, 1.0) for value in transport["joint_directions"]):
        raise ValueError("transport.joint_directions 只能为 -1 或 1")
    if bool(config.observation.get("require_height_scan", True)):
        auxiliary = transport.get("auxiliary_state", {})
        if auxiliary.get("mode") != "udp_json":
            raise ValueError("Rough 策略要求真实 height_scan；必须配置 auxiliary_state.mode=udp_json")
