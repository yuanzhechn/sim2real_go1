"""训练侧导出 bundle 的完整性与部署规格校验。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import RuntimeConfig


@dataclass(frozen=True)
class Sim2RealBundle:
    root: Path
    manifest: dict[str, Any]
    training_config: dict[str, Any]
    policy_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 PyYAML，无法读取 bundle 配置") from exc
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"bundle 配置不是 YAML 对象: {path}")
    return value


def load_bundle(path: str | Path) -> Sim2RealBundle:
    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bundle 缺少 manifest.json: {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("bundle manifest 必须是 JSON 对象")
    config_relative = manifest.get("config")
    if config_relative is None:
        candidates = sorted((root / "config").glob("*.yaml"))
        if len(candidates) != 1:
            raise ValueError("bundle manifest.config 缺失，且 config 目录不是唯一 YAML")
        config_path = candidates[0].resolve()
    else:
        config_path = (root / str(config_relative)).resolve()
    try:
        config_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("bundle manifest.config 不能指向 bundle 目录外") from exc
    if not config_path.is_file():
        raise FileNotFoundError(f"bundle 缺少训练规格配置: {config_path}")
    policy_relative = manifest.get("policy")
    if not isinstance(policy_relative, str) or not policy_relative:
        raise ValueError("bundle manifest.policy 无效")
    policy_path = (root / policy_relative).resolve()
    try:
        policy_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("bundle manifest.policy 不能指向 bundle 目录外") from exc
    if not policy_path.is_file():
        raise FileNotFoundError(f"bundle 策略不存在: {policy_path}")

    expected_sha = str(manifest.get("policy_sha256", "")).lower()
    actual_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if expected_sha != actual_sha:
        raise ValueError(
            f"bundle 策略 SHA256 不匹配: manifest={expected_sha or '<empty>'}, actual={actual_sha}"
        )

    training_config = _load_yaml(config_path)
    _validate_internal_spec(manifest, training_config)
    return Sim2RealBundle(root, manifest, training_config, policy_path)


def _validate_internal_spec(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    try:
        robot = config["robot"]
        observation = config["observation"]
        policy = config["policy"]
        dimensions = [int(value) for value in observation["dimensions"]]
        terms = observation.get("terms")
        if terms is None and dimensions == [3, 3, 3, 3, 12, 12, 12, 187]:
            # 兼容第一版 Rough bundle；新 bundle 必须显式导出 terms。
            terms = [
                "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                "joint_pos", "joint_vel", "actions", "height_scan",
            ]
            observation["terms"] = terms
        terms = list(terms)
        history_length = int(observation.get("history_length", 1))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("bundle 训练配置缺少有效的 robot/observation/policy 规格") from exc

    checks = {
        "observation_dim": int(policy["observation_dim"]),
        "action_dim": int(policy["action_dim"]),
        "hidden_layers": [int(value) for value in policy["hidden_layers"]],
        "action_scale": float(robot["action_scale"]),
        "height_scan_dim": dimensions[terms.index("height_scan")] if "height_scan" in terms else 0,
    }
    if sum(dimensions) * history_length != checks["observation_dim"]:
        raise ValueError("bundle observation dimensions/history 与 policy.observation_dim 不一致")
    for key, config_value in checks.items():
        manifest_value = manifest.get(key)
        if manifest_value != config_value:
            raise ValueError(
                f"bundle manifest.{key}={manifest_value!r} 与训练配置 {config_value!r} 不一致"
            )


def validate_bundle_runtime_config(bundle: Sim2RealBundle, runtime: RuntimeConfig) -> None:
    """确保硬件运行配置没有改变模型训练时的输入输出语义。"""

    source = bundle.training_config
    comparisons = [
        ("robot.control_dt", source["robot"]["control_dt"], runtime.robot["control_dt"]),
        ("robot.joint_count", source["robot"]["joint_count"], runtime.robot["joint_count"]),
        ("robot.joint_names", source["robot"]["joint_names"], runtime.robot["joint_names"]),
        (
            "robot.default_joint_pos",
            source["robot"]["default_joint_pos"],
            runtime.robot["default_joint_pos"],
        ),
        ("robot.action_scale", source["robot"]["action_scale"], runtime.robot["action_scale"]),
        (
            "observation.dimensions",
            source["observation"]["dimensions"],
            runtime.observation["dimensions"],
        ),
        ("observation.terms", source["observation"]["terms"], runtime.observation["terms"]),
        (
            "observation.history_length",
            source["observation"]["history_length"],
            runtime.observation["history_length"],
        ),
        ("policy.observation_dim", source["policy"]["observation_dim"], runtime.policy["observation_dim"]),
        ("policy.action_dim", source["policy"]["action_dim"], runtime.policy["action_dim"]),
        ("policy.hidden_layers", source["policy"]["hidden_layers"], runtime.policy["hidden_layers"]),
        ("policy.action_clip", source["policy"]["action_clip"], runtime.policy["action_clip"]),
    ]
    if "height_scan" in source["observation"]["terms"]:
        comparisons.append(
            (
                "observation.height_scan_clip",
                source["observation"]["height_scan_clip"],
                runtime.observation.get("height_scan_clip"),
            )
        )
    for name, trained, deployed in comparisons:
        if isinstance(trained, (list, tuple)) or isinstance(deployed, (list, tuple)):
            matches = np.array_equal(np.asarray(trained), np.asarray(deployed))
        elif isinstance(trained, (float, int)) and isinstance(deployed, (float, int)):
            matches = bool(np.isclose(float(trained), float(deployed), rtol=0.0, atol=1e-8))
        else:
            matches = trained == deployed
        if not matches:
            raise ValueError(f"bundle 规格不匹配: {name} 训练值={trained!r}, 运行值={deployed!r}")
