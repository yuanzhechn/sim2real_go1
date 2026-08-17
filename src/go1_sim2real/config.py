"""配置读取。YAML 只在运行时读取，不把硬件参数写死在代码中。"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    return RuntimeConfig(raw)
