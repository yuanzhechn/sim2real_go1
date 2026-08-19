"""安全、可测试的 Unitree Go1 sim2real 运行层。"""

from .config import RuntimeConfig, load_config, validate_hardware_config
from .action import Go1JointPositionMapper
from .observation import Go1ObservationBuilder
from .safety import SafetySupervisor
from .types import RobotState
from .sensors import AuxiliaryState, UdpJsonAuxiliaryStateProvider


def __getattr__(name: str):
    # 状态只读/标定工具不应仅因尚未安装 PyTorch 而无法启动。
    if name in {"SkrlPPOPolicy", "TorchScriptPolicy"}:
        from .policy import SkrlPPOPolicy, TorchScriptPolicy

        return {"SkrlPPOPolicy": SkrlPPOPolicy, "TorchScriptPolicy": TorchScriptPolicy}[name]
    raise AttributeError(name)

__all__ = [
    "Go1ObservationBuilder",
    "Go1JointPositionMapper",
    "RobotState",
    "RuntimeConfig",
    "SafetySupervisor",
    "SkrlPPOPolicy",
    "TorchScriptPolicy",
    "AuxiliaryState",
    "UdpJsonAuxiliaryStateProvider",
    "load_config",
    "validate_hardware_config",
]
