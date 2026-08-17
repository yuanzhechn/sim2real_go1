"""策略加载、skrl checkpoint 转换和确定性动作推理。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn


class MeanPolicyMLP(nn.Module):
    """与当前 skrl GaussianMixin policy 网络对应的确定性策略。"""

    def __init__(self, observation_dim: int, action_dim: int, hidden_layers: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = observation_dim
        for hidden_dim in hidden_layers:
            layers.extend((nn.Linear(input_dim, int(hidden_dim)), nn.ELU()))
            input_dim = int(hidden_dim)
        self.net_container = nn.Sequential(*layers)
        self.policy_layer = nn.Linear(input_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        return self.policy_layer(self.net_container(observation))


class SkrlPPOPolicy:
    """从 skrl 的 agent checkpoint 提取 policy 均值网络。"""

    def __init__(self, model: MeanPolicyMLP, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        observation_dim: int = 235,
        action_dim: int = 12,
        hidden_layers: Sequence[int] = (512, 256, 128),
        device: str = "cpu",
    ) -> "SkrlPPOPolicy":
        model = MeanPolicyMLP(observation_dim, action_dim, hidden_layers)
        checkpoint_data = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if "policy" not in checkpoint_data:
            raise ValueError("不是 skrl agent checkpoint：缺少 policy 权重")
        policy_state = {
            key: value
            for key, value in checkpoint_data["policy"].items()
            if key.startswith("net_container.") or key.startswith("policy_layer.")
        }
        missing, unexpected = model.load_state_dict(policy_state, strict=False)
        if missing or unexpected:
            raise ValueError(f"policy 网络结构不匹配，缺少={missing}，多余={unexpected}")
        return cls(model, device=device)

    @torch.inference_mode()
    def __call__(self, observation: object) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return self.model(tensor).squeeze(0).cpu()

    def export(self, output: str | Path) -> None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scripted = torch.jit.script(self.model.cpu())
        scripted.save(str(output_path))


class TorchScriptPolicy:
    def __init__(self, path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()

    @torch.inference_mode()
    def __call__(self, observation: object) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.squeeze(0).cpu()
