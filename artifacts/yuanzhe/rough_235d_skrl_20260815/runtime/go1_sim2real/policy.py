from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn


class MeanPolicy(nn.Module):
    def __init__(self, observation_dim: int = 235, action_dim: int = 12, hidden_layers: Sequence[int] = (512, 256, 128)):
        super().__init__()
        layers = []
        input_dim = observation_dim
        for width in hidden_layers:
            layers += [nn.Linear(input_dim, int(width)), nn.ELU()]
            input_dim = int(width)
        self.net_container = nn.Sequential(*layers)
        self.policy_layer = nn.Linear(input_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        return self.policy_layer(self.net_container(observation))


def export_skrl_checkpoint(checkpoint: str | Path, output: str | Path, observation_dim: int = 235, action_dim: int = 12, hidden_layers: Sequence[int] = (512, 256, 128)) -> None:
    model = MeanPolicy(observation_dim, action_dim, hidden_layers)
    data = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = {k: v for k, v in data["policy"].items() if k.startswith("net_container.") or k.startswith("policy_layer.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint 网络不匹配：missing={missing}, unexpected={unexpected}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model.eval()).save(str(destination))


class TorchScriptPolicy:
    def __init__(self, path: str | Path):
        self.model = torch.jit.load(str(path), map_location="cpu").eval()

    @torch.inference_mode()
    def __call__(self, observation: object) -> torch.Tensor:
        value = self.model(torch.as_tensor(observation, dtype=torch.float32))
        return value.squeeze(0).cpu()
