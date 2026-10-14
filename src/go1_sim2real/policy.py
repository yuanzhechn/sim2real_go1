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
    def __init__(self, path: str | Path, device: str = "cpu", cpu_threads: int | None = None,
                 freeze_for_inference: bool = False) -> None:
        if cpu_threads is not None:
            if not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 64:
                raise ValueError("cpu_threads must be an integer in [1,64]")
            torch.set_num_threads(cpu_threads)
        self.device = torch.device(device)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        if freeze_for_inference:
            self.model = torch.jit.freeze(self.model)

    @torch.inference_mode()
    def __call__(self, observation: object) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.squeeze(0).cpu()


class NumpyElevationPolicy:
    """Float32 execution of the exact exported ELU elevation actor, no quantization."""
    def __init__(self, path: str | Path):
        import numpy as np
        model=torch.jit.load(str(path),map_location='cpu').eval()
        state=model.state_dict()
        indexes=(0,2,4,6)
        expected={f'mlp.{i}.{part}' for i in indexes for part in ('weight','bias')}
        if set(state)!=expected:
            raise ValueError('Only the verified unnormalized four-layer elevation actor is supported')
        shapes=((512,1072),(256,512),(128,256),(12,128))
        self.layers=[]
        for i,shape in zip(indexes,shapes):
            w=state[f'mlp.{i}.weight'].detach().numpy().copy()
            b=state[f'mlp.{i}.bias'].detach().numpy().copy()
            if w.shape!=shape or b.shape!=(shape[0],) or w.dtype!=np.float32 or b.dtype!=np.float32:
                raise ValueError('Elevation architecture or dtype mismatch')
            if not np.isfinite(w).all() or not np.isfinite(b).all():raise ValueError('Nonfinite weights')
            self.layers.append((w,b))

    def __call__(self, observation):
        import numpy as np
        h=np.asarray(observation,dtype=np.float32)
        if h.shape!=(1072,) or not np.isfinite(h).all():raise ValueError('Expected finite 1072-D observation')
        for i,(w,b) in enumerate(self.layers):
            h=w@h+b
            if i<3:
                negative=h<0
                h[negative]=np.expm1(h[negative])
        if not np.isfinite(h).all():raise ValueError('Nonfinite policy output')
        return h
