#!/usr/bin/env python3
"""从当前项目的 skrl PPO checkpoint 导出确定性 TorchScript policy。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go1_sim2real.policy import SkrlPPOPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--obs-dim", type=int, default=235)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    policy = SkrlPPOPolicy.from_checkpoint(
        args.checkpoint,
        observation_dim=args.obs_dim,
        action_dim=args.action_dim,
        device=args.device,
    )
    policy.export(args.output)
    print(f"已导出确定性 TorchScript 策略: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
