#!/usr/bin/env python3
"""将 skrl PPO checkpoint 导出为独立 TorchScript policy。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from go1_sim2real.policy import export_skrl_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--obs-dim", type=int, default=48)
    parser.add_argument("--action-dim", type=int, default=12)
    args = parser.parse_args()
    export_skrl_checkpoint(args.checkpoint, args.output, args.obs_dim, args.action_dim)
    print(f"已导出确定性 TorchScript 策略: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
