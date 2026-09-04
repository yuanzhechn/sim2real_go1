#!/usr/bin/env python3
"""当前仓库自带的 Go1 sim2real dry-run 入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import yaml
except ImportError as exc:
    raise SystemExit("缺少 PyYAML，请安装 PyYAML") from exc

from go1_sim2real.observation import ObservationBuilder
from go1_sim2real.policy import TorchScriptPolicy
from go1_sim2real.runtime import run_loop
from go1_sim2real.safety import SafetySupervisor
from go1_sim2real.transport import DryRunTransport, UnitreeSdkTransport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--config", default=str(ROOT / "config/go1_rough_sim2real.yaml"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--command", nargs=3, type=float, default=[0.2, 0.0, 0.0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not args.dry_run:
        raise SystemExit("真机 transport 尚未绑定；请先使用 --dry-run")
    default_pos = np.asarray(config["robot"]["default_joint_pos"], dtype=np.float32)
    transport = DryRunTransport(default_pos) if args.dry_run else UnitreeSdkTransport()
    policy = TorchScriptPolicy(args.policy)
    builder = ObservationBuilder(default_pos, config["observation"]["history_length"])
    safety = SafetySupervisor(config["safety"], default_pos)
    safety.set_enabled(True)
    run_loop(transport, policy, builder, safety, args.command, config["robot"]["control_dt"], args.steps, config["policy"]["action_clip"])


if __name__ == "__main__":
    main()
