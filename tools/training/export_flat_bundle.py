#!/usr/bin/env python3
"""从重新训练的 48 维 skrl Flat checkpoint 生成完整部署 bundle。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from go1_sim2real.policy import SkrlPPOPolicy


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="重新训练的 48 维 skrl best_agent.pt")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "go1_flat_bundle"))
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "templates" / "go1_flat_sim2real.yaml"),
        help="与 checkpoint 训练规格匹配的部署 YAML",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[512, 256, 128],
        help="policy 隐藏层宽度，例如 --hidden-layers 128 128 128",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_config = Path(args.config).expanduser().resolve()
    source_config_data = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    output = Path(args.output).expanduser().resolve()
    policy_path = output / "policy" / "go1_flat_policy.ts"
    config_path = output / "config" / "go1_flat_sim2real.yaml"
    output.mkdir(parents=True, exist_ok=True)

    policy = SkrlPPOPolicy.from_checkpoint(
        checkpoint,
        observation_dim=48,
        action_dim=12,
        hidden_layers=tuple(args.hidden_layers),
    )
    policy.export(policy_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, config_path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework": "skrl",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "policy": "policy/go1_flat_policy.ts",
        "policy_sha256": sha256(policy_path),
        "config": "config/go1_flat_sim2real.yaml",
        "observation_dim": 48,
        "action_dim": 12,
        "hidden_layers": args.hidden_layers,
        "action_scale": 0.25,
        "height_scan_dim": 0,
        "transport_default": source_config_data["transport"]["mode"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Flat bundle 已导出: {output}")
    print(f"policy_sha256={manifest['policy_sha256']}")


if __name__ == "__main__":
    main()
