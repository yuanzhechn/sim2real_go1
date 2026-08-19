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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go1_sim2real.policy import SkrlPPOPolicy


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="重新训练的 48 维 skrl best_agent.pt")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "go1_flat_bundle"))
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    policy_path = output / "policy" / "go1_flat_policy.ts"
    config_path = output / "config" / "go1_flat_sim2real.yaml"
    output.mkdir(parents=True, exist_ok=True)

    policy = SkrlPPOPolicy.from_checkpoint(
        checkpoint,
        observation_dim=48,
        action_dim=12,
        hidden_layers=(512, 256, 128),
    )
    policy.export(policy_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "config" / "go1_flat_sim2real.yaml", config_path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "policy": "policy/go1_flat_policy.ts",
        "policy_sha256": sha256(policy_path),
        "config": "config/go1_flat_sim2real.yaml",
        "observation_dim": 48,
        "action_dim": 12,
        "hidden_layers": [512, 256, 128],
        "action_scale": 0.25,
        "height_scan_dim": 0,
        "transport_default": "dry_run",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Flat bundle 已导出: {output}")
    print(f"policy_sha256={manifest['policy_sha256']}")


if __name__ == "__main__":
    main()
