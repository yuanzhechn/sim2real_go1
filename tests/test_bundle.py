from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from go1_sim2real.bundle import load_bundle, validate_bundle_runtime_config
from go1_sim2real.config import load_config


def _make_bundle(tmp_path: Path, policy_bytes: bytes = b"torchscript-placeholder") -> Path:
    root = tmp_path / "bundle"
    (root / "policy").mkdir(parents=True)
    (root / "config").mkdir()
    policy = root / "policy" / "policy.ts"
    policy.write_bytes(policy_bytes)
    training = {
        "robot": {
            "control_dt": 0.02,
            "joint_count": 12,
            "joint_names": [f"joint_{i}" for i in range(12)],
            "default_joint_pos": [0.0] * 12,
            "action_scale": 0.25,
        },
        "observation": {
            "dimensions": [3, 3, 3, 3, 12, 12, 12, 187],
            "height_scan_clip": 1.0,
            "history_length": 1,
        },
        "policy": {
            "observation_dim": 235,
            "action_dim": 12,
            "hidden_layers": [512, 256, 128],
            "action_clip": 1.0,
        },
    }
    (root / "config" / "go1_rough_sim2real.yaml").write_text(
        yaml.safe_dump(training), encoding="utf-8"
    )
    manifest = {
        "policy": "policy/policy.ts",
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "observation_dim": 235,
        "action_dim": 12,
        "hidden_layers": [512, 256, 128],
        "action_scale": 0.25,
        "height_scan_dim": 187,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_bundle_integrity_and_runtime_match(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path)
    bundle = load_bundle(root)
    runtime_path = tmp_path / "runtime.yaml"
    runtime = dict(bundle.training_config)
    runtime["safety"] = {}
    runtime["transport"] = {"mode": "dry_run"}
    runtime_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")
    validate_bundle_runtime_config(bundle, load_config(runtime_path))


def test_bundle_rejects_modified_policy(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path)
    (root / "policy" / "policy.ts").write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256"):
        load_bundle(root)


def test_bundle_rejects_runtime_action_scale_mismatch(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path)
    bundle = load_bundle(root)
    runtime_path = tmp_path / "runtime.yaml"
    runtime = dict(bundle.training_config)
    runtime["robot"] = dict(runtime["robot"], action_scale=0.5)
    runtime["safety"] = {}
    runtime["transport"] = {"mode": "dry_run"}
    runtime_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="robot.action_scale"):
        validate_bundle_runtime_config(bundle, load_config(runtime_path))
