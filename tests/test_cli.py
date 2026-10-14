"""Exercise the deployed bundle through the actual CLI without touching hardware."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/yuanzhe/single_flat_48d_low_level_v1_20260911"
CONFIG = ROOT / "config/robots/unitree5151.yaml"


def invoke(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_runtime.py"), "--bundle", str(BUNDLE), *args],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )


def test_cli_requires_explicit_config():
    result = invoke("--dry-run")
    assert result.returncode == 2
    assert "--config" in result.stderr


def test_cli_dry_run_works_without_sdk(tmp_path):
    import yaml
    data = yaml.safe_load(CONFIG.read_text())
    data["transport"]["sdk_module"] = "sdk_must_not_be_imported"
    config = tmp_path / "robot.yaml"
    config.write_text(yaml.safe_dump(data))
    result = invoke("--config", str(config), "--dry-run", "--command", "0", "0", "0", "--steps", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "step=000002" in result.stdout


def test_cli_refuses_hardware_without_enable():
    result = invoke("--config", str(CONFIG), "--steps", "1")
    assert result.returncode != 0
    assert "--enable-hardware" in result.stderr
