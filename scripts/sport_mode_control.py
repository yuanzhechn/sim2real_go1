#!/usr/bin/env python3
"""精确停止/恢复本机 Go1 原厂 sport mode 守护进程。必须以 root 运行。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time


SPORT_DIR = Path("/home/pi/Unitree/autostart/sportMode").resolve()
SPORT_EXECUTABLE = (SPORT_DIR / "bin" / "Legged_sport").resolve()
KEEP_SCRIPT = (SPORT_DIR / "keep_sport_alive.sh").resolve()


def _matches() -> list[tuple[int, int, str]]:
    matches: list[tuple[int, int, str]] = []
    for process_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(process_dir.name)
            cwd = process_dir.joinpath("cwd").resolve(strict=True)
            executable = process_dir.joinpath("exe").resolve(strict=True)
            command = [
                value.decode("utf-8", errors="replace")
                for value in process_dir.joinpath("cmdline").read_bytes().split(b"\0")
                if value
            ]
            pgid = os.getpgid(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        if cwd != SPORT_DIR or not command:
            continue
        if executable == SPORT_EXECUTABLE:
            matches.append((pid, pgid, "sport"))
        elif any(Path(value).name == KEEP_SCRIPT.name for value in command):
            matches.append((pid, pgid, "keeper"))
    return matches


def status() -> list[tuple[int, int, str]]:
    if os.geteuid() != 0:
        raise SystemExit("status 必须通过 sudo 运行，普通用户无法读取 root 进程信息")
    matches = _matches()
    if not matches:
        print("sport mode: stopped")
    else:
        for pid, pgid, role in matches:
            print(f"sport mode: {role} pid={pid} pgid={pgid}")
    return matches


def stop() -> None:
    if os.geteuid() != 0:
        raise SystemExit("stop 必须通过 sudo 运行")
    matches = _matches()
    groups = sorted({pgid for _pid, pgid, _role in matches})
    for pgid in groups:
        if pgid == os.getpgrp():
            raise RuntimeError("拒绝终止当前控制脚本所在进程组")
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while _matches() and time.monotonic() < deadline:
        time.sleep(0.05)
    remaining = _matches()
    if remaining:
        raise RuntimeError(f"原厂 sport mode 未能正常退出: {remaining}")
    print("原厂 sport mode 已停止")


def start() -> None:
    if os.geteuid() != 0:
        raise SystemExit("start 必须通过 sudo 运行")
    matches = _matches()
    if any(role == "keeper" for _pid, _pgid, role in matches):
        print("原厂 sport mode 守护进程已经运行")
        return
    if not KEEP_SCRIPT.is_file() or not SPORT_EXECUTABLE.is_file():
        raise FileNotFoundError(f"原厂 sport mode 文件不完整: {SPORT_DIR}")
    log_path = SPORT_DIR / "log"
    with log_path.open("ab", buffering=0) as log_stream:
        subprocess.Popen(
            [str(KEEP_SCRIPT)],
            cwd=str(SPORT_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        matches = _matches()
        if any(role == "keeper" for _pid, _pgid, role in matches) and any(
            role == "sport" for _pid, _pgid, role in matches
        ):
            print("原厂 sport mode 已恢复")
            return
        time.sleep(0.05)
    raise RuntimeError("原厂 sport mode 启动超时")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "stop", "start"))
    args = parser.parse_args()
    {"status": status, "stop": stop, "start": start}[args.action]()


if __name__ == "__main__":
    main()
