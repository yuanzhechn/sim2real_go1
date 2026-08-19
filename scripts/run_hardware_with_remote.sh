#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPORT_CONTROL="${PROJECT_ROOT}/scripts/sport_mode_control.py"
RUNTIME="${PROJECT_ROOT}/scripts/run_runtime.py"
PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "请先激活 go1-sim2real Conda 环境" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "用法: scripts/run_hardware_with_remote.sh --bundle BUNDLE --config HARDWARE_CONFIG [其他参数]" >&2
  exit 2
fi
for argument in "$@"; do
  case "${argument}" in
    --dry-run|--preflight-only|--manage-sport-mode)
      echo "真机包装脚本不接受 ${argument}" >&2
      exit 2
      ;;
  esac
done

sudo -v
# 长时间运行时保持本次 sudo 授权有效，以便 Python/退出陷阱无需再次交互即可恢复 sport。
( while kill -0 "$$" 2>/dev/null; do sudo -n -v || exit; sleep 30; done ) &
sudo_keepalive_pid=$!

restore_sport_mode() {
  local exit_code=$?
  trap - EXIT HUP INT TERM
  echo "确认原厂 sport mode 已恢复..."
  sudo -n /usr/bin/python3 "${SPORT_CONTROL}" start || {
    echo "严重警告：原厂 sport mode 自动恢复失败，请执行 sudo python3 ${SPORT_CONTROL} start" >&2
    exit_code=1
  }
  kill "${sudo_keepalive_pid}" 2>/dev/null || true
  exit "${exit_code}"
}
trap restore_sport_mode EXIT HUP INT TERM

"${PYTHON_BIN}" -u "${RUNTIME}" "$@" \
  --command-source remote --enable-hardware --manage-sport-mode
