#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sdk_root=${1:-"$project_root/../unitree_legged_sdk"}

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "请先执行: conda activate go1-sim2real" >&2
  exit 2
fi
if [[ ! -d "$sdk_root/.git" ]]; then
  echo "未找到 SDK 仓库: $sdk_root" >&2
  echo "先克隆: git clone --branch go1 https://github.com/unitreerobotics/unitree_legged_sdk.git '$sdk_root'" >&2
  exit 2
fi

compat_patch="$project_root/patches/unitree_sdk_modern_pybind.patch"
if git -C "$sdk_root" apply --check "$compat_patch" 2>/dev/null; then
  git -C "$sdk_root" apply "$compat_patch"
elif ! git -C "$sdk_root" apply --reverse --check "$compat_patch" 2>/dev/null; then
  echo "SDK CMake 文件与兼容补丁不匹配，请确认使用官方 go1 分支" >&2
  exit 2
fi

build_dir="$sdk_root/build-conda-no-ros"
env -u PYTHONPATH -u CMAKE_PREFIX_PATH \
  PATH="$CONDA_PREFIX/bin:$PATH" \
  CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" \
  cmake -S "$sdk_root" -B "$build_dir" \
    -DPYTHON_BUILD=TRUE \
    -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE

env -u PYTHONPATH -u CMAKE_PREFIX_PATH \
  PATH="$CONDA_PREFIX/bin:$PATH" \
  CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" \
  cmake --build "$build_dir" -j2

echo "robot_interface 输出目录: $sdk_root/lib/python/arm64"
