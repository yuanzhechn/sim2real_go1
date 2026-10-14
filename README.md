# Go1 Sim2Real

Unitree Go1 部署运行层：策略加载、观测拼接、关节映射、遥控器、急停和通信故障处理。
工作目录：`/home/pi/yzWorkspace/sim2real_go1`。

## 当前模型

旧版平地 bundle：`artifacts/yuanzhe/single_flat_48d_low_level_v1_20260911`。
skrl PPO，48 维本体观测、12 维关节位置动作、50 Hz，网络 `[128,128,128]`。
不需要摄像机；机身线速度由着地足运动学估计，仍需验证接触与速度估计质量。
已有 SDK/关节/IMU 标定不代表此 checkpoint 已通过吊挂与落地验证。

此旧模型不是 Go1_Locomotion 最新的 1072 维高程图策略；当前运行层已新增其 1072 维高程图接口。19499 的部署说明见
[模型 README](artifacts/yuanzhe/collective_harder_19499_1072d_20261014/DEPLOYMENT_README.md)。
无摄像机时使用显式 unknown 地形模式，目前仅离线部署验证，尚未通过盲走和新模型真机测试。

## 目录

```text
scripts/                 # 策略运行与真机包装入口
src/go1_sim2real/         # 观测、策略、动作、安全、SDK、速度估计
config/robots/           # 当前机器人配置
config/templates/        # 训练导出规格模板
config/legacy/           # 旧模型配置
artifacts/               # 完整模型 bundle
tools/hardware/          # 状态读取、标定、原厂模式管理
tools/training/          # checkpoint 导出与 Isaac Lab 检查
tools/bridges/           # 可选 ROS1 感知桥接
tools/setup/             # SDK 构建
patches/                 # SDK 兼容补丁
tests/                   # 自动化测试
docs/                    # 部署说明与历史记录
logs/                    # 本地日志，Git 忽略
```

## 安装与离线验证

从仓库根目录执行。新机器先执行 `conda env create -f environment.yml`。

```bash
conda activate go1-sim2real
python -m pip install -e . --no-deps
python -m pytest -q

python scripts/run_runtime.py \
  --bundle artifacts/yuanzhe/single_flat_48d_low_level_v1_20260911 \
  --config config/robots/unitree5151.yaml \
  --dry-run --preflight-only

python scripts/run_runtime.py \
  --bundle artifacts/yuanzhe/single_flat_48d_low_level_v1_20260911 \
  --config config/robots/unitree5151.yaml \
  --dry-run --command 0 0 0 --steps 100
```

运行器要求显式传入 `--config`，避免误用旧 Rough 配置。
`--dry-run` 和 JSONL 回放不连接机器人；遗漏真机使能参数会拒绝运行。
Bundle 校验检查 SHA256、观测/动作维度、关节顺序、默认姿态、缩放和控制周期。

## 真机入口

阅读 [部署说明](docs/deployment.md)，完成当前模型的保护测试后使用：

```bash
mkdir -p logs
scripts/run_hardware_with_remote.sh \
  --bundle artifacts/yuanzhe/single_flat_48d_low_level_v1_20260911 \
  --config config/robots/unitree5151.yaml \
  --steps 0 --log-jsonl logs/hardware-run.jsonl
```

此命令会发送电机控制。包装脚本在预检和预热完成后停止原厂 sport mode，退出时恢复。
摇杆连续回中后使能，B 锁存急停，Ctrl-C 退出。代码维护仅使用离线验证命令。

模型索引见 [artifacts/README.md](artifacts/README.md)，
工具入口见 [tools/README.md](tools/README.md)，
目录迁移见 [docs/reorganization.md](docs/reorganization.md)。
旧技术记录见 [历史文档](docs/history/SIM2REAL_MIGRATION.md)，旧命令仅作历史资料。

完成修改后验证、提交并及时推送 GitHub，规则见 [AGENTS.md](AGENTS.md)。

19499 当前状态：模型安装与离线功能验证完成；完整控制循环 50 Hz 尚未通过，未启动电机。详细时限数据见 docs/deploy_19499.md。
