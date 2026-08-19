# Go1 Sim2Real

Unitree Go1 的 sim2real 运行层，连接 Isaac Lab 训练策略与真实机器人控制器。

详细的完成状态、迁移步骤、真机接口和安全要求见：

- [SIM2REAL_MIGRATION.md](SIM2REAL_MIGRATION.md)

当前目录提供：

- 235 维 Rough 观测拼接器；
- skrl PPO checkpoint 到 TorchScript 的导出；
- Go1 动作映射和安全层；
- dry-run、JSONL 回放和 Unitree Go1 低层 SDK transport；
- 外部里程计/187 点高度扫描 UDP 接口；
- 遥控器使能、急停、通信 watchdog、安全阻尼退出和真机日志。

默认运行模式是 dry-run，不会发送任何机器人控制包。

## 快速验证

```bash
conda env create -f environment.yml
conda activate go1-sim2real
python -m pip install -e . --no-deps
python -m pytest -q
python scripts/run_runtime.py \
  --bundle artifacts/go1_sim2real_bundle \
  --config config/go1_rough.yaml \
  --dry-run --steps 100
```

优先使用 `--bundle`：运行层会校验 manifest、策略 SHA256，以及训练和部署配置中的
观测维度、关节顺序、默认姿态、动作缩放和控制周期。只有单独复制策略文件时才使用
`--policy artifacts/go1_rough_policy.ts`。

## 真机接入

Go1 使用 `unitree_legged_sdk`（不是面向 Go2 的 sdk2）。先构建官方 Go1 分支的低层
Python wrapper；仓库脚本会应用 Python 3.11/现代 pybind11 所需的最小 CMake 兼容补丁：

```bash
git clone --branch go1 https://github.com/unitreerobotics/unitree_legged_sdk.git ../unitree_legged_sdk
conda activate go1-sim2real
scripts/build_unitree_sdk.sh ../unitree_legged_sdk
```

将输出的 `lib/python/arm64` 绝对路径填入 `transport.sdk_python_path`，确认可导入
`robot_interface`。复制配置文件后逐项
标定 `sdk_joint_names`、方向、零位、关节限位和 PD 参数。只读检查不会发送任何控制包：

```bash
python3 scripts/read_robot_state.py \
  --config config/my_go1.yaml \
  --enable-hardware-read
```

发送低层命令前必须按 Unitree 的安全流程停止本机 `Legged_sport`；运行层默认也会检查
并拒绝两者并发。只读检查不要求停止该进程，但可能需要为本机 UDP 端口选择未占用值。
低层 UDP 在新客户端首次发送前通常不会回传 LowState；显式真机模式会先发送零力矩、
电机失能的被动初始化包建立状态通道，而只读模式始终保持零发送并可能因此超时。

当前 Rough 策略必须接收真实的机身坐标系线速度与 187 点高度扫描。感知进程需向
`transport.auxiliary_state.bind_port` 发送一帧一个 UDP JSON 包：

```json
{"base_lin_vel":[0.0,0.0,0.0],"height_scan":["严格按训练顺序排列的 187 个数值"]}
```

只有完成文档中的逐关节、坐标系、急停和吊挂验证后，才能把副本配置中的
`transport.mode` 改为 `unitree_sdk`、填写非空 `calibration_id` 并将
`hardware_validated` 改为 `true`。真机运行还必须同时传入 `--enable-hardware`，并建议
始终使用 `--log-jsonl`。默认配置故意无法启动真机。

真机遥控安全逻辑默认为 `toggle`：程序启动时策略关闭；松开后按一次 `L2` 开启，
再按一次关闭。`B` 是锁存急停，按下后即使松开也不会恢复，必须退出并重新启动程序。
`--steps 0` 表示持续运行，按 `Ctrl-C` 也会进入退出阻尼流程。
