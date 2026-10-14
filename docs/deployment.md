# Go1 部署说明

## 配置与模型

当前配置 `config/robots/unitree5151.yaml` 对应
`artifacts/yuanzhe/single_flat_48d_low_level_v1_20260911`。
保留原有标定、PD、SDK 地址、动作限幅、watchdog 和遥控器逻辑。
`hardware_validated` 与 `calibration_id` 是已有接口标定记录，不能代替新模型验证。

48 维输入依次为机身线速度、角速度、重力方向、速度指令、相对关节位置、关节速度、上一动作。
`auxiliary_state.mode: kinematic_contact` 使用着地足运动学估计机身坐标系线速度，
用有实测值的原始 footForce 判断接触。不应把速度长期填零当作正常部署输入。

## SDK 和状态读取

Go1 使用 unitree_legged_sdk 的 go1 分支。

```bash
git clone --branch go1 https://github.com/unitreerobotics/unitree_legged_sdk.git ../unitree_legged_sdk
conda activate go1-sim2real
tools/setup/build_unitree_sdk.sh ../unitree_legged_sdk

python tools/hardware/read_robot_state.py \
  --config config/robots/unitree5151.yaml --enable-hardware-read
```

构建依赖见 environment.yml，sdk_python_path 指向 lib/python/arm64。
只读工具不发送控制包、不停止 sport mode；状态通道未建立或本地端口占用时可能超时。
显式真机模式会发送被动初始化包，不能当作纯只读检查。

## 真机验证

核对训练与真机的关节顺序、方向、零位、默认姿态、动作缩放、IMU 坐标系。
先保护吊挂检查逐关节响应、低增益过渡、B 急停与退出阻尼，再做有保护的平地低速验证。
记录接触数量、速度估计有效性、动作、关节状态与故障原因。

使用 README 中的包装脚本，避免 Legged_sport 和低层策略同时运行。
B 急停锁存，松开不自动恢复；退出并重启才能进入下次测试。
`--steps 0` 持续运行，真机必须记录日志。
若原厂模式恢复失败，执行：

```bash
sudo /usr/bin/python3 tools/hardware/sport_mode_control.py start
```

运行器保留旧吊挂和地面实验选项；这些选项有特殊观测/保护边界，执行前查看 --help。

## 后续高程图策略

Go1_Locomotion 1072 维策略使用 48 维本体观测、512 个相对高程值和 512 个有效位。
不能当作这里的 235 维、187 点 height_scan 策略运行。接入前需要复现网格坐标、顺序、
有效位、策略签名和 checkpoint 动作规格，并验证输入缺失、噪声与延迟条件。
