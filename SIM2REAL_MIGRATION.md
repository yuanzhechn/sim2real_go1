# Go1 Sim2Real 迁移与开发手册

本文档专门说明当前 sim2real 项目已经完成什么、迁移到 Unitree Go1 还需要什么，以及真机开发时必须注意什么。它不是 README 的替代品；README 只负责项目入口和快速说明。

## 1. 项目边界

当前工作流分成两台机器：

```text
训练服务器
Isaac Sim + Isaac Lab + skrl
        │ 导出 TorchScript
        ▼
Go1 机载计算机或外部控制机
状态读取 + 观测拼接 + 策略推理 + 安全层 + Unitree SDK
        │ 控制指令
        ▼
Unitree Go1
```

机器人上不需要安装 Isaac Sim 或 Isaac Lab。服务器负责训练，机器人侧只运行推理和控制循环。

当前仓库同时管理 Isaac Lab Manager-Based Rough 的 skrl 与 RSL-RL 单智能体 PPO 导出。
两者训练 checkpoint 格式不同，但部署均使用经过签名校验的 TorchScript policy。

## 2. 已经完成的内容

### 2.1 策略导出

已经实现从当前项目的 skrl PPO checkpoint 中提取 policy 网络并导出 TorchScript：

- 输入维度：235；
- 隐藏层：`512 -> 256 -> 128`；
- 激活函数：ELU；
- 输出维度：12；
- 使用 Gaussian policy 的确定性均值动作；
- 不包含 optimizer、value 网络、PPO memory 和 Isaac Lab 环境。

当前导出的文件：

```text
/workspace/sim2real/artifacts/go1_rough_policy.ts
```

部署时应使用 `.ts` 文件，不应直接把 skrl 的 `best_agent.pt` 当作独立真机模型。

### 2.2 观测拼接

`Go1ObservationBuilder` 已按当前 Isaac Lab Rough 配置实现 235 维观测。顺序不可改变：

| 索引 | 观测项 | 维度 | 真机需要提供 |
|---:|---|---:|---|
| 0–2 | `base_lin_vel` | 3 | 机身坐标系线速度 |
| 3–5 | `base_ang_vel` | 3 | IMU/状态估计器角速度 |
| 6–8 | `projected_gravity` | 3 | IMU 姿态转换后的重力方向 |
| 9–11 | `velocity_commands` | 3 | vx、vy、yaw 速度指令 |
| 12–23 | `joint_pos_rel` | 12 | 编码器绝对角减默认角 |
| 24–35 | `joint_vel_rel` | 12 | 关节速度 |
| 36–47 | `actions` | 12 | 上一个归一化策略动作 |
| 48–234 | `height_scan` | 187 | 与训练采样点一致的地形高度 |

维度计算：

```text
3 + 3 + 3 + 3 + 12 + 12 + 12 + 187 = 235
```

一个非常重要的细节：`RobotState.joint_pos` 表示真机编码器的绝对角度，观测层会计算：

```text
joint_pos_rel = joint_pos_absolute - default_joint_pos
```

不能把绝对角度直接送入策略。

### 2.3 动作映射

策略输出的是归一化动作，不是关节弧度。当前 Isaac Lab Go1 Rough 的动作缩放为 `0.25`：

```text
joint_target = default_joint_pos + 0.25 * action
```

`Go1JointPositionMapper` 已经实现这一转换。Isaac Lab 5.1 运行时解析出的 12 个关节顺序为：

```text
FL_hip, FR_hip, RL_hip, RR_hip,
FL_thigh, FR_thigh, RL_thigh, RR_thigh,
FL_calf, FR_calf, RL_calf, RR_calf
```

这个顺序必须同时和 USD、Isaac Lab 配置、Unitree SDK 适配器一致。不能仅凭名称猜测 SDK 顺序，必须在真机上逐关节验证。

### 2.4 安全层

`SafetySupervisor` 已实现：

- roll/pitch 姿态超限保护；
- 状态时间戳超时保护；
- 关节偏离默认姿态保护；
- NaN、维度错误和非法动作检查；
- 动作限幅；
- 动作变化速率限制；
- 故障时输出零动作；
- 使能开关。

dry-run 和 JSONL 回放 transport 只在本地运行，不会发送任何网络控制包。测试还覆盖
真机 transport 的关节重排、重力投影、目标/增益重排和只读零发送保证。

## 3. 真机适配实现与仍需现场完成的内容

以下事项必须在真机测试前完成，目前不能视为已经实现。

### 3.1 Unitree SDK 通信（代码已实现，现场验证未完成）

`UnitreeSdkTransport` 已针对 Go1 的 `unitree_legged_sdk` 低层 Python wrapper
（模块名 `robot_interface`）实现：

1. 读取并显式重排 12 个电机角度、速度、模式和温度；
2. 读取 IMU 的 wxyz 四元数、角速度和 roll/pitch，并计算 projected gravity；
3. 将归一化动作映射成目标角，转换关节顺序/方向/零位后发送 q、dq、Kp、Kd、tau；
4. 调用 SDK PositionLimit/PowerProtect，检测 UDP 错误和 LowState tick 超时；
5. 解析遥控器摇杆并按可配置死区/比例生成 vx、vy、yaw 指令，B 键作为锁存急停；
6. 故障与进程退出时发送阻尼模式，SDK 支持时启用断连 watchdog；
7. 提供严格只读、绝不发送命令的状态检查脚本。
8. 真机命令模式启动前检查并拒绝与本机 `Legged_sport` 进程并发；包装脚本在模型
   完成加载和预热后停止原厂 sport 守护进程，并在策略退出后恢复。

仍需在目标机安装/构建低层 `robot_interface`，并验证固件兼容性、UDP 地址、按键、
关节方向、零位和 PD。默认配置中的 `hardware_validated: false` 会阻止真机启动。

### 3.2 高度扫描（接入协议已实现，传感器投影仍需现场实现）

当前 Rough 策略依赖 187 维高度扫描。Isaac Lab 中的 RayCaster 是理想仿真传感器，真实 Go1 不会自动拥有同样的数据。

运行层已经提供 UDP JSON 辅助状态入口，同时接收机身线速度与 187 点扫描，并检查维度、
有限值、更新延迟和单调时间戳。感知进程仍必须把真实雷达/深度相机数据转换成训练时
完全一致的坐标系、采样点和顺序；运行层不会用零值替代真机扫描。

可选方案：

方案 A：接入激光雷达、深度相机或其他传感器，将数据投影为训练时相同的 187 个采样点。必须匹配坐标系、采样顺序、单位、裁剪范围、更新频率和延迟。

机载相机的平地调试桥见 `scripts/ros_rough_auxiliary_bridge.py`。它按照 Isaac Lab 默认
`GridPatternCfg(size=[1.6, 1.0], resolution=0.1, ordering="xy")` 生成 17×11 点，并采用
`base_height - ground_height - 0.5` 的观测定义。因为当前相机不能直接覆盖整个网格，
该桥只在点云能通过覆盖率、平面残差、坡度和新鲜度门槛时拟合/外推平面，**仅用于平地
commissioning**；它会抹掉台阶等非平面地形，不能作为最终 Rough 感知方案。

方案 B：重新训练不包含 `height_scan` 的策略，只使用 IMU、关节状态和速度指令。若目前只有平地传感器，这是更适合的第一阶段路线。

当前仓库已实现方案 B 的机器人端支持：Flat actor 使用 48 维观测，配置见
`config/my_go1.yaml`，机载 `/ros2udp/odom` 可通过
`scripts/ros_odom_auxiliary_bridge.py` 转成真实 `base_lin_vel` 辅助包。训练服务器重新
得到 48 维 skrl checkpoint 后，使用 `scripts/export_flat_bundle.py` 导出；旧 235 维
checkpoint 不能通过裁剪输入直接转换。

绝对不能把 187 维全部填零后直接把 Rough 模型放到复杂地形上。那等于改变了策略输入分布。

### 3.3 状态估计和坐标系（转换可配置，符号仍需现场验证）

必须在真机上确认：

- 线速度是 SDK 直接提供，还是由 IMU/里程计估计；
- IMU 坐标系到机身坐标系的旋转；
- 四元数顺序是 xyzw 还是 wxyz；
- roll、pitch 的正方向；
- 关节角的正方向；
- yaw 指令和机身坐标的定义；
- 时间戳是否使用单调时钟。

仿真中这些值通常是理想值，真机中任何一个符号或坐标系错误，都可能让策略立即输出错误动作。

### 3.4 实物参数标定（必须现场完成）

还没有针对你的具体 Go1 完成：

- 12 个关节零位；
- 关节顺序和方向；
- 默认站立姿态；
- Kp/Kd；
- action scale；
- 控制频率和控制延迟；
- 电流、速度和位置限幅；
- 电池电压、电机温度和通信保护。

配置文件中的默认角度来自 Isaac Lab 配置，只能作为初始参考，不能自动视为实物标定值。

## 4. 迁移目录和依赖

建议将以下内容复制到 Go1 机载计算机或外部控制机：

```text
sim2real/
├── config/go1_rough.yaml
├── artifacts/go1_rough_policy.ts
├── src/go1_sim2real/
│   ├── action.py
│   ├── observation.py
│   ├── policy.py
│   ├── runtime.py
│   ├── safety.py
│   ├── transport.py
│   └── types.py
└── scripts/run_runtime.py
```

机器人侧不需要：

- Isaac Sim；
- Isaac Lab；
- skrl；
- PPO 训练代码；
- `.pt` checkpoint 中的 optimizer/value 网络。

机器人侧只需要与目标 CPU/Jetson 兼容的 Python、PyTorch、NumPy、PyYAML，以及具体 Unitree SDK 运行库。仓库提供 `environment.yml`，当前 ARM64 CPU 环境选择 Python 3.11 和已验证的 conda-forge `pytorch-cpu` 2.7；这是基于当前包可用性与实测结果选择，不是策略本身强制要求 Python 3.11。如果机载计算机资源不足，可以在外部 NUC/工控机运行推理和 SDK，通过网线连接 Go1。

## 5. 迁移步骤

### 5.1 在服务器导出模型

```bash
cd /workspace/Multi_UnitreeGo1
/workspace/IsaacLab/isaaclab.sh -p /workspace/sim2real/scripts/export_skrl_policy.py \
  --checkpoint logs/skrl/unitree_go1_rough/2026-08-15_16-02-13_ppo_torch_go1_rough_1/checkpoints/best_agent.pt \
  --output /workspace/sim2real/artifacts/go1_rough_policy.ts
```

导出后检查文件存在并记录 SHA256：

```bash
sha256sum /workspace/sim2real/artifacts/go1_rough_policy.ts
```

### 5.2 在服务器 dry-run

```bash
cd /workspace/sim2real
/workspace/IsaacLab/isaaclab.sh -p scripts/run_runtime.py \
  --bundle artifacts/yuanzhe/rough_235d_skrl_20260815 \
  --config config/go1_rough.yaml \
  --dry-run --steps 100
```

部署 bundle 时优先传入 bundle 根目录。运行层会验证 manifest 中的策略 SHA256，
并逐项比对训练/部署的观测维度、关节顺序、默认姿态、动作缩放、控制周期和网络结构；
单独部署经过确认的 TorchScript 文件时仍可使用 `--policy`。

启动时看到 `safety=action_delta_limited` 是动作渐变保护，动作会逐步增加，不代表模型加载失败。

### 5.3 配置并验证真机 transport

构建 Go1 `unitree_legged_sdk` 的 Python wrapper，并在复制出的真机配置中填写
`sdk_python_path`、UDP 地址、关节重排/方向/零位、限位、PD 和外部感知端口。
先运行 `scripts/read_robot_state.py --enable-hardware-read` 做无命令状态检查。

约定：

- `read_state()` 返回绝对关节角；
- 观测层负责减 `default_joint_pos`；
- `send_action()` 接收归一化动作；
- 发送给 SDK 前调用 `Go1JointPositionMapper.to_joint_target()`；
- SDK 的关节顺序必须在适配器中显式转换；
- 所有动作发送都必须经过 `SafetySupervisor`。

### 5.4 安装和复制

在服务器导出后，将整个 sim2real 目录和策略文件复制到目标控制机。目标机不应依赖服务器的 `/workspace/Multi_UnitreeGo1` 或 `/workspace/IsaacLab` 路径。

先只验证：

1. TorchScript 能加载；
2. 输入为 `(235,)` 时输出为 `(12,)`；
3. 推理耗时满足控制周期；
4. 不接电机时 transport 可以读取测试状态并记录动作。

## 6. 真机测试流程

### 6.1 无策略的状态读取测试

在吊挂或断开执行器的安全状态下，先只读取状态：

1. 手动移动每个关节，确认数组中的关节对应正确；
2. 检查编码器角度单位是否为弧度；
3. 检查 IMU roll/pitch 方向；
4. 检查机身速度单位和坐标系；
5. 检查状态时间戳刷新频率；
6. 模拟网线断开，确认 watchdog 触发。

### 6.2 单关节动作测试

在吊挂状态下，只给一个关节很小的目标偏移，确认：

- 目标关节正确；
- 正负方向正确；
- 目标角度没有多乘或少乘 action scale；
- 其他关节保持安全姿态；
- 急停立即生效。

### 6.3 吊挂策略测试

第一次加载策略必须满足：

- Go1 离地吊挂；
- 遥控器处于安全/阻尼状态；
- 物理急停在旁边；
- 速度指令从零开始；
- 至少一人看机器人，一人看日志并负责断电；
- 先短时间运行，再检查温度、电流、姿态和动作饱和情况。

### 6.4 平地低速测试

只有吊挂测试通过后，才进行落地测试。建议顺序：

```text
站立 → 0 m/s 保持 → 低速前进 → 低速后退 → 低速转向 → 小幅侧移
```

不要第一轮就使用最大速度、快速 yaw 或 Rough 地形。

## 7. 安全要求

必须保留以下保护：

- 物理急停；
- 遥控器使能开关；
- 通信超时 watchdog；
- roll/pitch 超限保护；
- 关节位置和速度保护；
- 电机温度和电池电压保护；
- 动作变化速率限制；
- 控制程序退出时进入安全模式或阻尼模式。

禁止：

- 直接绕过安全层发送 PPO 输出；
- 把零值 height scan 当作真实地形；
- 未确认关节顺序就落地运行；
- 未吊挂就第一次启动低层控制；
- 修改零位、action scale、Kp、Kd 后不记录版本；
- 没有日志就调真机参数；
- 把仿真成功当成真机安全证明。

## 8. 训练侧后续工作

当前模型虽然使用了 Rough 地形和执行器网络，但这不等于已经完成 sim2real。建议继续确认或加入：

- 观测延迟和动作延迟随机化；
- 摩擦、质量、质心、重力、电机强度随机化；
- 关节零位偏差随机化；
- Kp/Kd 偏差随机化；
- IMU/编码器噪声和滤波延迟；
- 与真机一致的控制频率和 decimation；
- 真实高度传感器的噪声、遮挡和更新延迟；
- 真机电机动作和速度限幅。

如果当前没有高度传感器，优先训练一个不含 `height_scan` 的策略，而不是修改输入维度或用零值补齐当前策略。

## 9. 迁移验收清单

- [ ] TorchScript 在目标控制机可以独立加载；
- [ ] 输入固定为 `(235,)`，输出固定为 `(12,)`；
- [ ] 关节名称、顺序、单位和正方向已逐项确认；
- [ ] 默认姿态和零位已实测；
- [ ] IMU 坐标系和四元数顺序已验证；
- [ ] `base_lin_vel` 的来源和单位已确认；
- [ ] 187 维 height scan 已与真实传感器对应，或已改用重新训练的无扫描策略；
- [ ] 50 Hz 控制循环的延迟和抖动已测量；
- [ ] 断网、进程退出和 SDK 错误都会触发安全动作；
- [ ] 急停和遥控器使能有效；
- [ ] 吊挂测试通过；
- [ ] 平地低速测试通过；
- [ ] 温度、电流、姿态和策略动作已记录；
- [ ] 最后才尝试更高速度或 Rough 地形。

## 10. 当前状态总结

已经完成的是：

```text
策略导出
+ 235 维观测接口
+ 12 维动作接口
+ 安全层
+ dry-run/JSONL 验证
```

代码层已经新增：

```text
Go1 unitree_legged_sdk 低层通信
+ 显式关节/IMU 转换
+ 外部速度和高度扫描协议
+ 遥控器使能/急停/watchdog/阻尼退出
+ 真机配置门和 JSONL 日志
```

仍必须在实物上完成的是：

```text
+ SDK/固件通信实测
+ 真实状态估计器与高度传感器投影
+ 零位/关节方向/PD 标定
+ 吊挂测试和落地测试
```

因此，软件 transport 已不再是占位类，但在验收清单的现场项目完成前，仍不能直接让
Go1 自主行走。
