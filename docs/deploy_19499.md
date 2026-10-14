# 19499 部署记录（2026-10-14）

目标机器：pi@192.168.123.161，工作目录 /home/pi/yzWorkspace/sim2real_go1。

使用独立 bundle artifacts/yuanzhe/collective_harder_19499_1072d_20261014，
配置 config/robots/unitree5151_19499.yaml。旧 48 维 bundle 与旧机器人配置继续保留。

## 输入输出核对

Checkpoint 原始 SHA256 为 119461a7d5a665f42fddefe48732bcfa932461a22bc256c1308009ddca9907d4。
actor_state_dict 只有 mlp 的四层权重/偏置和 distribution.std_param，故无观测归一化与循环状态。
网络 1072→512→256→128→12，隐藏层 ELU，输出确定性均值。
输出 12 个 group-major 关节偏移，动作缩放 0.25，周期 20 ms。

默认关节姿态来自 TrotTwoBeatCentered 的 configure_centered_reset_pose，
并与原始 19499 old_hard 录像 diagnostics.initial_joints 一致：
hip [0.1,-0.1,0.1,-0.1]，thigh [0.75,0.75,0.75,0.75]，calf [-1.5,-1.5,-1.5,-1.5]。
不能沿用旧模型的大腿 0.8/1.0 默认角。

输入网格严格核对 Go1_Locomotion height_map_env_cfg、height_map_math 和部署协议。
新增独立 terrain_height_map term，保留 187 点 height_scan 旧协议。
速度估计与 SDK 标定参数沿用原机器人接口；新模型验证状态保持 false。

## 无摄像机与后续验证

当前使用 unknown 模式：512 个高度值与 512 个有效位全零。
这表示未知地形，不表示机身高度处的真实平地。未进行无地图步态验证，不能保证盲走能力。
未知地图硬件运行额外要求 blind_deployment_validated，当前为 false。

UDP 模式接受 base_lin_vel、terrain_heights[512]、terrain_validity[512]；
真实感知必须复现相同坐标、顺序和时间语义。接收时间采用本机 monotonic。
缺字段、非二进制有效位、NaN/Inf 会拒绝；全未知真实感知触发 height_map_unavailable。
JSONL 日志与回放保留这两个数组。

## 验证

导出在本机直接对照 checkpoint 的 functional linear/ELU 结果，32 帧参考最大绝对误差 0。
本机 17 项针对性测试通过，包含新网格接口与旧本体/感知协议。
远程 ARM64 / PyTorch 2.7.1：52 项测试通过，32 帧参考最大误差 9.53674316e-07。
最终配置使用 2 个 CPU 线程与 TorchScript 常量冻结，保留原模型二进制和动作语义。
只计模型调用（参考数组预先读入）的 500 次推理：平均 8.578 ms，
p99 17.217 ms，最大 21.701 ms。
推理 p99 低于 20 ms，但最大值仍超过周期，因此不能据此声称硬实时达标。
100 步 unknown-map 离线循环通过，平均日志间隔 45.024 ms，
p99 74.546 ms，最大 83.159 ms。
未验证真实 SDK 整体周期、无地图步态或当前 checkpoint 的真机控制；硬件门保持关闭。
初期计时工具重复读取 NPZ 数据被误计入模型耗时，早期 logs/ 中的计时报告不能用于推理容量结论。
最终数据见 docs/evaluations/19499_arm64_validation.json。
此任务不启动电机。

无 JSON 日志的同配置 100 步离线循环平均间隔 30.525 ms、p99 54.436 ms，仍超过 20 ms。完整循环 50 Hz 验证未通过；需要进一步处理运行层耗时/调度，不能因为模型单次推理 p99 达标就启动真机。
