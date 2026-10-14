# Collective harder model 19499

原始 checkpoint SHA256：119461a7d5a665f42fddefe48732bcfa932461a22bc256c1308009ddca9907d4。
来源：Go1_Locomotion/logs/collective_harder_final_snapshot_20261013/collective_harder_20261013/model_19499.pt。
RSL-RL 确定性 ELU MLP [1072,512,256,128,12]，无 GRU、无观测归一化。
只导出 actor，保留原始 checkpoint 与训练快照。distribution.std_param 是训练采样参数，不参与确定性动作。

输入：48 维本体观测 + 512 个 terrain_z-base_z 高程值 + 512 个二进制有效位。
网格 16 行(y) x 32 列(x)，x 最快变化；前向 0–3.1 m，横向 ±0.75 m，间距 0.1 m。
高程截断 [-1,1]，未知格点值为 0、有效位为 0；不能假装未知地图是已知平地。
默认大腿角四腿均为 0.75 rad（与旧 48 维模型不同），默认姿态来自 centered-reset 训练配置与原始录像记录。
动作：12 个 group-major 关节偏移，q_target=q_default+0.25*action，控制周期 0.02 s。

当前无摄像机，config/robots/unitree5151_19499.yaml 使用 height_map_mode: unknown。
本体速度继续采用着地足运动学估计。没有声称屏蔽地图后能正常盲走。
新模型 hardware_validated=false，blind_deployment_validated=false；旧接口标定编号单独保留。
不得仅把旧模型的验证状态复制过来启动 19499 电机控制。

离线入口（仓库根目录）：

```bash
python scripts/run_runtime.py \
  --bundle artifacts/yuanzhe/collective_harder_19499_1072d_20261014 \
  --config config/robots/unitree5151_19499.yaml \
  --dry-run --preflight-only

python scripts/run_runtime.py \
  --bundle artifacts/yuanzhe/collective_harder_19499_1072d_20261014 \
  --config config/robots/unitree5151_19499.yaml \
  --dry-run --command 0 0 0 --steps 100
```

validation/parity.npz 为直接 checkpoint 线性层/ELU 计算的参考，包含未知地图和已知平地样本。
导出误差为 0；数值一致性不代表步态或真机验证。
后续感知采用 UDP terrain_heights[512] / terrain_validity[512]，同时提供 base_lin_vel[3]。
设置 height_map_mode: udp_json 时必须使用相同网格的真实感知，并配置 auxiliary_state.mode: udp_json。
缺失字段、非法有效位、陈旧状态或全未知地图会被拒绝/触发故障。

远程最终结果：52 项测试、预检与 100 步离线循环通过。配置使用 2 CPU 线程与 TorchScript 常量冻结；推理平均 8.58 ms、p99 17.22 ms、最大 21.70 ms，未认证硬实时或真机步态。详见 docs/deploy_19499.md。

完整离线控制循环平均间隔约 45 ms（含 JSON 日志），无日志约 30.53 ms；完整 50 Hz 时限尚未通过，当前只完成安装与功能验证。
