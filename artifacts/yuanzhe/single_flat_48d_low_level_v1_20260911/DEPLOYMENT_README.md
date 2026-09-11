# Single Flat 48D Low-Level v1

训练 run：`2026-09-11_10-00-53_ppo_torch_flat_48d_low_level_v1`

- 框架：skrl PPO Gaussian policy（部署使用确定性均值）
- 输入：48维，依次为 base linear velocity、base angular velocity、
  projected gravity、velocity command、12维相对关节位置、12维关节速度、
  12维上一动作
- 输出：12维 group-major 关节位置动作
- 网络：`[128, 128, 128]`，ELU
- 控制频率：50 Hz
- 动作映射：`q_target = q_default + 0.25 * action`
- 地形：plane；不需要 height scan
- 训练执行器：Unitree `ActuatorNetMLP`

离线校验：

```bash
conda activate go1-sim2real
cd /home/pi/yzWorkspace/sim2real_go1
python -u scripts/run_runtime.py \
  --bundle artifacts/yuanzhe/single_flat_48d_low_level_v1_20260911 \
  --config config/yuanzhe_single_flat_48d_low_level_v1.yaml \
  --command 0 0 0 --dry-run --steps 100
```

真机部署配置沿用本机已完成的 SDK/关节/IMU 标定，但这个 checkpoint 本身
尚未完成吊挂与落地验证。原始 checkpoint 和该训练 run 的参数快照均已保存在
bundle 内，后续不依赖服务器日志目录。
