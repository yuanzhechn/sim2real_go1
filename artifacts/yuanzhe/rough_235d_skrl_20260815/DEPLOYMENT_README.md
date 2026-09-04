# Unitree Go1 sim2real 部署包

由当前项目 `scripts/export_sim2real_bundle.py` 自动生成，所有运行代码均包含在本包内。

## 内容

- `policy/go1_rough_policy.ts`：确定性 TorchScript 策略
- `config/go1_rough_sim2real.yaml`：观测、动作和安全参数
- `runtime/go1_sim2real/`：观测、动作、策略和安全代码
- `runtime/run_sim2real.py`：运行入口
- `manifest.json`：来源和 SHA256 校验

## 当前限制

默认是 dry-run，不会发送机器人控制包。`UnitreeSdkTransport` 仍需根据实际 Go1 SDK 实现。
当前 Rough 策略需要 235 维输入，其中包括 187 维 height_scan，真机必须提供等价传感器，
不能直接用零值伪造。

## 服务器验证

```bash
python runtime/run_sim2real.py \
  --policy policy/go1_rough_policy.ts \
  --config config/go1_rough_sim2real.yaml --dry-run --steps 100
```

来源 checkpoint：`/workspace/Multi_UnitreeGo1/logs/skrl/unitree_go1_rough/2026-08-15_16-02-13_ppo_torch_go1_rough_1/checkpoints/best_agent.pt`
