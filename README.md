# Go1 Sim2Real

Unitree Go1 的 sim2real 运行层，连接 Isaac Lab 训练策略与真实机器人控制器。

详细的完成状态、迁移步骤、真机接口和安全要求见：

- [SIM2REAL_MIGRATION.md](SIM2REAL_MIGRATION.md)

当前目录提供：

- 235 维 Rough 观测拼接器；
- skrl PPO checkpoint 到 TorchScript 的导出；
- Go1 动作映射和安全层；
- dry-run、JSONL 回放和 Unitree SDK 适配入口。

默认运行模式是 dry-run，不会发送任何机器人控制包。
