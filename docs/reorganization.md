# 目录整理记录（2026-10-14）

## 调整

- scripts/ 保留 run_runtime.py 和 run_hardware_with_remote.sh。
- 状态、标定和 sport mode 工具迁入 tools/hardware/。
- 导出/Isaac 工具迁入 tools/training/；ROS1 桥接迁入 tools/bridges/。
- SDK 构建迁入 tools/setup/，仓库根目录计算同步修正。
- config/yuanzhe_single_flat_48d_low_level_v1.yaml 迁入 config/robots/unitree5151.yaml。
- 导出模板迁入 config/templates/；其余旧配置迁入 config/legacy/。
- SIM2REAL_MIGRATION.md 迁入 docs/history/，保留原文并标记历史路径。
- 删除生成的 src/go1_sim2real.egg-info 和本地 Python/pytest 缓存，完善忽略规则。
- 更新内部路径、模型部署说明和测试；运行器 --config 改为必填。
- 增加 pytest 开发依赖、源码路径配置和 AGENTS.md 的提交/推送规则。

模型二进制、manifest、训练快照、bundle 内配置和机器人控制参数保留原值。
旧配置与可复用工具保留，以便复现。整理前备份位于仓库相邻目录
sim2real_go1_backup_20261013_203824.tar.gz，未提交到 Git（备份文件名使用远程机器时钟）。

## 验证

整理前 36 项测试通过；整理后 39 项测试全部通过（含三项 CLI 集成测试）。
当前 bundle 的离线 preflight 与 100 步零指令 dry-run 通过。
六个迁移后 Python 工具 --help、全部 Python 源码语法、两个 shell 入口语法和 git diff --check 通过。
逐文件 SHA256 核对确认模型 payload/manifest/训练配置不变；唯一 bundle 文本变化是外部部署命令路径。
当前机器人 YAML 与迁移前字节一致。
Isaac Lab 与 ROS 工具只完成源码语法和路径检查，未在机器人上执行对应服务。
本次维护不连接 SDK 状态通道、不发送电机命令、不停止 sport mode。
