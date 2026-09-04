这是最初的单狗 Go1 运控模型导出包。

任务：Isaac-Go1-Project-Rough-v0
训练 run：2026-08-17_10-00-56
最终 checkpoint：model_2999.pt
观测输入：235 维
动作输出：12 维关节位置动作

文件说明：
- model_2999.pt：RSL-RL 完整训练 checkpoint，可继续训练/用 IsaacLab play.py 加载
- policy.pt：TorchScript 部署策略，输入 [1,235]，输出 [1,12]
- policy.onnx：ONNX 部署策略，输入 obs [1,235]，输出 actions [1,12]
- env.yaml：训练环境配置
- agent.yaml：PPO/网络配置

SHA256：
model_2999.pt  44c35dac57e03de97c33b92aa73d73f2a28ca4538228286f6fa4d9f4a604b14a
policy.pt     9d79e505959ce49a1115958a8b4dfc8a6e3f41bd84a8a23880be27a71c8a638d
policy.onnx   9fd02ea9f5f1a114eae1c5f71da98a9597c3b38433c38b7bddbe5e7e8a160175
