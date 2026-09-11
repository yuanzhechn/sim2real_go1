# 模型文件索引

所有模型统一采用以下目录规则：

```text
artifacts/<owner>/<task>_<obs>_<variant>_<date>/
├── checkpoint/   # 原始训练 checkpoint
├── policy/       # 真机加载的确定性 TorchScript/ONNX
├── training/     # 该 run 自带的训练参数快照
├── config/       # 与 manifest 匹配的部署规格
├── manifest.json # 来源、维度和 SHA256
└── DEPLOYMENT_README.md
```

当前只保留：

| 所有者 | Bundle | 框架 | 观测/动作 | 用途 |
|---|---|---|---:|---|
| yuanzhe | `yuanzhe/single_flat_48d_low_level_v1_20260911` | skrl PPO | 48/12 | 单 Go1 平地低层策略 |

运行时应始终传入完整 bundle 路径，不再维护旧模型兼容软链接。
