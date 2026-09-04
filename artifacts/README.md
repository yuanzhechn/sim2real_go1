# 模型文件索引

模型先按所有者、再按训练 run/模型系列分组。

| 所有者 | Bundle | 框架 | 观测维度 | 说明 |
|---|---|---|---:|---|
| yuanzhe | `yuanzhe/rough_48d_skrl_20260820` | skrl | 48 | 本体感知 Rough policy，不含高度扫描 |
| yuanzhe | `yuanzhe/rough_235d_skrl_20260815` | skrl | 235 | 需要 187 点高度扫描 |
| teammate | `teammate/single_go1_rough_235d_rsl_rl_20260817` | RSL-RL | 235 | 从 `single_go1_model_20260817.zip` 完整导入 |

旧路径 `go1_sim2real_bundle`、`go1_sim2real_bundle_rough_48d_proprio_v1` 和
`go1_rough_policy.ts` 是相对软链接，原命令仍能使用；新命令应使用上表中带所有者的路径。

`yuanzhe/legacy/go1_rough_policy.ts` 是没有配套 manifest 的早期单文件导出，优先使用完整 bundle。
