# 开源sim2real参考与Go1基线（2026-10-15）

用户要求参考PACE与Walk These Ways，必要时直接使用Go1基线；参考仓库放在部署仓库上级的同级目录。

## 仓库位置与来源

- 本地WTW：`/home/yuanzhe/Research/walk-these-ways`
- 机器人WTW：`/home/pi/yzWorkspace/walk-these-ways`
- 本地PACE：`/home/yuanzhe/Research/pace-sim2real`
- WTW上游：https://github.com/Improbable-AI/walk-these-ways
- WTW提交：`0e7236bdc81ce855cbe3d70345a7899452bdeb1c`
- PACE上游：https://github.com/leggedrobotics/pace-sim2real
- PACE提交：`7ad0f45db5ed01103543ddc25a812fef61a998ab`

原始参考仓库保持Git干净，没有嵌套到sim2real_go1，也未修改其上游remote。
PACE辨识/仿真代码保留在开发机；未在Pi上安装Isaac或运行系统辨识激励。

## 已核对的实现差异

WTW的`go1_gym_deploy/unitree_legged_sdk_bin/lcm_position.cpp`使用独立控制、UDP发送、UDP接收循环，dt=0.002。
Python策略通过LCM发送位置/速度/增益/前馈力矩，策略dt=0.02。
当前19499路径的Python同步读状态/推理/发命令，与这种底层和策略循环分离架构不同。
这是应优先复用和验证的结构，不足以单独证明已有振动的根因。

WTW提供逐段姿态校准和R2启动流程，并通过配置恢复其训练时的参数。
不能直接照搬其上游启动脚本：脚本含旧目录、容器与进程管理假设，且会实际启动低层控制。
本次没有运行这些脚本。上游C++桥也没有完整的命令时效检查，不能未经适配直接当作本机器人常驻控制器。
本项目既有急停、时效、力矩、姿态和电池保护应保留并接入新链路。

| 项目 | 19499 | WTW随附Go1基线 |
|---|---|---|
| actor输入 | 1072维，含地图与有效位 | 70维×30帧历史=2100，另预测2维自适应latent |
| 指令 | 3维速度 | 15维，含步态参数与时钟 |
| 关节输出顺序 | 4髋、4大腿、4小腿 | 每腿髋/大腿/小腿排列 |
| action_scale | 0.25 | 0.25 |
| 髋关节额外缩放 | 1.0 | 0.5 |
| 默认大腿 | 均0.75rad | 前0.8rad、后1.0rad |
| 摄像机地图 | 未知图模式尚未验证行走 | 此基线actor不含高度图 |

WTW模型必须使用自己的输入缩放、两帧历史动作、步态时钟、历史窗口与自适应模块，
不能只换checkpoint或套用19499的观测构造器。
19499的不可变模型bundle保留原样；WTW将作为单独策略标识和独立对照基线。

PACE的`pace_actuator.py`及教程用于后续电机/关节响应辨识：
编码器偏置、关节摩擦、惯量和延迟需要与真实链路对齐；命令延迟与力矩延迟应明确区分。
不能把PACE辨识替换为一次任意调Kp/Kd，也不能把ANYmal示例参数移植到Go1。

## 已完成的纯离线预检

工具：`tools/training/preflight_walk_these_ways.py`。
只读取上游配置与checkpoint，在CPU重建原始ELU student网络并导出TorchScript；
不导入机器人transport，不创建LCM，不接管sport mode，不发送电机命令。

权重：`runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/checkpoints/ac_weights_last.pt`
SHA256：`d1542b0b8c7b767753a286f1cbabe5b4dc86e67611fb95844e870b3c61f8869e`。
32组输入导出前后最大误差为0；这验证导出数值一致，不是动态仿真或硬件验证。
Pi单线程300次CPU推理：平均9.908ms、P99 13.342ms，仅计网络，不含完整通信/控制周期。
两台机器的Torch版本不同，导出归档哈希可能不同；原checkpoint哈希一致。
结果仍标记hardware_validated=false。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/training/preflight_walk_these_ways.py \
  --source-root ../walk-these-ways --output logs/reference_audit/wtw_0e7236b
```

导出结果保存在各自部署仓库的`logs/reference_audit/wtw_0e7236b/`，不覆盖上游原仓库或19499。
离线预检摘要见`evaluations/wtw_upstream_preflight_pi.json`。

后续先适配带命令时效和急停的高频底层通信，再分别验证WTW观测/动作接口与完整控制周期；
未完成这些步骤前，不宣称“同为Go1所以直接上狗即可”。
