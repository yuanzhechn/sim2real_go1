# 辅助工具

从仓库根目录运行。需要主运行包的工具会定位仓库的 src/。

| 目录 | 工具 | 用途与依赖 |
|---|---|---|
| hardware | check_policy_offline.py | 无 SDK 的策略一致性检查和 CPU 推理计时 |
| hardware | read_robot_state.py | SDK 只读状态检查，显式 --enable-hardware-read |
| hardware | calibrate_joint_targets.py | 吊挂低增益标定，会驱动电机 |
| hardware | calibrate_ground_contacts.py | 接触/速度估计标定，按工具硬件开关使用 |
| hardware | sport_mode_control.py | 原厂模式管理，需要 sudo |
| setup | build_unitree_sdk.sh | ARM64 SDK 构建，应用 patches/ 补丁 |
| training | export_flat_bundle.py | skrl 48 维完整 bundle 导出 |
| training | export_skrl_policy.py | skrl 策略导出 |
| training | inspect_isaac_runtime_spec.py | Isaac Lab 训练侧规格检查 |
| training | evaluate_rsl_policy_isaac.py | Isaac Lab 中离线评估旧 235 维模型 |
| bridges | ros_odom_auxiliary_bridge.py | ROS1 里程计到 UDP，需要 Python2/ROS Melodic |
| bridges | ros_rough_auxiliary_bridge.py | ROS1 Rough 感知到 UDP |

当前机器人使用接触运动学速度估计，不需要启动 ROS 桥接。
训练侧 Isaac 工具需要训练机器上的 Isaac Lab，不属于 Go1 的运行依赖。

`training/export_rsl_elevation_bundle.py`：严格导出 1072 维、无归一化/无循环状态的 RSL actor，生成 checkpoint 参考动作；拒绝覆盖已有 bundle。

`hardware/test_19499_stand.py`：用户明确确认吊带/急停后的 19499 短时零速调试；电池>=21V、动作±0.2、20ms周期保护、退出恢复原厂模式。当前真实尝试被13%电量拦截，未进入姿态过渡。见 docs/19499_protected_stand_test.md。
