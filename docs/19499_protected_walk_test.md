# 19499受保护直行验证（2026-10-14）

用户明确要求验证行走，并确认前方至少0.5m平整无障碍、吊带可允许小幅前移、遥控急停可用。
继续使用19499原checkpoint、未知地图输入、前腿Kp20/后腿Kp40。
常规模型hardware_validated和blind_deployment_validated保持false。

## 指令与保护

策略阶段15秒：零速站稳3秒、2秒线性升至0.05m/s直行、保持5秒、2秒回零、零速保持3秒。
左右速度和偏航指令始终为零，不做转弯。
动作上限从0.2渐增至0.5或1.0，随后随指令回到0.2；动作变化仍限制每帧0.05。
上限1.0对应0.25rad目标偏移，不表示完整未经限幅的actor已经验证。
仍保留20Nm请求PD力矩、0.45rad跟踪误差、6rad/s关节速度、60°C温度、21V电池、
0.45rad roll/pitch、新鲜度、B急停、SDK PositionLimit/PowerProtect和计算周期保护。
新增要求有效机身速度估计、遥控摇杆回中；估计速度超过0.3m/s、
估计移动超过0.3m、侧向超过0.1m、偏航变化超过0.25rad或无估计足支撑则退出。

机身速度来自足端运动学约束；积分是body-frame近似，不能代替独立位移测量。
Go1 SDK footForceEst在本次为零，速度估计沿用已配置的raw footForce回退。
接触数量和压力变化只是辅助证据，必须由用户现场确认交替迈步及真实前移。

## 实际试验

0.5动作上限：750帧、15秒完成，无计算超时，最大计算13.73ms。
用户反馈“一直站着或触发了停机”；日志确认正常完成而非保护停机，判定未验证出行走。
保持指令阶段平均估计前进速度约0.00064m/s，整个运动观察窗口积分前向约0.022m；
姿态调整造成的积分不能作为迈步证据。

离线近似重建站姿观测：12个原始动作中10个超出0.5上限，说明限幅抹去了多数动作变化。
该分析并非精确硬件观测回放：初次日志未保留完整角速度和关节速度。
下一轮补记48维本体观测、原始actor动作、SDK原始及估计足压力，便于精确复查。

1.0动作上限：750帧、15秒完成，无计算超时，最大计算15.22ms。
保持指令阶段平均估计前进速度约0.00175m/s；前向积分约0.034m，侧向约0.030m。
该阶段约58.6%原始动作仍受到限幅。腿部变化增大，但遥测未证明稳定前进；
用户现场观察仍待记录，不计为行走验证通过。
两次均恢复原厂控制，所有保护参数没有为取得行走结果而放宽。

本机与机器人23项推理等价性、指令渐变与保护测试通过。
详细日志位于机器人 logs/19499_walk005_trial.json 和 logs/19499_walk005_limit1_trial.json。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u tools/hardware/test_19499_stand.py \
  --enable-hardware --protected --ground-support-startup --ground-support-hold \
  --rear-support-kp 40 --startup-duration 10 --backend numpy --hold-seconds 15 \
  --forward-speed 0.05 --walking-action-limit 1.0 \
  --output logs/19499_walk005_limit1_trial.json
```

这是一轮有界调试，不是常驻或自动导航入口；到时减速回零并恢复原厂控制。
