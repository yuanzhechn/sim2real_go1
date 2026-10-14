# 配置用途

- robots/unitree5151.yaml：当前 48 维模型的单 Go1 部署配置，保留已有标定。
- templates/go1_flat_sim2real.yaml：48 维导出模板，隐藏层默认 [512,256,128]；必须显式匹配实际 checkpoint。
- legacy/my_go1.yaml：旧 48 维模型配置 [512,256,128]，不对应当前 bundle。
- legacy/go1_rough.yaml：旧 235 维模板，需要 187 点扫描。
- legacy/teammate_single_go1_rough_235d.yaml：旧队友模型配置，对应 bundle 已不在仓库。

配置不能互换。训练规格随 bundle 保存在 artifacts/；不要修改 bundle 内配置绕过校验。
运行器必须显式传入 --config。

- robots/unitree5151_19499.yaml：1072 维 19499 策略，无摄像机 unknown-map 模式，新模型真机与盲走验证状态为 false。
