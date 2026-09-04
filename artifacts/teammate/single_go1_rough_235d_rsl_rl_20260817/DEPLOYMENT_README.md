# Teammate single-Go1 RSL-RL model

This directory is a normalized sim2real bundle built from
`source/single_go1_model_20260817.zip`.

- Framework: RSL-RL `ActorCritic`
- TorchScript policy: `policy/policy.pt`
- ONNX policy: `policy/policy.onnx`
- Full checkpoint: `checkpoint/model_2999.pt`
- Input: 235 values
- Output: 12 joint-position actions
- Action mapping: `q_target = q_default + 0.25 * action`
- Height scan: 17 x 11 = 187 values, 1.6 m x 1.0 m, 0.1 m resolution,
  Isaac Lab `xy` flatten order and clip `[-1, 1]`

Safe offline verification from the repository root:

```bash
python scripts/run_runtime.py \
  --bundle artifacts/teammate/single_go1_rough_235d_rsl_rl_20260817 \
  --config config/teammate_single_go1_rough_235d.yaml \
  --dry-run --command 0 0 0 --steps 100
```

Real-hardware execution requires a fresh UDP stream containing both
`base_lin_vel` and `height_scan`; the runtime will reject a missing or stale
stream. `scripts/ros_rough_auxiliary_bridge.py` is only a flat-floor
commissioning bridge and is not a real rough-terrain scanner.
