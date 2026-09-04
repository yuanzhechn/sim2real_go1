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
- Training actuator: Unitree `ActuatorNetMLP`; the hardware runtime currently
  uses joint-position PD, so matching observations and action scaling does not
  reproduce the training actuator dynamics exactly.
- Height scan: 17 x 11 = 187 values, 1.6 m x 1.0 m, 0.1 m resolution,
  Isaac Lab `xy` flatten order and clip `[-1, 1]`

Safe offline verification from the repository root:

```bash
python scripts/run_runtime.py \
  --bundle artifacts/teammate/single_go1_rough_235d_rsl_rl_20260817 \
  --config config/teammate_single_go1_rough_235d.yaml \
  --dry-run --command 0 0 0 --steps 100
```

The checked-in hardware config uses contact kinematics for planar base velocity
and generates a uniform flat-floor scan from the estimated trunk height
(`trunk_height - 0.5`). Stopping `Legged_sport` also stops the robot's ROS
odometry and point-cloud streams. This mode is only for flat-floor commissioning,
not rough-terrain perception. Use `udp_json` with a sensor process independent
of `Legged_sport` before testing actual rough terrain.
