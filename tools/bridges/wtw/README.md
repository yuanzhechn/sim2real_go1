# WTW-compatible adapter owned by sim2real_go1

This directory contains our adapter, safety gate and status schema. It does not vendor the upstream WTW source or model weights.
The upstream clone remains a sibling at `../walk-these-ways` (commit `0e7236bdc81ce855cbe3d70345a7899452bdeb1c`).
Its existing LCM schemas are read at configure time; generated files stay under ignored `build/`.

## Build and test without moving the robot

On the Pi, using the existing SDK and LCM development installation:

```bash
cmake -S tools/bridges/wtw -B build/wtw_bridge \
  -DUNITREE_SDK_ROOT=/home/pi/yzWorkspace/unitree_legged_sdk
cmake --build build/wtw_bridge -j2
ctest --test-dir build/wtw_bridge --output-on-failure
```

The Python LCM binding used for integration checks is `lcm==1.5.2`.
Export the upstream baseline using `tools/training/preflight_walk_these_ways.py` first.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/hardware/check_wtw_bridge_sim.py \
  --output logs/reference_audit/wtw_simulated_pipeline.json
```

That script always launches `--simulate`, uses `GO1_WTW_TEST_` channels and never constructs a Unitree UDP connection.
It checks 100 inference frames, then stops publishing commands and verifies a latched timeout.
The simulator is a bounded position follower, not a dynamics simulation or evidence of locomotion.

For SDK-independent gate tests on a development machine:

```bash
cmake -S tools/bridges/wtw -B build/wtw_gate -DBUILD_LCM_BRIDGE=OFF
cmake --build build/wtw_gate
ctest --test-dir build/wtw_gate --output-on-failure
```

## Interface

- Target control and native UDP send/receive loops: 2ms. Coherent status snapshots: 10ms.
- Original WTW messages remain available; our Python path consumes one complete status snapshot.
- Simulator prefix: `GO1_WTW_TEST_`. Hardware prefix: `GO1_WTW_`. Neither subscribes to upstream unprefixed motor commands.
- Default LCM URL uses multicast TTL0. Remote routing/interface behavior has not been commissioned.
- Hardware is off by default. SDK construction requires explicit hardware/protection/sport-mode flags; R2 then arms, B latches damping.
- Commands are sequenced and wall-clock checked, with a 100ms monotonic receipt timeout. Sensor freshness is also checked at 100ms.
- A separate sender watchdog requests damping if the validated packet stops updating. This cannot protect against a completely frozen host or network failure; firmware behavior still requires hardware validation.
- Invalid commands and faults latch until process restart. A new command cannot clear a fault.
- First target must be within0.1rad of feedback; subsequent target changes are bounded at2.5rad/s plus0.002rad tolerance.
- True target tracking error:0.45rad. Estimated requested PD effort:20Nm. SDK PositionLimit and PowerProtect level1 remain enabled.
- Kp<=40, Kd<=2, zero desired joint velocity/feed-forward, joint velocity<=6rad/s, battery>=21V, temperature<=60°C, |roll/pitch|<=0.45rad, centered sticks.

The gate uses current-target tracking, not deviation from a fixed standing pose.
The bridge has a maximum30-second lifetime and requests damping on exit; it does not manage or restore sport mode.
Hardware startup, exclusive controller ownership, native feedback, shutdown handoff, real timing and locomotion remain unvalidated.
There is no commissioned hardware launch procedure in this change.

## Policy adapter

`src/go1_sim2real/wtw.py` implements the pinned70x30 observation contract, two action histories, gait clock and per-leg joint layout.
`tests/test_wtw_interface.py` compares50 steps against the sibling upstream LCMAgent/HistoryWrapper using mocked network/camera objects.
It also compares the SDK target positions against upstream serialized messages. The upstream implementation is read, not copied into this repository.
The19499 bundle and its observation contract are unchanged.
