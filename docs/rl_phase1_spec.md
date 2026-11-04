# Phase 1 RL Training & Deployment Spec

Nov 4, 2026 · @Akshat Guduru

## Goal and scope

Phase 1 trains a LiDAR-only PPO policy in f1tenth\_gym\_jax on Newton and runs it on the real car. The point is to learn whether a policy that outputs speed and steering angle transfers at all, before any teacher, localization or opponents are added.

**In scope**

- Patching f1tenth\_gym\_jax so vehicle parameters can be randomized per environment, plus wrappers for latency, steering offset, LiDAR noise and scan offset.
- Multi-map training with narrow, duct-style tracks and a held-out evaluation set.
- PPO training with checkpoints and held-out evaluation.
- A parity test between the JAX env and the classic f1tenth\_gym used by the ROS sim.
- A ROS 2 inference node with a safety layer, plus a car test protocol.

**Out of scope for phase 1:** the raceline teacher (TAL reward, BC), localization inputs, opponents, FastTD3, and tuning lap time.

**Deliverables**

1. A patched env package with tests.
2. A training script and Slurm job file.
3. Checkpoints and an evaluation report on held-out maps.
4. The `rl_policy` ROS 2 package in roboracer\_ros2, with exported weights.
5. A short results note: what worked, what failed, and the parameters used.

&#91;embedded content: phase-1 architecture · training and car pipelines\]

The observation builder and the actor weights are the only pieces that cross from training to the car, so both are tested for exact agreement.

## Repos, versions and compute

All work happens in two repos. Pin the JAX env to the commit reviewed for this spec, since it is a 1.0.0.dev0 release.

| Repo | Pin / branch | Role |
| --- | --- | --- |
| [akki-g/roboracer\_ros2](https://github.com/akki-g/roboracer_ros2) | new branch `rl-phase1` | Car stack, ROS sim, new `rl_policy` package |
| [f1tenth/f1tenth\_gym\_jax](https://github.com/f1tenth/f1tenth_gym_jax) | commit `1b4eb3f` (2026-05-28), fork it | Training env, patched |
| [f1tenth/f1tenth\_racetracks](https://github.com/f1tenth/f1tenth_racetracks) | main | Track maps and centerlines |

**Compute**

- Development and tests: Mac, JAX on CPU, small `num_envs` (64).
- Training: UCF Newton cluster, one H100 through Slurm, JAX with CUDA. Use the env's own `uv.lock` for versions.
- Logging: Weights & Biases in offline mode on Newton (the PPO example already uses `wandb`); sync after the job.

**Layout to create inside the fork**

```text
rl/
  car_params.py        # nominal values + randomization widths (section 3)
  domain_rand.py       # per-env parameter sampling
  wrappers.py          # delay, steering offset, LiDAR noise, frame stack, reward
  obs.py               # observation builder, pure NumPy/JAX, shared with the car
  maps.py              # map loading, train / held-out split, track generation
  train_ppo.py         # based on examples/train_ppo_example.py
  evaluate.py          # held-out evaluation, writes a JSON report
  export_policy.py     # params -> .npz for the car
  tests/
slurm/train_phase1.sbatch
```

On the car side, add `src/rl_policy/` to roboracer\_ros2 (section 7). Copy `obs.py` there verbatim; a test compares the two files' outputs.

## Vehicle and sensor parameters

Dynamics use the gym's own speed and acceleration limits (v\_max 20 m/s, a\_max 9.51 m/s²), as decided for phase 1. Physical parameters are randomized per episode by a multiplicative factor drawn from N(1, σ) and clipped to ±2σ. Values marked *measure* are estimates to replace with a real measurement.

| Parameter | Nominal | σ (×) | Source |
| --- | --- | --- | --- |
| Mass m (kg) | 3.45 | 0.05 | Estimate from parts; *measure* |
| CG to front axle lf (m) | 0.1587 | 0.05 | Gym split scaled to 0.33 m wheelbase; *measure* |
| CG to rear axle lr (m) | 0.1713 | 0.05 | Same |
| CG height h (m) | 0.074 | 0.15 | Gym |
| Yaw inertia I (kg·m²) | 0.0435 | 0.20 | Gym, scaled by mass |
| Front cornering stiffness C\_Sf (1/rad) | 4.718 | 0.20 | Gym |
| Rear cornering stiffness C\_Sr (1/rad) | 5.4562 | 0.20 | Gym |
| Friction μ | 1.10 | 0.10 | Carpet estimate; *measure* |
| Steering limit s\_min / s\_max (rad) | ±0.4189 | fixed | Gym; *measure* full lock |
| Steering rate sv\_min / sv\_max (rad/s) | ±3.2 | fixed | Gym, matches car `max_servo_speed` |
| Speed v\_min / v\_max (m/s) | −5.0 / 20.0 | fixed | Gym (phase-1 decision) |
| Acceleration a\_max (m/s²) | 9.51 | fixed | Gym (phase-1 decision) |
| v\_switch (m/s) | 7.319 | fixed | Gym |
| Footprint width × length (m) | 0.31 × 0.58 | fixed | Gym; *measure* with bumper and cage |
| Wheelbase (m) | 0.33 | fixed | Car config `vesc.yaml` |

**Sensing and timing**

| Parameter | Value | Source |
| --- | --- | --- |
| LiDAR field of view | 4.7124 rad (270°) | Hokuyo UST-10LX spec |
| Policy beams | 109 (every 10th of 1081, 2.5° apart) | Design choice |
| Max range | 10 m, clip and normalize to \[0, 1\] | Spec |
| Range noise | Gaussian, σ = 0.03 m | Spec repeatability |
| Beam dropout | 1% of beams set to max range | Estimate |
| LiDAR offset ahead of the car's reference point | 0.275 m | `sim.yaml`; *measure* |
| Physics step | 0.01 s | Gym |
| Policy rate | 25 Hz (`timestep_ratio` = 4) | Below the 40 Hz scan rate |
| Action delay | 0 to 2 policy steps (0–80 ms), uniform per episode | Estimate |
| Steering offset | Uniform ±0.02 rad per episode | Estimate |

**The command range is separate from the dynamics limits.** The policy's speed output maps to \[0, `v_cmd_max`\], with `v_cmd_max` = 8.0 m/s as a config value. That keeps 20 m/s possible in the dynamics without asking the policy to explore it. Section 8 lists the car-side consequences.

## Environment changes

The env needs one structural patch, per-environment vehicle parameters, and five wrappers. Everything else (scan casting, collision, progress, track loading) is used as shipped.

### 1. Per-environment vehicle parameters (blocker)

`Param` in `envs/utils.py` is passed through `static_argnums` into the jitted dynamics and integrator, so its values are compiled in. Changing them per episode either has no effect or forces a recompile.

- Split `Param` into a static config (strings, ints, bools, map name, scan settings, timestep, limits) and a traced `DynParams` pytree holding `mu, C_Sf, C_Sr, lf, lr, h, m, I`.
- Store `DynParams` in `State`, one value per env (and per agent).
- Pass it as a normal traced argument to `integrator` and the `vehicle_dynamics_*` functions. Remove it from their `static_argnums` (27 call sites carry `static_argnums`; most stay as they are).
- Sample it in `reset` from the table in section 3 using the reset key.
- Keep the change minimal and in the fork, so the diff can be upstreamed later.

### 2. Wrappers (`rl/wrappers.py`)

Apply in this order around the patched env:

1. **Action delay.** A ring buffer of the last 2 actions lives in wrapper state. The executed action is `buffer[k]`, with k drawn uniformly from {0, 1, 2} at reset.
2. **Steering offset.** Add a constant drawn at reset, uniform in ±0.02 rad, to the executed steering command.
3. **Scan origin offset.** The env casts rays from `cartesian_states[:, [0, 1, 4]]`, the model's reference point. Check in `dynamic_models.py` whether that point is the CG or the rear axle. Shift the origin forward along the heading by d = 0.275 m minus the reference point's distance from the rear axle (d ≈ 0.10 m if it is the CG). This changes `_scan` in `f110_env.py`.
4. **LiDAR noise.** Add Gaussian noise with σ = 0.03 m, clip to \[0, 10\], and set 1% of beams to 10 m. The dropouts also stand in for the gaps between duct segments.
5. **Observation and reward.** Build the observation (below), compute the reward, and keep a 3-frame scan stack in wrapper state.

### 3. Maps (`rl/maps.py`)

- `find_track_dir` downloads racetrack maps on demand; custom maps go in a map search directory with the same `{name}.yaml`, image and `{name}_centerline.csv` layout.
- Generate narrow, duct-style tracks with `examples/random_trackgen.py`: widths 1.5–2.5 m and tight corners, since duct tracks are narrower than the scaled F1 circuits.
- **Training set:** 8 racetracks plus 8 generated tracks. **Held-out set:** 3 racetracks plus 3 generated tracks, never used for training. Fix the generator seeds and write both lists to the run config.
- Each env instance holds one track, and grid sizes differ, so build one env per map. Split `num_envs` evenly across maps and step all of them inside one jitted function; a Python loop over at most 16 maps unrolls fine.

### 4. Action

- Use the env's built-in `velocity+steeringangle` control type. It maps one-to-one onto the car's `AckermannDriveStamped` (speed, steering\_angle).
- The policy outputs a = (a\_v, a\_s) in \[−1, 1\]. Speed = (a\_v + 1) / 2 × `v_cmd_max`; steering = a\_s × 0.4189.
- The dynamics enforce the steering-rate limit (3.2 rad/s) and acceleration limit (9.51 m/s²) themselves.

### 5. Observation (`rl/obs.py`, shared with the car)

One function builds the observation from raw inputs, so the sim and the car call identical code:

- **Scan:** 109 beams at 2.5° spacing over 270°. In sim, set `num_beams` = 109 and `fov` = 4.7124. On the car, take every 10th beam of the 1081-beam scan (indices 0, 10, …, 1080). Clip to \[0, 10\] m and divide by 10.
- **Stack:** the current scan plus the previous 2.
- **Speed:** longitudinal speed divided by `v_cmd_max`.
- **Previous action:** (a\_v, a\_s) as executed.
- **Total size:** 3 × 109 + 1 + 2 = 330.

The observation leaves out yaw rate and measured steering angle on purpose. The car has no steering sensor, and its odometry yaw rate is computed from the steering command rather than measured. Using only signals the car really has avoids a sim-to-real mismatch.

### 6. Reward and termination

- **Reward per step:** r = 10 × Δprogress (fraction of a lap) − 0.01 × |a\_t − a\_{t−1}|₁, minus 1 on collision.
- Use the env's built-in `progress` reward for Δprogress, and add the other terms in the wrapper.
- **Episode ends** on collision, after 2 laps, or at `max_steps` (90 s of sim time).
- **Reset:** a random point on the centerline (the env already does this), heading along the track, speed 0.

The TAL teacher reward is phase 1b. Add it only after this baseline trains, so the effect of each change can be measured on its own.

## Policy and PPO training

Start from `examples/train_ppo_example.py`, which already has a PureJaxRL-style loop, GAE, minibatching and safetensors checkpoints. Change the env construction, network and action distribution as described here.

**Network**

- Separate actor and critic MLPs, each with two hidden layers of 256 units, tanh activations, orthogonal initialization (as in the example).
- Both take the 330-dimensional observation. The privileged critic is phase 2.
- **Action distribution:** a diagonal Gaussian with a state-independent learned log-std (initialized at −0.5), squashed by tanh with the log-probability correction (distrax `Transformed` with `Tanh`). Clipping a Gaussian instead would bias the gradients at the bounds.
- **At deployment and evaluation,** use tanh(mean).

**Starting hyperparameters**

| Setting | Value | Note |
| --- | --- | --- |
| num\_envs | 2048 | Split evenly across training maps |
| num\_steps (rollout) | 256 | About 10 s of driving at 25 Hz |
| total\_timesteps | 2e8 | Checkpoints let you stop early |
| num\_minibatches | 32 |  |
| update\_epochs | 4 |  |
| learning rate | 3e-4, linear anneal |  |
| clip ε | 0.2 |  |
| γ / GAE λ | 0.99 / 0.95 |  |
| entropy coef | 0.001 | Raise to 0.005 if the std collapses early |
| value coef / max grad norm | 0.5 / 0.5 |  |
| Observation normalization | Running mean/std on the 3 non-scan inputs only | Scans are already in \[0, 1\] |

**Logging, checkpoints and evaluation**

- Log every update: mean return, episode length, completion rate, collision rate, mean speed, policy std, value loss, approx KL.
- Log the sampled parameters (μ, delay, mass and so on) alongside episode outcomes, so failures can be traced to particular draws.
- Save a checkpoint every 25 updates, including params, the optimizer state, the observation normalizer, and the full config.
- Every 50 updates, run `evaluate.py` on the held-out maps: nominal parameters, no noise or delay, deterministic actions, 10 episodes per map from fixed start points. Record completion rate, lap time, collisions and mean speed per map.
- Keep the checkpoint with the best held-out completion rate as `best.safetensors`, separate from the latest.

**Slurm job (`slurm/train_phase1.sbatch`)**

- One H100, 12 hours wall time.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`; W&B in offline mode.
- Write outputs to a run directory named after the config hash and date.
- The job must resume from the latest checkpoint if restarted.

## Validation before the overnight run

No overnight job starts until every check below passes. Each is a pytest in `rl/tests/` unless noted.

**Unit checks**

- [ ] **Randomization varies per env.** Reset 64 envs; μ, mass and delay differ across them and stay within ±2σ.
- [ ] **Parameters change the physics.** Two envs with μ = 0.8 and μ = 1.3 get the same constant hard-turn action. Their trajectories diverge, and the low-μ car slips more.
- [ ] **No recompile on reset.** Run reset and step 100 times with `jax.log_compiles` on; one compile only.
- [ ] **Delay works.** With k = 2, the executed action equals the action from 2 steps earlier.
- [ ] **Scan offset.** Facing a wall at a known distance, the front beam reads that distance minus d, within 1 cm.
- [ ] **Observation shape and range.** Shape is (330,); scan values lie in \[0, 1\]; the stacked frames update in order.
- [ ] **Shared observation code.** The sim and car copies of `obs.py` give identical output on 100 recorded inputs.
- [ ] **Beam angle match.** The sim's 109 beam angles equal the car's subsampled angles (indices 0, 10, …, 1080) within 1e-4 rad.

**Parity test against the ROS sim (`rl/tests/test_parity.py`)**

The car is tested in the ROS sim first, which runs the classic f1tenth\_gym, so the two simulators must agree.

1. Fix all parameters to nominal, with no noise, delay or offset.
2. Drive both sims with the same 500-step action sequence (straights, a slalom, and a hard turn) from the same pose on the same map.
3. Pass if position error stays under 0.05 m and heading error under 0.05 rad for the first 200 steps.
4. If it fails, check in this order: action ordering (steering vs speed), units, timestep and `timestep_ratio`, the dynamics model (`st` vs `ks`), and the scan origin.

**Smoke run**

- On Newton, with 256 envs for 10 minutes.
- Pass if returns rise, there are no NaNs, episode lengths grow, the per-env sampled parameters show variance in the log, and one checkpoint plus one held-out evaluation report get written.
- Record steps per second. That sets `total_timesteps` for the 12-hour job.

## Deployment

The car runs the actor as plain NumPy inside a ROS 2 node, with no JAX on the Jetson. The node publishes to the existing `ackermann_mux` at navigation priority, so the joystick (priority 100) always overrides it.

### 1. Export (`rl/export_policy.py`)

Write `policy.npz` with the actor weights, log-std, observation normalizer mean and std, and a JSON config: beam indices, `v_cmd_max`, steering limit, policy rate, frame-stack length, and the checkpoint hash.

Test: a NumPy forward pass matches the JAX actor's tanh(mean) within 1e-5 on 1000 observations.

### 2. The `rl_policy` node (roboracer\_ros2, `src/rl_policy/`)

- An ament\_python package with one node, `policy_node`. Its dependencies are rclpy, sensor\_msgs, nav\_msgs, ackermann\_msgs and numpy.
- Subscribe to the scan and odometry topics; keep the latest message of each.
- A 25 Hz timer builds the observation with the shared `obs.py`, runs the forward pass, scales the action and publishes `AckermannDriveStamped`.
- The executed action becomes the next observation's previous action. Reset the frame stack and previous action when the node activates.
- **Topics are parameters:**

| Parameter | Car (bringup) | ROS sim (f1tenth\_gym\_ros) |
| --- | --- | --- |
| `scan_topic` | `/scan` | `/scan` |
| `odom_topic` | `/odom` | `/ego_racecar/odom` |
| `drive_topic` | `/drive` (mux `navigation`, priority 10) | `/drive` |

Confirm the sim topic names with `ros2 topic list` before the first run; namespacing differs between bridge versions.

### 3. Safety layer (inside the node, applied after the policy)

1. **Speed cap:** clamp the speed command to the `max_speed` parameter, 1.5 m/s by default.
2. **Steering clamp:** ±0.4189 rad.
3. **Emergency brake:** time-to-collision over the beams within ±30° of straight ahead, TTC = range / max(v·cos θ, 0.01). Below `ttc_threshold` (0.5 s), command speed 0 and log the event.
4. **Watchdog:** a scan older than 100 ms or odometry older than 200 ms commands speed 0.
5. **Deadman:** the joystick teleop at priority 100 overrides everything, and the mux timeout stops the car if this node dies.

The speed cap also covers the gap between the training limits and the car. The policy may command up to `v_cmd_max` (8 m/s) and assume 9.51 m/s², but the car's VESC tops out near 4.9 m/s and its throttle smoother allows 2.5 m/s².

### 4. Test protocol

Record a rosbag of `/scan`, `/odom`, `/drive` and the node's debug topic (observation summary, raw action, safety overrides) on every run.

| Stage | Setup | Pass |
| --- | --- | --- |
| 0. ROS sim | Node against f1tenth\_gym\_ros on a held-out map | Completes 3 laps at `max_speed` 3.0 |
| 1. Stand | Wheels off the ground, objects placed by hand at the LiDAR | Steering turns toward open space, and the sign convention is correct; the brake triggers on an object ahead |
| 2. Straight | Floor, straight corridor, `max_speed` 1.0 | Holds the centre with no oscillation for 10 m |
| 3. Loop | Small duct loop, `max_speed` 1.5 | 3 laps without intervention |
| 4. Speed steps | Same loop, `max_speed` 2.0, then 3.0 | Lap completion stays at 100%; record lap times |

Stop and pull the bag after any intervention. Compare the logged observations with the sim's at the same speed to locate the gap.

## Acceptance, task order and risks

Phase 1 succeeds if the policy completes held-out maps in sim and finishes 3 laps on the real duct loop at a capped speed without intervention.

**Acceptance criteria**

| Gate | Criterion |
| --- | --- |
| Env | All unit checks and the parity test pass |
| Training | Held-out completion rate of at least 90% at nominal parameters |
| Robustness | Held-out completion of at least 75% with randomization and noise on |
| Export | NumPy and JAX actions match within 1e-5 |
| Car | Test-protocol stage 3 passes (3 laps at 1.5 m/s, no intervention) |

**Task order for the implementing agent**

1. Fork f1tenth\_gym\_jax at `1b4eb3f`, install it, and run its own tests.
2. Do the per-env parameter refactor and its three unit checks.
3. Write the wrappers and `obs.py`, with their unit checks.
4. Write `maps.py`: generate the narrow tracks and fix the train and held-out lists.
5. Adapt `train_ppo.py` and `evaluate.py`, then run 5 minutes on CPU with 64 envs.
6. Run the parity test against the classic f1tenth\_gym.
7. Smoke run on Newton, then submit the 12-hour job.
8. Write `export_policy.py` and the `rl_policy` ROS 2 package, then run stage 0 in the ROS sim with an early checkpoint.
9. Write a results note: commands run, test results, throughput, and anything that deviated from this spec.

**Risks and open decisions**

- **Training limits exceed the car's.** Dynamics use 20 m/s and 9.51 m/s², while the car runs at most about 4.9 m/s and 2.5 m/s² (`vesc.yaml`). The speed cap keeps tests safe, but the car will lag every speed change the policy expects. Before racing, either raise the car's limits or retrain with matching ones.
- **Estimated parameters.** Mass, CG split, friction and the LiDAR offset are estimates. Replace them once measured, retrain, and compare.
- **The env refactor may sprawl.** If the parameter patch is not working within about 3 hours, ship the first run without dynamics randomization. Keep delay, steering offset and LiDAR noise, which need no env change, and record the deviation.
- **Reference point of the dynamics model.** The scan offset depends on whether the state's x, y is the CG or the rear axle. Verify it in code; don't assume.
- **Steering sign.** `vesc.yaml` uses a negative steering gain. Confirm on the stand that positive `steering_angle` turns left on the car, as it does in sim.
- **Generated tracks vs duct tracks.** If held-out results are much worse on generated narrow tracks than on racetracks, the duct-style geometry needs more weight in training.
