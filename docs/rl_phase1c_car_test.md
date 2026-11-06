# Phase-1c GRU: sim and car test checklist

The phase-1c CNN+GRU policy runs in `rl_policy` like the phase-1 MLP: same launch file, same
safety layer. The node reads the layout from the `.npz` (271 beams, one scan per step) and keeps
the GRU state between ticks.

## What is ready

| File | Source | Held-out eval, nominal | ROS sim, Duct1077, 90 s, no cap, no TTC brake |
|---|---|---|---|
| `models/gru_u00150.npz` | GRU run `20261106_f9b314198a`, update 150 | 0.93 (narrow ducts 0.7-0.9) | 8.4 laps, 0 crashes, 6.5 m/s mean (a rerun: 6.8 laps/90 s, two slow creeps near a wall) |
| `models/gru_u00100.npz` | same run, update 100 | **1.00** | 7.8 laps, 0 crashes, 6.0 m/s mean |
| newer | `tools/fetch_gru_policy.sh` (below) | picks the fastest eval with completion 1.00 | run the sim check below |

The 12 h training run keeps going (about update 1400 by 8 am); a newer checkpoint is usually better.

## 1. Newest policy (Mac, optional)

```bash
ssh -fN newton                                   # password + Duo
cd ~/Desktop/CARVEL/roboracer_ros2
tools/fetch_gru_policy.sh                        # or: tools/fetch_gru_policy.sh 20261106_f9b314198a <update>
git add src/rl_policy/models/gru_u*.npz && git commit -m "rl_policy: GRU policy update <N>" && git push
```

## 2. Sim check (Ubuntu VM)

```bash
git pull && colcon build --packages-select rl_policy && source install/setup.bash
./start_sim.sh sim/sim_rl.yaml                                                            # terminal 1
P=$(ros2 pkg prefix rl_policy)/share/rl_policy/models/gru_u00100.npz
ros2 launch rl_policy policy.launch.py mode:=sim policy_path:=$P                          # terminal 2
ros2 run rl_policy stage0_monitor --ros-args -p centerline:=$PWD/sim/maps/Duct1077_centerline.csv -p duration:=90.0   # terminal 3
```

Pass: 3+ laps, 0 crashes, `/drive` at 25 Hz. The node log shows `loaded ... gru_u...npz`, the
scan resampled to 271 angles (worst error 0.6 deg, ok), and `policy active: ... recurrent state reset`.

## 3. Car (Jetson)

```bash
git pull && docker compose up -d car && docker compose exec car bash
colcon build --packages-select rl_policy && source install_docker/setup.bash
ros2 run rl_policy bench_policy $(ros2 pkg prefix rl_policy)/share/rl_policy/models/gru_u00100.npz
```

Pass: `PASS` (p99 < 5 ms; the Mac does 0.15 ms p50). Then:

```bash
ros2 launch f1tenth_stack bringup_launch.py                                              # terminal 1
ros2 launch rl_policy policy.launch.py mode:=car policy_path:=$P record:=true            # terminal 2
ros2 topic echo /policy_node/debug                                                        # terminal 3 (optional)
```

**Driving:** the policy's `/drive` reaches the motor only while the autonomy deadman (button 5,
RB on the F710) is held. Release it and the joystick's zero command takes over within 0.2 s.
Button 4 (LB) is manual driving.

**Order:**
1. **Wheels off the ground.** Hold RB: the wheels spin up, and steering turns away from a box held
   near one side of the LiDAR. Check the steering sign: `vesc.yaml` has a negative gain. Release RB:
   the wheels stop.
2. **On the track, slow first.** The policy was trained without a speed cap, and a low cap puts it
   out of its training range (that broke the phase-1 MLP in the sim), so keep the first runs short:
   `max_speed:=3.0` for a lap or two, then the config's 10.0 (the policy itself tops out at 8 m/s).
3. **Keep the TTC brake as configured** (`policy_car.yaml`: on, 0 deg cone, centre beam only).

**What the node does on its own:**
- **Hidden-state resets.** The GRU state, frame stack and previous action reset at start-up,
  after a watchdog stop (scan older than 100 ms or odometry older than 200 ms), when `enabled` is
  toggled on, and after 2 s below 0.2 m/s. The last one mirrors training's stall episode end: if
  the car sits held by the joystick before you press RB, it starts from a fresh state.
- **Debug topic.** `/policy_node/debug` reports `policy_ms` (last, p50, p99), `gru_h_norm`,
  `stall_resets` and every safety override.

**Watch for:**
- slow creeping at 0.5-1 m/s close to a wall in tight corners (seen twice in the sim; the
  0.5-1.5 m/s band is where the training and ROS sims differ);
- the LiDAR offset (training assumes 0.275 m ahead of the rear axle);
- the steering sign on the stand.

Bags from `record:=true` land in the repo root (`rosbag2_*`).
