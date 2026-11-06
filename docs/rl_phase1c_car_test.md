# Phase-1c GRU: sim and car test checklist

The phase-1c CNN+GRU policy runs in `rl_policy` like the phase-1 MLP: same launch file, same
safety layer. The node reads the layout from the `.npz` (271 beams, one scan per step) and keeps
the GRU state between ticks.

## What is ready

Both 12 h runs finished (2077 updates). The final checkpoints are the ones to test:

| File | Source | Held-out eval, nominal | Laps (Spielberg / Duct1077 / Duct1112) |
|---|---|---|---|
| `models/gru_u02050.npz` | **CNN+GRU, final** (run `20261106_f9b314198a`, update 2050) | **1.00** on all 6 held-out maps | 42.7 / 9.1 / 7.9 s |
| `models/mlp3_u02050.npz` | MLP-3 baseline, final (run `20261106_4ca8863226`, update 2050; 271 beams, 3-frame stack) | **1.00** | 48.6 / 10.9 / 9.9 s |
| `models/gru_u00100.npz`, `gru_u00150.npz` | early GRU checkpoints | 1.00, 0.93 | ROS sim results below |

The GRU is about 15% faster per lap. Run the GRU first and keep the MLP-3 as the fallback: it has
no hidden state, so it is easier to reason about if the car does something odd.

## 1. Sim check (dev container: Linux, the VM, or Windows)

The sim and the policy run in the `dev` container and RViz opens as a normal window. The scripts
start the container with this machine's display settings, rebuild `rl_policy` (so new models are
installed) and run `tools/sim_gru.sh` inside. Ctrl-C stops everything. The first run builds the
image (10-20 min).

### Linux (Ubuntu laptop or desktop, or the VMware VM)

One-time setup:

```bash
curl -fsSL https://get.docker.com | sh          # Docker Engine and the compose plugin
sudo usermod -aG docker $USER                   # then log out and back in
sudo apt install -y git x11-xserver-utils
git clone https://github.com/akki-g/roboracer_ros2 ~/roboracer_ros2
```

From a terminal on the desktop:

```bash
cd ~/roboracer_ros2 && git pull
tools/sim_gru_linux.sh                                       # gru_u02050.npz on Duct1077, until Ctrl-C
tools/sim_gru_linux.sh gru_u02050.npz 90                     # lap monitor for 90 s, prints laps/crashes, then stops
tools/sim_gru_linux.sh mlp3_u02050.npz 90                    # the MLP-3 fallback
MAP=EvalDuct3012 tools/sim_gru_linux.sh gru_u02050.npz 90    # a map never used in training or selection
tools/sim_gru_linux.sh --maps                                # list the maps
```

`tools/sim_gru_vm.sh` still works (it runs the same script). With an NVIDIA driver, RViz renders on
the CPU inside the container, which is fast enough for this view. Use Docker Engine, not Docker
Desktop for Linux, which can't share the desktop's display. The container runs as root, so its
build folders (`build_docker/` etc.) need `sudo rm -rf` to delete.

### Windows (Docker Desktop + WSL 2)

Needs Windows 11, or Windows 10 with WSL from the Microsoft Store: its WSLg shows Linux windows.
One-time setup in PowerShell (and Git for Windows):

```powershell
wsl --install -d Ubuntu-22.04                         # admin PowerShell; reboot, create the Ubuntu user
# Docker Desktop: keep "Use the WSL 2 based engine" on, and turn on
#   Settings > Resources > WSL integration > Ubuntu-22.04
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # lets PowerShell run the repo's script
git clone https://github.com/akki-g/roboracer_ros2
```

Then in PowerShell:

```powershell
cd roboracer_ros2; git pull
.\tools\sim_gru_windows.ps1                                      # gru_u02050.npz on Duct1077, until Ctrl-C
.\tools\sim_gru_windows.ps1 gru_u02050.npz 90                    # lap monitor for 90 s
.\tools\sim_gru_windows.ps1 mlp3_u02050.npz 90                   # the MLP-3 fallback
.\tools\sim_gru_windows.ps1 gru_u02050.npz 90 -Map EvalDuct3012  # a never-used map
.\tools\sim_gru_windows.ps1 -Maps                                # list the maps
```

The script checks WSL and Docker Desktop, then runs `tools/sim_gru_linux.sh` in the WSL distro
(`-Distro <name>` picks one). It runs faster from a clone inside WSL: in the Ubuntu terminal,
`git clone https://github.com/akki-g/roboracer_ros2 ~/roboracer_ros2` and use the Linux commands
above. A clone made before the repo's `.gitattributes` has Windows line endings, which bash can't
run; the script says so and how to refresh it.

### By hand (Linux)

```bash
export COMPOSE_FILE=docker-compose.yml:docker-compose.linux.yml
xhost +local:
docker compose up -d dev && docker compose exec dev bash
# inside the container:
colcon build --packages-select rl_policy && source install_docker/setup.bash
tools/sim_gru.sh gru_u02050.npz 90
```

Pass: 3+ laps in 90 s on a narrow track, 0 crashes, `/drive` at 25 Hz
(`ros2 topic hz /drive` in a second `docker compose exec dev bash`).

Never-used maps (`sim/sim_rl_<name>.yaml`, from `tools/make_sim_maps.py`):

| Narrow ducts, like the car track | Racetracks (2.2 m wide) |
|---|---|
| EvalDuct3012 (66 m, 1.7 m wide), EvalDuct3014 (61 m, 1.7 m), EvalDuct3020 (71 m, 2.3 m), EvalDuct3023 (79 m, 1.8 m), EvalDuct3034 (74 m, 2.4 m), EvalDuct3053 (71 m, 2.0 m) | IMS 293 m, Melbourne 474 m, MexicoCity 357 m, Nuerburgring 446 m, Sakhir 442 m, SaoPaulo 345 m, Sepang 487 m, Shanghai 498 m, Sochi 464 m, Spa 554 m, YasMarina 398 m, Zandvoort 388 m |

Measured with the early checkpoints in the dev container, no cap, no TTC brake (the u02050 files
have not been run in the ROS sim yet; that is the check above):

| Map | Policy | Result |
|---|---|---|
| Duct1077 (held-out) | gru_u00150 | 8.4 laps / 90 s, 0 crashes, 6.5 m/s |
| Duct1077 | gru_u00100 | 7.8 laps / 90 s, 0 crashes, 6.0 m/s |
| EvalDuct3012 (never used) | gru_u00100 | 6.0 laps / 60 s, 0 crashes, 6.4 m/s |
| EvalDuct3034 (never used) | gru_u00150 | 5.6 laps / 60 s, 0 crashes, 6.7 m/s |
| Sepang (never used, 487 m) | gru_u00100 | 1.4 laps / 90 s, 0 crashes, 7.4 m/s |

## 2. Car (Jetson)

```bash
cd ~/roboracer_ros2 && git pull && docker compose up -d car && docker compose exec car bash
# inside the container:
colcon build --packages-select rl_policy && source install_docker/setup.bash
M=$(ros2 pkg prefix rl_policy)/share/rl_policy/models
ros2 run rl_policy bench_policy $M/gru_u02050.npz
ros2 run rl_policy bench_policy $M/mlp3_u02050.npz
```

Pass: `PASS` (p99 < 5 ms; the Mac does 0.27 ms p99 for the GRU, 0.03 ms for the MLP-3). Then, each
terminal a `docker compose exec car bash` (`.bashrc` sources `install_docker`). The launch line spells
out the full path: `M` from above exists only in the terminal that set it, and an empty `$M` gives
`no usable policy at '/gru_u02050.npz'` (the node then holds speed 0).

```bash
ros2 launch f1tenth_stack bringup_launch.py                                                       # terminal 1
ros2 launch rl_policy policy.launch.py mode:=car record:=true max_speed:=3.0 \
  policy_path:=$(ros2 pkg prefix rl_policy)/share/rl_policy/models/gru_u02050.npz                 # terminal 2
ros2 topic echo /policy_node/debug                                                                 # terminal 3 (optional)
```

Fallback: the same terminal 2 line with `mlp3_u02050.npz`. Drop `max_speed:=3.0` after the first slow laps.

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
