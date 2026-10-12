# roboracer_ros2

ROS 2 Humble workspace for our RoboRacer (F1TENTH) car, run in Docker on the Jetson.

**What stays on the Jetson itself (not in Docker):** the udev rules in `/etc/udev/rules.d`, Bluetooth pairing of the controller, and Docker. The container sees the VESC, LiDAR and controller through `/dev`.

## First-time setup on the Jetson

1. **Docker.** Check with `docker --version`. If it's missing: `sudo apt install -y docker.io`. To run Docker without `sudo`: `sudo usermod -aG docker $USER`, then log out and SSH back in.
2. **Clone this repo** on the Jetson, e.g. `git clone https://github.com/akki-g/roboracer_ros2.git ~/roboracer_ros2`.
3. **Bring over the working packages (one time).** Copies `~/f1tenth_ws/src` into `src/` without their `.git` folders, so the controller mapping and calibration come along:
   ```bash
   bash migrate_native_ws.sh            # or: bash migrate_native_ws.sh /path/to/f1tenth_ws
   git add src && git commit -m "Add driver stack" && git push
   ```
4. **Build the image and container:** `bash setup_docker.sh`
5. **Build the workspace:** `bash setup.sh`

## Every day

```bash
bash start.sh                                  # shell inside the container, everything sourced
ros2 launch f1tenth_stack bringup_launch.py    # wheels off the ground for the first test
```

Each `bash start.sh` opens a separate shell in the same container, so teammates can SSH in and work side by side. Only run **one** `bringup_launch.py` at a time, and don't run the old native workspace alongside the container.

| Script | Does |
| --- | --- |
| `setup_docker.sh` | Build the image and create the container. `--recreate` rebuilds after adding packages or dependencies. |
| `setup.sh` | `colcon build --symlink-install` inside the container (run it from the Jetson or inside the container). |
| `start.sh` | Open a new shell in the container. |
| `migrate_native_ws.sh` | One-time copy of the native workspace's packages into `src/`. |

Config edits (`vesc.yaml`, `joy_teleop.yaml`) apply on the next launch without rebuilding, because of `--symlink-install`. C++ changes need `bash setup.sh`.

If other cars share your Wi-Fi, give yours its own ROS domain so topics don't mix: `ROS_DOMAIN_ID=42 bash setup_docker.sh --recreate`.
