#!/bin/bash
# Start the F1TENTH ROS sim (f1tenth_gym_ros). Optional argument: the sim
# config to use (default ./sim/sim.yaml), e.g. ./start_sim.sh sim/sim_rl.yaml
#
# gym_bridge_launch.py reads config/sim.yaml from the package's *install*
# share directory, and /sim_ws is not a symlink install, so the config must
# be copied there (copying into src/ alone has no effect).
set -e
CONFIG="${1:-./sim/sim.yaml}"
SHARE="$(ros2 pkg prefix f1tenth_gym_ros)/share/f1tenth_gym_ros"
cp "$CONFIG" "$SHARE/config/sim.yaml"
cp "$CONFIG" /sim_ws/src/f1tenth_gym_ros/config/sim.yaml
cd "$SHARE/launch"
ros2 launch gym_bridge_launch.py
