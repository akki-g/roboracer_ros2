#!/bin/bash
# Start the F1TENTH ROS sim (f1tenth_gym_ros), in the Docker dev container or
# on a native ROS 2 Humble install (e.g. an Ubuntu VM). Optional argument: the
# sim config (default sim/sim.yaml), e.g.  ./start_sim.sh sim/sim_rl.yaml
#
# gym_bridge_launch.py reads config/sim.yaml from the package's *install*
# share directory, so the chosen config is written there (copying it into
# src/ alone has no effect without a symlink install). Configs use the
# container's paths; they are rewritten for this machine:
#   /root/ros2_ws                 -> this repo
#   /sim_ws/src/f1tenth_gym_ros   -> the sim package's source (found in
#                                    /sim_ws or ~/sim_ws, or set GYM_ROS_SRC)
# ROS 2 and the sim workspace must already be sourced.
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$(realpath "${1:-$WS/sim/sim.yaml}")"

# this workspace's overlay: install_docker/ in the container, install/ natively
if [ -f /.dockerenv ]; then OVERLAY="$WS/install_docker/setup.bash"; else OVERLAY="$WS/install/setup.bash"; fi
if [ -f "$OVERLAY" ]; then source "$OVERLAY"; fi

command -v ros2 >/dev/null || { echo "ros2 not found: source /opt/ros/humble/setup.bash and the sim workspace" >&2; exit 1; }
SHARE="$(ros2 pkg prefix f1tenth_gym_ros)/share/f1tenth_gym_ros"

GYM_SRC="${GYM_ROS_SRC:-}"
if [ -z "$GYM_SRC" ]; then
  for d in /sim_ws/src/f1tenth_gym_ros "$HOME/sim_ws/src/f1tenth_gym_ros"; do
    if [ -d "$d" ]; then GYM_SRC="$d"; break; fi
  done
fi

SED=(-e "s|/root/ros2_ws|$WS|g")
if [ -n "$GYM_SRC" ]; then SED+=(-e "s|/sim_ws/src/f1tenth_gym_ros|$GYM_SRC|g"); fi
sed "${SED[@]}" "$CONFIG" > "$SHARE/config/sim.yaml"
if [ -n "$GYM_SRC" ] && ! [ "$SHARE/config/sim.yaml" -ef "$GYM_SRC/config/sim.yaml" ]; then
  cp "$SHARE/config/sim.yaml" "$GYM_SRC/config/sim.yaml"
fi

# RViz layout: upstream's draws only the ego car; with two cars use sim/gym_bridge_two_cars.rviz,
# which also draws the opponent (the original is kept as gym_bridge.rviz.one_car and put back
# for one-car configs)
RVIZ="$SHARE/launch/gym_bridge.rviz"
if [ -f "$RVIZ" ] && [ ! -f "$RVIZ.one_car" ]; then cp "$RVIZ" "$RVIZ.one_car"; fi
if grep -q "^ *num_agent: *2" "$SHARE/config/sim.yaml"; then
  cp --remove-destination "$WS/sim/gym_bridge_two_cars.rviz" "$RVIZ"
elif [ -f "$RVIZ.one_car" ]; then
  cp --remove-destination "$RVIZ.one_car" "$RVIZ"
fi

MAP="$(sed -n "s/^ *map_path: *['\"]\{0,1\}\([^'\"]*\)['\"]\{0,1\}.*/\1/p" "$SHARE/config/sim.yaml" | head -1)"
if [ ! -f "$MAP.yaml" ]; then
  echo "map not found: $MAP.yaml (from $CONFIG; set GYM_ROS_SRC if the sim package's source is elsewhere)" >&2
  exit 1
fi
echo "sim config: $CONFIG  map: $MAP"
exec ros2 launch f1tenth_gym_ros gym_bridge_launch.py
