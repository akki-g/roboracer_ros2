#!/usr/bin/env bash
# Build the ROS 2 workspace. Works from the Jetson or from a shell inside the container.
# Re-run it after changing C++ code or adding packages. With --symlink-install,
# edits to Python, launch and config files (vesc.yaml, joy_teleop.yaml) apply
# without rebuilding.
set -euo pipefail

NAME="roboracer"
BUILD='source /opt/ros/humble/setup.bash && cd /root/ros2_ws && colcon build --symlink-install'

if [ -f /.dockerenv ]; then
  bash -c "$BUILD"
  echo "Build done. In this shell run: source install/setup.bash"
else
  DOCKER=(docker)
  docker info >/dev/null 2>&1 || DOCKER=(sudo docker)
  if ! "${DOCKER[@]}" container inspect "$NAME" >/dev/null 2>&1; then
    echo "No '$NAME' container yet. Run: bash setup_docker.sh" >&2
    exit 1
  fi
  "${DOCKER[@]}" start "$NAME" >/dev/null
  "${DOCKER[@]}" exec "$NAME" bash -c "$BUILD"
  echo "Build done. Open a shell with: bash start.sh"
fi
