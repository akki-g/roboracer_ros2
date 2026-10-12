#!/usr/bin/env bash
# Build the image and create the long-running container. Run this on the Jetson.
#   bash setup_docker.sh             first-time setup
#   bash setup_docker.sh --recreate  rebuild the image and replace the container
#                                    (after adding packages or dependencies)
set -euo pipefail

IMAGE="roboracer_ros2:humble"
NAME="roboracer"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker isn't installed. Install it with: sudo apt install -y docker.io" >&2
  exit 1
fi
DOCKER=(docker)
docker info >/dev/null 2>&1 || DOCKER=(sudo docker)

if "${DOCKER[@]}" container inspect "$NAME" >/dev/null 2>&1; then
  if [ "${1:-}" = "--recreate" ]; then
    echo "Removing the old '$NAME' container (your code is untouched; it lives in $REPO_DIR)"
    "${DOCKER[@]}" rm -f "$NAME" >/dev/null
  else
    echo "Container '$NAME' already exists. Open a shell with: bash start.sh"
    echo "To rebuild it after changing dependencies: bash setup_docker.sh --recreate"
    exit 0
  fi
fi

mkdir -p "$REPO_DIR/src"
echo "Building image $IMAGE (the first build on the Jetson takes a while)..."
"${DOCKER[@]}" build -t "$IMAGE" "$REPO_DIR"

echo "Creating container $NAME..."
# --privileged + /dev:/dev  VESC (/dev/sensors/vesc), LiDAR and controller, including re-plugs
# /run/udev (read-only)     lets the joy driver discover controllers inside the container
# --network/--ipc host      ROS 2 topics reach the Jetson host and the pit laptop over Wi-Fi
# --restart unless-stopped  container comes back after a reboot (it only sleeps; nothing drives)
"${DOCKER[@]}" run -d --name "$NAME" \
  --restart unless-stopped \
  --privileged --network host --ipc host \
  -v /dev:/dev -v /run/udev:/run/udev:ro \
  -v "$REPO_DIR":/root/ros2_ws -w /root/ros2_ws \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  "$IMAGE" >/dev/null

echo "Done. Next: bash setup.sh   (builds the workspace inside the container)"
