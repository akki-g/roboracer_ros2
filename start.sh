#!/usr/bin/env bash
# Open a new shell inside the car's container, with ROS and the workspace sourced.
# Each SSH session or teammate gets its own shell, all in the same container.
set -euo pipefail

NAME="roboracer"
DOCKER=(docker)
docker info >/dev/null 2>&1 || DOCKER=(sudo docker)

if ! "${DOCKER[@]}" container inspect "$NAME" >/dev/null 2>&1; then
  echo "No '$NAME' container yet. Run: bash setup_docker.sh" >&2
  exit 1
fi
"${DOCKER[@]}" start "$NAME" >/dev/null
exec "${DOCKER[@]}" exec -it "$NAME" bash
