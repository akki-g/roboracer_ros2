#!/bin/bash
# tools/car_tests.sh <grip|straight|steer|policy-sweep> [--help | options]
set -eo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
die() { echo "error: $*" >&2; exit 1; }
if [ "${1:-}" = "--help" ] || [ "${2:-}" = "--help" ]; then
  PYTHONPATH="$WS/src/car_tests${PYTHONPATH:+:$PYTHONPATH}" python3 -m car_tests.runner "$@"
  exit
fi
[ -f /opt/ros/humble/setup.bash ] || die "ROS 2 Humble is missing; run tools/car_native.sh setup on the car"
source /opt/ros/humble/setup.bash
[ -f "$WS/install/setup.bash" ] || die "workspace is not built; run tools/car_native.sh build"
source "$WS/install/setup.bash"
ros2 pkg prefix car_tests >/dev/null || die "car_tests is not installed; run tools/car_native.sh build"
exec python3 -m car_tests.runner "$@"
