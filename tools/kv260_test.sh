#!/bin/bash
# Policy test on the Kria KV260 (any Ubuntu 22.04 car computer). No car hardware is needed and nothing
# reaches the motor: the ROS part uses its own topics under /board_test, not /scan, /odom or /drive.
#
#   tools/car_native.sh setup      # once (sudo): ROS 2 Humble if missing, dependencies, build
#   tools/kv260_test.sh            # the test; exit code 0 = PASS
#
#   1. same actions as the Jetson   20 s of a recorded drive (Nov 7, on the car) go through the policy;
#                                   the actions must match the ones the Jetson computed
#   2. timing                       one policy tick's work, 5000 times
#   3. ROS loop                     the real policy_node on the recorded scans at 40 Hz for 20 s: command
#                                   rate, late commands, tick time, CPU use
#
#   tools/kv260_test.sh --no-ros                  # 1 and 2 only (needs just python3-numpy)
#   tools/kv260_test.sh gru1025_obs_u0750.npz     # time another policy in src/rl_policy/models (2 and 3)
#   SECONDS_ROS=60 tools/kv260_test.sh            # a longer ROS loop
set -eo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"

ROS=1
POLICY=""
for arg in "$@"; do
  case "$arg" in
    --no-ros) ROS=0 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) POLICY="$arg" ;;
  esac
done
if [ -n "$POLICY" ] && [ ! -f "$POLICY" ]; then
  [ -f "src/rl_policy/models/$POLICY" ] || { echo "error: no policy $POLICY (looked in src/rl_policy/models)" >&2; exit 2; }
  POLICY="$WS/src/rl_policy/models/$POLICY"
fi

command -v python3 >/dev/null || { echo "error: no python3" >&2; exit 2; }
python3 -c "import numpy" 2>/dev/null ||
  { echo "error: python3 has no numpy. Run  tools/car_native.sh setup  (or just: sudo apt-get install -y python3-numpy)" >&2; exit 2; }

fail=0
(cd src/rl_policy && python3 -m rl_policy.board_test ${POLICY:+--policy "$POLICY"}) || fail=1

if [ "$ROS" = 1 ]; then
  if [ -f /.dockerenv ] && [ -d install_docker ]; then INSTALL=install_docker; else INSTALL=install; fi
  if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo "== 3. ROS loop: SKIPPED, no ROS 2 Humble on this board. Run  tools/car_native.sh setup  then this again."
    fail=1
  elif [ ! -f "$INSTALL/rl_policy/share/rl_policy/package.xml" ]; then
    echo "== 3. ROS loop: SKIPPED, the workspace is not built. Run  tools/car_native.sh build  then this again."
    fail=1
  else
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    # shellcheck disable=SC1090
    source "$INSTALL/setup.bash"
    if [ ! -f "$(ros2 pkg prefix rl_policy)/lib/rl_policy/board_loop_test" ]; then
      echo "== 3. ROS loop: SKIPPED, the build is older than this checkout. Run  tools/car_native.sh build  then this again."
      fail=1
    else
      ros2 run rl_policy board_loop_test --seconds "${SECONDS_ROS:-20}" ${POLICY:+--policy "$POLICY"} || fail=1
    fi
  fi
fi

echo
if [ "$fail" = 0 ]; then echo "KV260 policy test: PASS"; else echo "KV260 policy test: FAIL (see above)"; fi
exit "$fail"
