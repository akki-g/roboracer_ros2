#!/bin/bash
# Start the ROS sim (RViz comes up with the bridge) and drive it with an RL policy.
#
#   tools/sim_gru.sh                         # models/gru_u00100.npz on sim/sim_rl.yaml (Duct1077)
#   tools/sim_gru.sh gru_u00150.npz          # another file in src/rl_policy/models (or a full path)
#   tools/sim_gru.sh gru_u00100.npz 90       # also run the lap monitor for 90 s, print laps/crashes, then stop
#   MAP=EvalDuct3012 tools/sim_gru.sh        # a never-trained map: sim/sim_rl_<MAP>.yaml (tools/sim_gru.sh --maps lists them)
#   SIM_CONFIG=sim/sim.yaml tools/sim_gru.sh # any sim config (its map needs a *_centerline.csv for the monitor)
#
# Needs: ROS 2 Humble and the sim workspace sourced, and `colcon build --packages-select rl_policy`
# done in this repo (the script builds it if the install is missing). Ctrl-C stops everything.
set -eo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
if [ "${1:-}" = "--maps" ]; then
  echo "Duct1077 (default: held-out, used for checkpoint selection)"
  ls sim/sim_rl_*.yaml | sed 's|sim/sim_rl_||; s|\.yaml||' | sed 's/^/  /'
  echo "start one with  MAP=<name> tools/sim_gru.sh [policy.npz] [seconds]"
  exit 0
fi
POLICY="${1:-gru_u00100.npz}"
MONITOR_S="${2:-}"
if [ -n "${MAP:-}" ]; then
  SIM_CONFIG="sim/sim_rl_${MAP}.yaml"
  [ -f "$SIM_CONFIG" ] || { echo "no $SIM_CONFIG; tools/sim_gru.sh --maps lists the maps" >&2; exit 1; }
fi
SIM_CONFIG="${SIM_CONFIG:-sim/sim_rl.yaml}"

if [ -f /.dockerenv ]; then INSTALL=install_docker; else INSTALL=install; fi
if [ ! -f "$INSTALL/rl_policy/share/rl_policy/package.xml" ]; then
  echo "building rl_policy into $INSTALL/ ..."
  colcon build --packages-select rl_policy >/dev/null
fi
# shellcheck disable=SC1090
source "$INSTALL/setup.bash"
command -v ros2 >/dev/null || { echo "ros2 not found: source /opt/ros/humble/setup.bash and the sim workspace first" >&2; exit 1; }

case "$POLICY" in
  /*) POLICY_PATH="$POLICY" ;;
  *)  POLICY_PATH="$(ros2 pkg prefix rl_policy)/share/rl_policy/models/$POLICY" ;;
esac
[ -f "$POLICY_PATH" ] || { echo "policy not found: $POLICY_PATH (rebuild rl_policy after adding a model)" >&2; exit 1; }
MAP=$(sed -n "s/^ *map_path: *['\"]\{0,1\}\([^'\"]*\)['\"]\{0,1\}.*/\1/p" "$SIM_CONFIG" | head -1)
CENTERLINE="$(echo "$MAP" | sed "s|/root/ros2_ws|$WS|")_centerline.csv"

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
  sleep 2
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

LOG_DIR="${TMPDIR:-/tmp}/sim_gru_logs"; mkdir -p "$LOG_DIR"
echo "sim: $SIM_CONFIG   policy: $POLICY_PATH   logs: $LOG_DIR"
setsid ./start_sim.sh "$SIM_CONFIG" > "$LOG_DIR/sim.log" 2>&1 &
PIDS+=($!)
# first start after boot can be slow (ROS discovery daemon, numba compiling the classic gym)
for i in $(seq 1 180); do
  ros2 topic list 2>/dev/null | grep -q ego_racecar/odom && break
  [ $((i % 30)) -eq 0 ] && echo "  waiting for the sim ($i s)..."
  sleep 1
done
ros2 topic list 2>/dev/null | grep -q ego_racecar/odom || { echo "sim did not come up; see $LOG_DIR/sim.log" >&2; exit 1; }
echo "sim up"
setsid ros2 launch rl_policy policy.launch.py mode:=sim policy_path:="$POLICY_PATH" > "$LOG_DIR/policy.log" 2>&1 &
PIDS+=($!)
sleep 4
grep -E "loaded|resampling|refus|error" "$LOG_DIR/policy.log" | sed 's/^/  /' | cut -c1-180 || true

if [ -n "$MONITOR_S" ]; then
  if [ -f "$CENTERLINE" ]; then
    echo "monitoring for ${MONITOR_S}s ..."
    ros2 run rl_policy stage0_monitor --ros-args -p centerline:="$CENTERLINE" -p duration:="$MONITOR_S.0"
  else
    echo "no centerline at $CENTERLINE: skipping the monitor" >&2
  fi
else
  echo "driving; watch RViz. Ctrl-C to stop.   debug: ros2 topic echo /policy_node/debug"
  wait
fi
