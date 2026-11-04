#!/usr/bin/env bash
# Start the standalone FTG diagnostic overlay and its RViz layout.
set -eo pipefail

WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZER="$WORKSPACE_DIR/src/gap_follow/scripts/gap_follow_visualizer.py"
RVIZ_CONFIG="$WORKSPACE_DIR/src/gap_follow/config/live_monitor.rviz"

usage() {
  cat <<'EOF'
Usage: ./start_rviz_ftg.sh [monitor options] [-- rviz2 options]

Starts the read-only FTG monitor and RViz. Run hardware bringup and the
standalone planner separately. Closing either process stops both.

Monitor options include:
  --output-file PATH   Append diagnostic summaries to a JSONL file.
  --duration SECONDS   Stop after this duration (default: run until closed).
  --wheelbase METRES   Match a planner wheelbase override (default: 0.33).
  --front METRES       Front of padded footprint (default: 0.43).
  --rear METRES        Rear of padded footprint (default: 0.12).
  --half-width METRES  Padded footprint half-width (default: 0.25).

Uses ROS 2 Humble and this workspace's install/setup.bash automatically.
ROS_DOMAIN_ID defaults to 42; set it beforehand to use another domain.
A graphical desktop (DISPLAY or WAYLAND_DISPLAY) is required.

Examples:
  ./start_rviz_ftg.sh
  ROS_DOMAIN_ID=7 ./start_rviz_ftg.sh --output-file /tmp/ftg-monitor.jsonl
  ./start_rviz_ftg.sh -- -f base_link
EOF
}

monitor_args=()
rviz_args=()
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --) shift; rviz_args=("$@"); break ;;
    *) monitor_args+=("$1"); shift ;;
  esac
done

for file in /opt/ros/humble/setup.bash "$WORKSPACE_DIR/install/setup.bash" \
            "$VISUALIZER" "$RVIZ_CONFIG"; do
  if [[ ! -f "$file" ]]; then
    printf 'ERROR: Required file not found: %s\n' "$file" >&2
    exit 1
  fi
done

# ROS/colcon setup scripts reference unset variables; enable nounset afterward.
source /opt/ros/humble/setup.bash
source "$WORKSPACE_DIR/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

for program in python3 rviz2; do
  if ! command -v "$program" >/dev/null 2>&1; then
    printf 'ERROR: %s is unavailable after sourcing ROS and the workspace.\n' "$program" >&2
    exit 1
  fi
done
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo 'ERROR: No graphical display. Run from a desktop terminal or an SSH session with X forwarding.' >&2
  exit 1
fi

child_pids=()
cleanup() {
  trap - EXIT INT TERM
  if (( ${#child_pids[@]} )); then
    kill -TERM "${child_pids[@]}" 2>/dev/null || true
    wait "${child_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Starting FTG monitor and RViz on ROS_DOMAIN_ID=%s. Close RViz or press Ctrl+C to stop.\n' "$ROS_DOMAIN_ID"
python3 "$VISUALIZER" "${monitor_args[@]}" &
child_pids+=("$!")
rviz2 -d "$RVIZ_CONFIG" "${rviz_args[@]}" &
child_pids+=("$!")

# Exit when either child finishes, preserving failures and cleaning up its peer.
status=0
wait -n "${child_pids[@]}" || status=$?
exit "$status"
