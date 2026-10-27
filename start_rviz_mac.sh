#!/usr/bin/env bash
# RoboRacer RViz startup for the *Mac's ROS dev container* (not the Jetson).
# This launches RViz on a virtual Xvfb desktop, NOT on XQuartz.
# In VS Code Ports forward container 5910 -> Mac localhost 5901; then on Mac:
#   open vnc://127.0.0.1:5901
# Usage: ./start_rviz_mac.sh [rviz2 args, e.g. -d /path/to/layout.rviz]
set -Eeuo pipefail

# Only source ROS if the current shell has not already done so.
if ! command -v rviz2 >/dev/null 2>&1 && [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi

DISPLAY_ID=':99'
VNC_PORT=5910
MAC_PORT=5901
LOG_DIR="${HOME}/.cache/roboracer"
PASS_DIR="${HOME}/.config/roboracer"
PASS_FILE="${PASS_DIR}/rviz-vnc.pass"
PID_FILE="${LOG_DIR}/rviz.pid"

# Prevent XQuartz from leaking into the Xvfb/Mesa environment.
export DISPLAY="$DISPLAY_ID"
export LIBGL_ALWAYS_SOFTWARE=1
unset LIBGL_ALWAYS_INDIRECT || true
export QT_X11_NO_MITSHM=1

umask 077
mkdir -p "$LOG_DIR" "$PASS_DIR"
chmod 700 "$PASS_DIR"

for program in Xvfb x11vnc glxinfo python3 rviz2; do
  if ! command -v "$program" >/dev/null 2>&1; then
    echo "ERROR: Missing $program inside the Mac ROS dev container." >&2
    echo 'Install: apt-get update && apt-get install -y xvfb x11vnc mesa-utils libgl1-mesa-dri' >&2
    exit 1
  fi
done

# Test the actual rendering connection, rather than just the process name.
open_gl_check() {
  env DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 glxinfo -B >/dev/null 2>&1
}

printf '\nRoboRacer RViz Mac startup (v3)\n'
if open_gl_check; then
  echo 'Xvfb :99 already available.'
else
  echo 'Starting Xvfb :99...'
  nohup Xvfb :99 -screen 0 1600x900x24 +extension GLX -noreset \
    >"${LOG_DIR}/xvfb.log" 2>&1 </dev/null &
  for ((attempt=0; attempt<40; attempt++)); do
    if open_gl_check; then break; fi
    sleep 0.25
  done
  if ! open_gl_check; then
    echo 'ERROR: Xvfb or OpenGL failed. Xvfb log:' >&2
    tail -n 30 "${LOG_DIR}/xvfb.log" >&2
    exit 1
  fi
fi

GL_INFO="$(env DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 glxinfo -B 2>&1)"
printf '%s\n' "$GL_INFO" | grep -E '^OpenGL renderer string:|^OpenGL version string:' || true
if ! grep -Eiq '^OpenGL renderer string:[[:space:]]*(llvmpipe|softpipe)' <<<"$GL_INFO"; then
  echo 'ERROR: Expected Mesa software OpenGL on :99. Aborting rather than using XQuartz.' >&2
  exit 1
fi

if [[ ! -s "$PASS_FILE" ]]; then
  if [[ -s /tmp/rviz-vnc.pass ]]; then
    cp /tmp/rviz-vnc.pass "$PASS_FILE"
    echo 'Reused your existing VNC password.'
  else
    echo 'Set a password for Screen Sharing (first startup only):'
    x11vnc -storepasswd "$PASS_FILE"
  fi
fi
chmod 600 "$PASS_FILE"

# Verify the RFB protocol, not just whether the TCP port accepts a connection.
vnc_ready() {
  python3 -c 'import socket,sys
try:
 s=socket.create_connection(("127.0.0.1",int(sys.argv[1])),timeout=0.6)
 s.settimeout(0.6)
 banner=s.recv(12)
 s.close()
 sys.exit(0 if banner.startswith(b"RFB ") else 1)
except OSError:
 sys.exit(1)' "$VNC_PORT" >/dev/null 2>&1
}

# Never silently use a VNC server attached to the wrong X display.
expected_vnc_process() {
  pgrep -af x11vnc 2>/dev/null | grep -F -- '-display :99' | grep -F -- '-rfbport 5910' >/dev/null
}

if vnc_ready; then
  if ! expected_vnc_process; then
    echo 'ERROR: Port 5910 is serving VNC, but not by our expected Xvfb :99 process.' >&2
    echo 'Inspect it with: pgrep -af x11vnc' >&2
    exit 1
  fi
  echo 'VNC server already running on container port 5910.'
else
  echo 'Starting password-protected VNC server on container port 5910...'
  nohup x11vnc -display :99 -localhost -forever -shared -noshm \
    -rfbauth "$PASS_FILE" -rfbport "$VNC_PORT" \
    >"${LOG_DIR}/x11vnc.log" 2>&1 </dev/null &
  for ((attempt=0; attempt<40; attempt++)); do
    if vnc_ready; then break; fi
    sleep 0.25
  done
  if ! vnc_ready; then
    echo 'ERROR: VNC server did not start; log:' >&2
    tail -n 30 "${LOG_DIR}/x11vnc.log" >&2
    exit 1
  fi
fi

# RViz is launched in the background so the script returns and prints guidance.
if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "RViz already running (PID $(cat "$PID_FILE"))."
else
  echo 'Launching RViz on virtual display :99...'
  nohup env DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 \
    QT_X11_NO_MITSHM=1 rviz2 "$@" \
    >"${LOG_DIR}/rviz.log" 2>&1 </dev/null &
  rviz_pid=$!
  echo "$rviz_pid" > "$PID_FILE"
  sleep 2
  if ! kill -0 "$rviz_pid" 2>/dev/null; then
    echo 'ERROR: RViz exited immediately. Recent log:' >&2
    tail -n 35 "${LOG_DIR}/rviz.log" >&2
    exit 1
  fi
  echo "RViz process running (PID $rviz_pid)."
fi

cat <<EOF

NEXT STEP ON THE MAC (not inside Docker):
  1. In VS Code Ports, forward container 5910 to Mac localhost 5901.
  2. In the normal Mac terminal, run:
       open vnc://127.0.0.1:${MAC_PORT}

RViz's Linux desktop is invisible until you open Mac Screen Sharing.
Logs are in ${LOG_DIR}/ (rviz.log, x11vnc.log, xvfb.log).
EOF
