#!/bin/bash
# Run the car stack on the Jetson without Docker, on a native ROS 2 Humble install.
# The native build goes to build/ install/ log/. The container's build_docker/ install_docker/
# log_docker/ are left alone, so the two can coexist.
#
#   tools/car_native.sh setup   # once, and when a package.xml gains a dependency (uses sudo)
#   tools/car_native.sh build   # after every git pull
#   tools/car_native.sh check   # VESC, joystick, LiDAR, groups, the car container
#
# Then, in each terminal (or put both lines in ~/.bashrc):
#   source /opt/ros/humble/setup.bash && source <this repo>/install/setup.bash
set -eo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"

die() { echo "error: $*" >&2; exit 1; }
ok() { echo "  ok    $*"; }
warn() { echo "  WARN  $*"; }

ros_env() {
  [ -f /opt/ros/humble/setup.bash ] ||
    die "no ROS 2 Humble in /opt/ros/humble. It needs Ubuntu 22.04 (JetPack 6): https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html (ros-humble-ros-base)"
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
}

build() {
  ros_env
  for d in build install log; do
    if [ -e "$d" ] && [ ! -w "$d" ]; then die "$WS/$d is not writable (made by root?): sudo chown -R $USER: $WS/$d"; fi
  done
  echo "== colcon build (native: build/ install/ log/)"
  colcon build --symlink-install
  echo "built. In each terminal: source /opt/ros/humble/setup.bash && source $WS/install/setup.bash"
}

check() {
  echo "== checks"
  if [ -e /dev/sensors/vesc ]; then
    dev="$(readlink -f /dev/sensors/vesc)"
    if [ -r "$dev" ] && [ -w "$dev" ]; then ok "VESC: /dev/sensors/vesc -> $dev"
    else warn "VESC at $dev, but $USER can't open it: needs the dialout group (setup adds it; log out and back in)"; fi
  else
    warn "no /dev/sensors/vesc (vesc.yaml's port): the VESC is unplugged or its udev rule is missing"
  fi
  if compgen -G "/dev/input/js*" >/dev/null; then ok "joystick: $(echo /dev/input/js*)"
  else warn "no /dev/input/js*: plug in the F710 receiver (mode switch on the back at X)"; fi
  ip="$(sed -n 's/^ *ip_address: *"\{0,1\}\([0-9.]*\)"\{0,1\}.*/\1/p' src/f1tenth_system/f1tenth_stack/config/sensors.yaml | head -1)"
  if [ -n "$ip" ]; then
    if ping -c 1 -W 1 "$ip" >/dev/null 2>&1; then ok "LiDAR answers at $ip"
    else warn "LiDAR at $ip doesn't answer: check its Ethernet cable and that this Jetson has an address on that subnet"; fi
  fi
  for g in dialout input; do
    if id -nG | tr ' ' '\n' | grep -qx "$g"; then ok "this session is in $g"
    else warn "this session is not in $g yet: log out and back in (setup adds it)"; fi
  done
  if command -v docker >/dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q car; then
    warn "the car container is running. It only sleeps, but a bringup started in it would fight this one for the VESC: docker compose stop car"
  fi
}

setup() {
  ros_env
  # shellcheck disable=SC1091
  . /etc/os-release
  [ "$VERSION_ID" = "22.04" ] || warn "Ubuntu $VERSION_ID: Humble's apt packages are built for 22.04"
  echo "== apt: build tools and the packages rosdep misses (as in the Dockerfile)"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends python3-colcon-common-extensions python3-rosdep python3-pip \
    ros-humble-asio-cmake-module ros-humble-foxglove-bridge
  echo "== rosdep: every dependency in src/**/package.xml"
  [ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
  rosdep update --rosdistro humble
  rosdep install --from-paths src --ignore-src -y --rosdistro humble
  echo "== groups: dialout (VESC serial port), input (joystick)"
  for g in dialout input; do
    if id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then ok "$USER in $g"
    else sudo usermod -aG "$g" "$USER"; echo "  added $USER to $g: log out and back in before driving"; fi
  done
  build
  check
}

case "${1:-}" in
  setup) setup ;;
  build) build ;;
  check) check ;;
  *) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
