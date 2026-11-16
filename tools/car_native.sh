#!/bin/bash
# Run the car stack on the car's computer (the Jetson or the Kria KV260, Ubuntu 22.04) without Docker,
# on a native ROS 2 Humble install. The native build goes to build/ install/ log/. The container's
# build_docker/ install_docker/ log_docker/ are left alone, so the two can coexist.
#
#   tools/car_native.sh setup      # once, and when a package.xml gains a dependency (uses sudo):
#                                  # ROS 2 Humble if it is missing, dependencies, the VESC device rule,
#                                  # groups, build, checks
#   tools/car_native.sh build      # after every git pull
#   tools/car_native.sh check      # VESC, joystick, LiDAR, groups, the car container
#   tools/car_native.sh lidar-net [interface]   # put this computer on the LiDAR's subnet (uses sudo)
#
# Then, in each terminal (or put both lines in ~/.bashrc):
#   source /opt/ros/humble/setup.bash && source <this repo>/install/setup.bash
set -eo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
USER="${USER:-$(id -un)}"
LIDAR_HOST_ADDR="${LIDAR_HOST_ADDR:-192.168.0.15/24}"  # this computer's address on the LiDAR's Ethernet link

die() { echo "error: $*" >&2; exit 1; }
ok() { echo "  ok    $*"; }
warn() { echo "  WARN  $*"; }
if [ "$(id -u)" = 0 ] && ! command -v sudo >/dev/null; then sudo() { "$@"; }; fi  # root without sudo (containers)
apt_install() { sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"; }

ros_env() {
  [ -f /opt/ros/humble/setup.bash ] || die "no ROS 2 Humble in /opt/ros/humble: run  tools/car_native.sh setup  (it installs it)"
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
}

lidar_ip() {
  sed -n 's/^ *ip_address: *"\{0,1\}\([0-9.]*\)"\{0,1\}.*/\1/p' src/f1tenth_system/f1tenth_stack/config/sensors.yaml | head -1
}

install_ros() {
  # shellcheck disable=SC1091
  . /etc/os-release
  [ "$VERSION_ID" = "22.04" ] || die "ROS 2 Humble's apt packages need Ubuntu 22.04; this is $PRETTY_NAME"
  echo "== ROS 2 Humble (ros-base) from packages.ros.org: about 1 GB, 10-20 minutes on the KV260"
  sudo apt-get update
  apt_install curl ca-certificates gnupg locales software-properties-common
  if ! locale 2>/dev/null | grep -qi 'utf-\{0,1\}8'; then
    sudo locale-gen en_US.UTF-8 && sudo update-locale LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
  fi
  sudo add-apt-repository -y universe
  local ver deb=/tmp/ros2-apt-source.deb
  ver="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
    sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)" || true
  if [ -n "$ver" ] && curl -fsSL -o "$deb" \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.${UBUNTU_CODENAME}_all.deb"; then
    sudo dpkg -i "$deb"  # ROS's apt source and signing key, kept up to date by apt
  else
    echo "  ros2-apt-source not reachable: adding the ROS apt key and source by hand"
    sudo curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $UBUNTU_CODENAME main" |
      sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
  fi
  sudo apt-get update
  apt_install ros-humble-ros-base
  [ -f /opt/ros/humble/setup.bash ] || die "ros-humble-ros-base did not install: see the apt output above"
}

vesc_rule() {
  echo "== udev: /dev/sensors/vesc (the port in vesc.yaml)"
  if grep -rqs 'sensors/vesc' /etc/udev/rules.d /lib/udev/rules.d; then ok "a rule for /dev/sensors/vesc is installed"; return; fi
  sudo mkdir -p /etc/udev/rules.d
  echo 'KERNEL=="ttyACM[0-9]*", ACTION=="add", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0666", GROUP="dialout", SYMLINK+="sensors/vesc"' |
    sudo tee /etc/udev/rules.d/99-vesc.rules >/dev/null
  if command -v udevadm >/dev/null; then sudo udevadm control --reload-rules 2>/dev/null && sudo udevadm trigger 2>/dev/null || true; fi
  echo "  added /etc/udev/rules.d/99-vesc.rules: replug the VESC's USB cable if it is connected"
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

has_driver() {  # a kernel driver, as a module or built in
  modinfo "$1" >/dev/null 2>&1 || grep -qs "/${1//_/[_-]}\.ko" "/lib/modules/$(uname -r)/modules.builtin"
}

check() {
  echo "== checks"
  if [ -e /dev/sensors/vesc ]; then
    dev="$(readlink -f /dev/sensors/vesc)"
    if [ -r "$dev" ] && [ -w "$dev" ]; then ok "VESC: /dev/sensors/vesc -> $dev"
    else warn "VESC at $dev, but $USER can't open it: needs the dialout group (setup adds it; log out and back in)"; fi
  elif ! has_driver cdc_acm; then
    warn "no /dev/sensors/vesc, and this kernel ($(uname -r)) has no cdc_acm driver for the VESC's USB serial port"
  else
    warn "no /dev/sensors/vesc (vesc.yaml's port): the VESC is unplugged or its udev rule is missing (setup adds it)"
  fi
  if compgen -G "/dev/input/js*" >/dev/null; then ok "joystick: $(echo /dev/input/js*)"
  elif ! has_driver joydev; then warn "no /dev/input/js*, and this kernel ($(uname -r)) has no joydev driver for joysticks"
  else warn "no /dev/input/js*: plug in the F710 receiver (mode switch on the back at X)"; fi
  ip="$(lidar_ip)"
  if [ -n "$ip" ]; then
    if ping -c 1 -W 1 "$ip" >/dev/null 2>&1; then ok "LiDAR answers at $ip"
    else warn "LiDAR at $ip doesn't answer: check its Ethernet cable and that this computer has an address on that subnet (tools/car_native.sh lidar-net)"; fi
  fi
  for g in dialout input; do
    if id -nG | tr ' ' '\n' | grep -qx "$g"; then ok "this session is in $g"
    else warn "this session is not in $g yet: log out and back in (setup adds it)"; fi
  done
  if command -v docker >/dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q car; then
    warn "the car container is running. It only sleeps, but a bringup started in it would fight this one for the VESC: docker compose stop car"
  fi
}

lidar_net() {
  local ifc="${1:-}" ip
  ip="$(lidar_ip)"
  if [ -z "$ifc" ]; then  # the first wired interface (the KV260 has one: eth0)
    for d in /sys/class/net/*; do
      n="$(basename "$d")"
      if [ "$n" != lo ] && [ -e "$d/device" ] && [ ! -d "$d/wireless" ]; then ifc="$n"; break; fi
    done
  fi
  [ -n "$ifc" ] && [ -e "/sys/class/net/$ifc" ] || die "no wired network interface found: tools/car_native.sh lidar-net <interface> (see: ip link)"
  echo "== LiDAR link: $LIDAR_HOST_ADDR on $ifc (the LiDAR is at $ip), kept across reboots"
  if [ "${FORCE:-0}" != 1 ] && ip route show default 2>/dev/null | grep -qw "dev $ifc"; then
    die "$ifc is this computer's network connection right now (its default route). Giving it the LiDAR's static address drops SSH and internet on this port. From a keyboard or the serial console: FORCE=1 tools/car_native.sh lidar-net $ifc"
  fi
  if command -v nmcli >/dev/null && systemctl is-active --quiet NetworkManager 2>/dev/null; then
    sudo nmcli connection delete lidar >/dev/null 2>&1 || true
    sudo nmcli connection add type ethernet ifname "$ifc" con-name lidar ipv4.method manual \
      ipv4.addresses "$LIDAR_HOST_ADDR" ipv6.method ignore connection.autoconnect yes connection.autoconnect-priority 100
    sudo nmcli connection up lidar
    echo "  undo (back to DHCP on $ifc): sudo nmcli connection delete lidar"
  elif command -v netplan >/dev/null; then
    printf 'network:\n  version: 2\n  ethernets:\n    %s:\n      dhcp4: false\n      optional: true\n      addresses: [%s]\n' "$ifc" "$LIDAR_HOST_ADDR" |
      sudo tee /etc/netplan/60-lidar.yaml >/dev/null
    sudo chmod 600 /etc/netplan/60-lidar.yaml
    sudo netplan apply
    echo "  undo (back to DHCP on $ifc): sudo rm /etc/netplan/60-lidar.yaml && sudo netplan apply"
  else
    die "neither NetworkManager nor netplan here: set $LIDAR_HOST_ADDR on $ifc by hand"
  fi
  sleep 2
  if ping -c 1 -W 2 "$ip" >/dev/null 2>&1; then ok "LiDAR answers at $ip"
  else warn "LiDAR at $ip doesn't answer yet: is it powered and its cable in $ifc?"; fi
}

setup() {
  # shellcheck disable=SC1091
  . /etc/os-release
  [ "$VERSION_ID" = "22.04" ] || warn "Ubuntu $VERSION_ID: Humble's apt packages are built for 22.04"
  [ -f /opt/ros/humble/setup.bash ] || install_ros
  ros_env
  echo "== apt: build tools and the packages rosdep misses (as in the Dockerfile)"
  sudo apt-get update
  apt_install python3-colcon-common-extensions python3-rosdep python3-pip python3-numpy build-essential git \
    ros-humble-asio-cmake-module ros-humble-foxglove-bridge
  echo "== rosdep: every dependency in src/**/package.xml"
  [ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
  rosdep update --rosdistro humble
  DEBIAN_FRONTEND=noninteractive rosdep install --from-paths src --ignore-src -y --rosdistro humble
  vesc_rule
  echo "== groups: dialout (VESC serial port), input (joystick)"
  for g in dialout input; do
    if ! getent group "$g" >/dev/null; then warn "this system has no $g group"
    elif id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then ok "$USER in $g"
    else sudo usermod -aG "$g" "$USER"; echo "  added $USER to $g: log out and back in before driving"; fi
  done
  build
  check
  echo "setup done. Policy test, no car needed:  tools/kv260_test.sh"
}

case "${1:-}" in
  setup) setup ;;
  build) build ;;
  check) check ;;
  lidar-net) lidar_net "${2:-}" ;;
  *) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
