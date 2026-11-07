#!/bin/bash
# Run the ROS sim with an RL policy in the dev container, with RViz as a normal window, from
#   - a Linux desktop (Ubuntu laptop or desktop, or the VMware VM), or
#   - WSL 2 on Windows (the Ubuntu terminal; tools/sim_gru_windows.ps1 starts this from PowerShell).
# Same arguments as tools/sim_gru.sh, which this runs inside the container:
#
#   tools/sim_gru_linux.sh                                     # gru_u02050.npz on Duct1077, until Ctrl-C
#   tools/sim_gru_linux.sh mlp3_u02050.npz 90                  # another policy; lap monitor for 90 s, then stop
#   MAP=EvalDuct3012 tools/sim_gru_linux.sh gru_u02050.npz 90  # a map never used in training
#   tools/sim_gru_linux.sh --maps                              # list the maps
#   MAP=LastRace2 tools/sim_gru_linux.sh gru1025_solo_u0950.npz              # the last race's track
#   MAP=LastRace2Obs tools/sim_gru_linux.sh gru1025_obs_u0750.npz            # ... with parked cars
#   MAP=LastRace2 OPP_SPEED=1.0-4.0 tools/sim_gru_linux.sh gru1025_opp_u0750.npz  # ... with an opponent car
#
# Needs Docker with the compose plugin. Linux: Docker Engine, with your user in the docker group.
# Windows: Docker Desktop (WSL 2 engine) with WSL integration on for this distro.
# The first run builds the image, which takes 10-20 minutes. Ctrl-C stops everything.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

die() { echo "error: $*" >&2; exit 1; }

if [ "${1:-}" = "--maps" ]; then exec bash tools/sim_gru.sh --maps; fi

WSL=
if [ -n "${WSL_DISTRO_NAME:-}" ] || grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then WSL=1; fi

if ! command -v docker >/dev/null; then
  [ -z "$WSL" ] || die "docker not found in this WSL distro: Docker Desktop > Settings > Resources > WSL integration, turn on ${WSL_DISTRO_NAME:-this distro}"
  die "docker not found: install Docker Engine (curl -fsSL https://get.docker.com | sh)"
fi
docker compose version >/dev/null 2>&1 ||
  die "the docker compose plugin is missing: sudo apt install docker-compose-plugin (Docker's apt repo) or docker-compose-v2 (Ubuntu's)"
if ! err=$(docker info 2>&1 >/dev/null); then
  case "$err" in
    *"permission denied"*) die "your user can't use Docker: sudo usermod -aG docker \$USER, then log out and back in" ;;
  esac
  [ -z "$WSL" ] || die "Docker is not reachable: start Docker Desktop and wait for 'Engine running'"
  die "the Docker daemon is not running: sudo systemctl start docker"
fi

EXEC_ENV=(-e "MAP=${MAP:-}" -e "SIM_CONFIG=${SIM_CONFIG:-}" -e "OPP_SPEED=${OPP_SPEED:-}" -e "OPP_GAP=${OPP_GAP:-}"
          -e "OPP_PERIOD=${OPP_PERIOD:-}")
if [ -n "$WSL" ]; then
  [ -S /mnt/wslg/.X11-unix/X0 ] ||
    die "WSLg is missing (no /mnt/wslg/.X11-unix/X0). It needs Windows 11, or Windows 10 with WSL from the Microsoft Store: in PowerShell run  wsl --update  then  wsl --shutdown, and try again"
  case "$PWD" in
    /mnt/[a-z]/*) echo "note: the repo is on a Windows drive; a clone inside WSL (~/roboracer_ros2) runs faster" >&2 ;;
  esac
  COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.wsl.yml)
else
  [ -n "${DISPLAY:-}" ] || die "DISPLAY is not set: run this from a terminal on the Linux desktop (not over ssh)"
  case "$DISPLAY" in
    :*) n="${DISPLAY#:}"; n="${n%%.*}"
        [ -S "/tmp/.X11-unix/X$n" ] || echo "warning: no X socket /tmp/.X11-unix/X$n; RViz may not open" >&2 ;;
    *) die "DISPLAY=$DISPLAY is not this machine's own display; run this from a terminal on its desktop" ;;
  esac
  [ -e /dev/dri ] || die "no /dev/dri (no graphics driver loaded); docker-compose.linux.yml passes it to the container"
  [ "$(docker context show 2>/dev/null)" != "desktop-linux" ] ||
    die "Docker Desktop for Linux can't share this desktop's X display; use Docker Engine: docker context use default"
  if [ -e /proc/driver/nvidia/version ]; then
    # the container's Mesa can't use NVIDIA's proprietary driver: RViz renders on the CPU
    EXEC_ENV+=(-e "LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}")
  fi
  if command -v xhost >/dev/null; then
    xhost +local: >/dev/null   # let the container's RViz use this desktop's X server
  else
    echo "warning: xhost not found (sudo apt install x11-xserver-utils); RViz may not open" >&2
  fi
  COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.linux.yml)
fi

"${COMPOSE[@]}" up -d dev
TTY=()
{ [ -t 0 ] && [ -t 1 ]; } || TTY=(-T)
# bash -i: the container's .bashrc sources ROS, /sim_ws and install_docker. The rebuild
# installs any policy file added to src/rl_policy/models since the last build.
exec "${COMPOSE[@]}" exec "${TTY[@]}" "${EXEC_ENV[@]}" dev bash -ic \
  "colcon build --packages-select rl_policy >/dev/null && source install_docker/setup.bash && bash tools/sim_gru.sh $*"
