#!/bin/bash
# Run on the Ubuntu VM (the host, not inside a container): start the dev
# container with your desktop's display (docker-compose.linux.yml) and run
# tools/sim_gru.sh inside it. RViz opens as a normal window. Same arguments:
#
#   tools/sim_gru_vm.sh                              # gru_u02050.npz on Duct1077, until Ctrl-C
#   tools/sim_gru_vm.sh mlp3_u02050.npz 90           # another policy; lap monitor for 90 s, then stop
#   MAP=EvalDuct3012 tools/sim_gru_vm.sh             # a never-trained map (tools/sim_gru.sh --maps lists them)
#
# First run builds the image (docker compose up), which takes a while.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export COMPOSE_FILE=docker-compose.yml:docker-compose.linux.yml
if command -v xhost >/dev/null; then
  xhost +local: >/dev/null   # let the container's RViz use this desktop's X server
else
  echo "xhost not found (sudo apt install x11-xserver-utils): RViz may not open" >&2
fi
docker compose up -d dev
# bash -i: the container's .bashrc sources ROS, /sim_ws and install_docker. The rebuild
# installs any policy file added to src/rl_policy/models since the last build.
docker compose exec -e MAP="${MAP:-}" -e SIM_CONFIG="${SIM_CONFIG:-}" dev bash -ic \
  "colcon build --packages-select rl_policy >/dev/null && source install_docker/setup.bash && tools/sim_gru.sh $*"
