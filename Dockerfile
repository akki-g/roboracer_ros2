# Two images from one file:
#   car - driver stack + your packages' dependencies (what runs on the Jetson)
#   dev - car + the F1TENTH simulator (for laptops)
FROM ros:humble-ros-base AS car

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# asio_cmake_module: rosdep misses it for vesc_driver
# foxglove_bridge:   lets Foxglove on a laptop show topics over Wi-Fi
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-pip \
        ros-humble-asio-cmake-module \
        ros-humble-foxglove-bridge \
        git nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/ros2_ws

# Install every dependency listed in src/**/package.xml. The code itself is
# mounted at runtime, so rebuild the image only when dependencies change:
#   docker compose up -d --build <car|dev>
COPY src/ src/
RUN apt-get update \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

# Plain `colcon build` now means: symlink install (config/launch/Python edits
# apply without rebuilding), into *_docker folders so a build made outside
# Docker is never overwritten.
RUN mkdir -p /root/.colcon && printf '%s\n' \
      'build: {symlink-install: true, build-base: build_docker, install-base: install_docker}' \
      'test: {build-base: build_docker, install-base: install_docker}' \
      'test-result: {test-result-base: build_docker}' \
      > /root/.colcon/defaults.yaml
ENV COLCON_LOG_PATH=/root/ros2_ws/log_docker

# Every shell: ROS, then the simulator (dev image only), then your workspace.
RUN printf '%s\n' \
      'source /opt/ros/humble/setup.bash' \
      '[ -f /sim_ws/install/setup.bash ] && source /sim_ws/install/setup.bash' \
      '[ -f /root/ros2_ws/install_docker/setup.bash ] && source /root/ros2_ws/install_docker/setup.bash' \
      >> /root/.bashrc

CMD ["sleep", "infinity"]


FROM car AS dev

# Simulator physics (Python). Upstream targets Foxy; it runs on Humble.
RUN git clone --depth 1 https://github.com/f1tenth/f1tenth_gym /opt/f1tenth_gym \
    && pip3 install --no-cache-dir -e /opt/f1tenth_gym transforms3d

# Simulator's ROS 2 bridge, built in /sim_ws so the mounted workspace can't hide it.
RUN mkdir -p /sim_ws/src \
    && git clone --depth 1 https://github.com/f1tenth/f1tenth_gym_ros /sim_ws/src/f1tenth_gym_ros \
    && apt-get update \
    && rosdep install --from-paths /sim_ws/src --ignore-src -y --rosdistro humble \
         --skip-keys python-transforms3d-pip \
    && rm -rf /var/lib/apt/lists/* \
    && source /opt/ros/humble/setup.bash \
    && cd /sim_ws \
    && COLCON_HOME=/nonexistent COLCON_LOG_PATH=/sim_ws/log colcon build

RUN apt update