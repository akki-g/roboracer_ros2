# RoboRacer / F1TENTH driver stack on ROS 2 Humble.
# Builds for the machine that runs `docker build` (arm64 on the Jetson).
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Build tools, plus asio_cmake_module, which rosdep misses for vesc_driver
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-humble-asio-cmake-module \
        git nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/ros2_ws

# Install every dependency listed in the packages' package.xml files.
# The source code itself is bind-mounted at runtime, so this layer only needs
# rebuilding when dependencies change (bash setup_docker.sh --recreate).
COPY src/ src/
RUN apt-get update \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

# Every interactive shell gets ROS and the built workspace sourced.
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
 && echo '[ -f /root/ros2_ws/install/setup.bash ] && source /root/ros2_ws/install/setup.bash' >> /root/.bashrc

# The container just stays up; you open shells in it with start.sh.
CMD ["sleep", "infinity"]
