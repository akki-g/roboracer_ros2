# RoboRacer / F1TENTH driver stack on ROS 2 Humble, for ~/f1tenth_ws on the Jetson.
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
# The code itself is mounted from the Jetson at runtime, so this layer only
# needs rebuilding when dependencies change (bash setup_docker.sh --recreate).
COPY src/ src/
RUN apt-get update \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

# Every shell opened in the container gets ROS and the container's build sourced.
# (The container builds into install_docker/ so it never overwrites the build
# made outside Docker in install/.)
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
 && echo '[ -f /root/ros2_ws/install_docker/setup.bash ] && source /root/ros2_ws/install_docker/setup.bash' >> /root/.bashrc

# The container just stays up; open shells in it with start.sh.
CMD ["sleep", "infinity"]