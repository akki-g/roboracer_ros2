source /opt/ros/humble/setup.bash

apt update
apt install -y python3-colcon-common-extensions python3-rosdep

rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -y

colcon build --symlink-install
source install/setup.bash

