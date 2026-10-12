docker run -it --name ros2-humble \
  -v "$(pwd):/root/ros2_ws" \
  -w /root/ros2_ws \
  ros:humble bash
