# !/bin/bash

cp ./sim/sim.yaml /sim_ws/src/f1tenth_gym_ros/config/sim.yaml
cd /sim_ws/src/f1tenth_gym_ros/launch
ros2 launch gym_bridge_launch.py