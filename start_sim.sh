#!/bin/bash
# Start the F1TENTH ROS sim (f1tenth_gym_ros). Optional argument: the sim
# config to use (default ./sim/sim.yaml), e.g. ./start_sim.sh sim/sim_rl.yaml
CONFIG="${1:-./sim/sim.yaml}"
cp "$CONFIG" /sim_ws/src/f1tenth_gym_ros/config/sim.yaml
cd /sim_ws/src/f1tenth_gym_ros/launch
ros2 launch gym_bridge_launch.py
