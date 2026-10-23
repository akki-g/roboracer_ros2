import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim.yaml')

def generate_launch_description():
    gym_share = get_package_share_directory('f1tenth_gym_ros')
    with open(CONFIG) as f:
        sim = yaml.safe_load(f)['bridge']['ros__parameters']

    return LaunchDescription([
        DeclareLaunchArgument('wall_follow', default_value='true'),
        DeclareLaunchArgument('safety', default_value='true'),

        # Simulator: publishes /scan and /ego_racecar/odom, follows /drive.
        # Same as upstream gym_bridge_launch.py minus RViz (no display in Docker).
        Node(package='f1tenth_gym_ros', executable='gym_bridge', name='bridge',
             parameters=[CONFIG]),
        Node(package='nav2_map_server', executable='map_server',
             parameters=[{'yaml_filename': sim['map_path'] + '.yaml',
                          'topic': 'map', 'frame_id': 'map', 'use_sim_time': True}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_localization',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': ['map_server']}]),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='ego_robot_state_publisher',
             parameters=[{'robot_description': Command(
                 ['xacro ', os.path.join(gym_share, 'launch', 'ego_racecar.xacro')])}],
             remappings=[('/robot_description', 'ego_robot_description')]),

        # Our nodes. The sim calls odometry /ego_racecar/odom; the car calls it /odom.
        Node(package='wall_follow_pkg', executable='wall_follow_node', output='screen',
             condition=IfCondition(LaunchConfiguration('wall_follow'))),
        Node(package='safety_node', executable='safety_node', output='screen',
             remappings=[('/odom', '/ego_racecar/odom')],
             condition=IfCondition(LaunchConfiguration('safety'))),

        # Visualization
        Node(package='foxglove_bridge', executable='foxglove_bridge',
             parameters=[{'port': 8765}]),
    ])