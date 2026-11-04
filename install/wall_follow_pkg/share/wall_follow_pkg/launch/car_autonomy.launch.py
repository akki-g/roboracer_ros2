import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Found by package name, so it works natively and in Docker
    bringup = os.path.join(
        get_package_share_directory('f1tenth_stack'), 'launch', 'bringup_launch.py')

    return LaunchDescription([
        # Real car: VESC, LiDAR, controller
        IncludeLaunchDescription(PythonLaunchDescriptionSource(bringup)),

        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[{'use_sim_time': False}],
        ),

        Node(package='wall_follow_pkg', executable='wall_follow_node', output='screen'),
        Node(package='safety_node', executable='safety_node', output='screen'),

        # Records into the folder you launch from (rosbag2_* is gitignored)
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '/scan', '/odom', '/drive'],
            output='screen',
        ),
    ])