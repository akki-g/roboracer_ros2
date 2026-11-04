from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():

    return LaunchDescription([

        # 🏎️ Simulator
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(
                    os.getenv('HOME'),
                    'f1tenth_ws/src/f1tenth_system/f1tenth_stack/launch/bringup_launch.py'
                )
            ])
        ),

	Node(
	    package='slam_toolbox',
	    executable='async_slam_toolbox_node',
	    name='slam_toolbox',
	    output='screen',
	    parameters=[{'use_sim_time': False}]
	),

        # Wall Follow
        Node(
            package='wall_follow_pkg',
            executable='wall_follow_node',
            output='screen'
        ),

        #  Safety
        Node(
            package='safety_node',
            executable='safety_node',
            output='screen'
        ),

        #  Record data
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '/scan',
                '/odom',
                '/drive'
            ],
            output='screen'
        ),
    ])
