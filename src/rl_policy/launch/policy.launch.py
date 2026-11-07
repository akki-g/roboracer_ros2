"""Launch the RL policy node.

    ros2 launch rl_policy policy.launch.py mode:=sim policy_path:=/path/policy.npz
    ros2 launch rl_policy policy.launch.py mode:=car max_speed:=1.0

mode selects config/policy_<mode>.yaml (topics); max_speed overrides the
yaml (default 1.5 m/s). scale_speed:=true maps the policy's 0-8 m/s onto
0..max_speed instead of clamping at max_speed. A bag of /scan, odom, /drive and the debug topic is
recorded when record:=true.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    share = get_package_share_directory('rl_policy')
    mode = LaunchConfiguration('mode').perform(context)
    if mode not in ('car', 'sim'):
        raise RuntimeError(f"mode must be 'car' or 'sim', got {mode!r}")
    params = [os.path.join(share, 'config', f'policy_{mode}.yaml')]
    overrides = {'policy_path': LaunchConfiguration('policy_path').perform(context)}
    max_speed = LaunchConfiguration('max_speed').perform(context)
    if max_speed:
        overrides['max_speed'] = float(max_speed)
    for name in ('ttc_brake', 'scale_speed'):
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = value.lower() in ('1', 'true', 'yes')
    odom = '/ego_racecar/odom' if mode == 'sim' else '/odom'
    return [
        Node(package='rl_policy', executable='policy_node', name='policy_node', output='screen',
             parameters=params + [overrides]),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '/scan', odom, '/drive', '/policy_node/debug'],
            output='screen', condition=IfCondition(LaunchConfiguration('record'))),
    ]


def generate_launch_description():
    # the best tested policy; policy.npz is the phase-1 MLP (update 325), kept for comparison only
    default_policy = os.path.join(get_package_share_directory('rl_policy'), 'models', 'gru_u02050.npz')
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='car', description='car or sim'),
        DeclareLaunchArgument('policy_path', default_value=default_policy),
        DeclareLaunchArgument('max_speed', default_value='', description='m/s; empty keeps the yaml value'),
        DeclareLaunchArgument('ttc_brake', default_value='', description='true/false; empty keeps the yaml value'),
        DeclareLaunchArgument('scale_speed', default_value='',
                              description='true: scale the policy speed range onto 0..max_speed instead of clamping; '
                                          'empty keeps the yaml value'),
        DeclareLaunchArgument('record', default_value='false'),
        OpaqueFunction(function=_nodes),
    ])
