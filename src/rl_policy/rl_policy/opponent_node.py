"""Opponent car for the two-car ROS sim (num_agent: 2): drives /opp_drive by pure pursuit
along the map's centerline at a set speed, or one that changes within a range.

    ros2 run rl_policy opponent_driver --ros-args -p centerline:=<map>_centerline.csv -p speed:=1.0
    ros2 run rl_policy opponent_driver --ros-args -p centerline:=<map>_centerline.csv -p speed:=1.0-4.0 -p period:=6.0

tools/sim_gru.sh starts it (with a two-car sim config) when OPP_SPEED is set.
"""

import math
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .opponent_pursuit import SpeedSchedule, Track, command, parse_speed


class OpponentDriver(Node):
    def __init__(self):
        super().__init__("opponent_driver")
        self.declare_parameter("centerline", "")
        self.declare_parameter("speed", "1.0")
        self.declare_parameter("period", 6.0)
        self.declare_parameter("a_lat", 4.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("odom_topic", "/opp_racecar/odom")
        self.declare_parameter("drive_topic", "/opp_drive")
        self.track = Track.from_csv(self.get_parameter("centerline").value)
        lo, hi = parse_speed(self.get_parameter("speed").value)
        self.schedule = SpeedSchedule(lo, hi, float(self.get_parameter("period").value),
                                      int(self.get_parameter("seed").value))
        self.a_lat = float(self.get_parameter("a_lat").value)
        self.pose = None
        self.t0 = time.monotonic()
        self.last_target = None
        self.pub = self.create_publisher(AckermannDriveStamped, self.get_parameter("drive_topic").value, 10)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self.on_odom, 10)
        self.create_timer(0.02, self.tick)
        self.get_logger().info(f"opponent: {lo:.1f}-{hi:.1f} m/s on a {self.track.length:.0f} m centerline")

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw, msg.twist.twist.linear.x)

    def tick(self):
        if self.pose is None:
            return
        speed, steer, target = command(self.track, self.schedule, time.monotonic() - self.t0, *self.pose, a_lat=self.a_lat)
        if target != self.last_target:
            self.get_logger().info(f"opponent target speed {target:.2f} m/s")
            self.last_target = target
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = OpponentDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
