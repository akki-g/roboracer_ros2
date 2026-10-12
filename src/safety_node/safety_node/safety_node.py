#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class SafetyNode(Node):

    def __init__(self):
        super().__init__('safety_node')

        self.brake_hold_time = 0.5
        self.last_brake_time = 0.0

        # TUNED FOR SIM
        self.ttc_threshold = 0.2
        self.speed_threshold = 0.05

        self.speed = 0.0
        self.brake_count = 0
        self.brake_threshold = 2

        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.get_logger().info("Safety Node Started")

    def odom_callback(self, msg):
        self.speed = msg.twist.twist.linear.x

    def scan_callback(self, msg):

        if abs(self.speed) < self.speed_threshold:
            return

        ranges = np.array(msg.ranges)

        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        range_rates = -self.speed * np.cos(angles)

        ttc = np.full(len(ranges), np.inf)

        approaching = range_rates < 0
        ttc[approaching] = ranges[approaching] / np.abs(range_rates[approaching])

        ttc = np.nan_to_num(ttc, nan=np.inf, posinf=np.inf, neginf=np.inf)

        min_ttc = np.min(ttc)

        if min_ttc < self.ttc_threshold:
            self.brake_count += 1
        else:
            self.brake_count = 0

        if time.time() - self.last_brake_time < self.brake_hold_time:
            self.brake()
            return

        if self.brake_count >= self.brake_threshold:
            self.brake()

    def brake(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.pub.publish(msg)
        self.last_break_time = time.time()
        self.get_logger().warn("🚨 EMERGENCY BRAKE")


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
