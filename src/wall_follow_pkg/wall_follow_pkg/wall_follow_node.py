#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int16
from ackermann_msgs.msg import AckermannDriveStamped


class WallFollow(Node):

    def __init__(self):
        super().__init__('wall_follow_node')

        # PARAMETERS
        self.kp = 0.5
        self.kd = 0.025
        self.ki = 0.0

        self.desired_distance = 1.2
        self.lookahead_distance = 1.0
        self.theta = np.deg2rad(50.0)

        self.wall_side = 'left'   # TRY 'right' IF NEEDED

        # PID
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        # SUBSCRIBE
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pid_error_pub = self.create_publisher(Float32, "/pid_error", 0)
        self.pid_target_pub = self.create_publisher(Float32, "/pid_target", 0)

        # PUBLISH
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.get_logger().info("Wall Follow Node Started")

    def get_range(self, data, angle):
        idx = int((angle - data.angle_min) / data.angle_increment)
        idx = np.clip(idx, 0, len(data.ranges) - 1)

        val = data.ranges[idx]

        if np.isnan(val) or np.isinf(val):
            return data.range_max

        return val

    def get_error(self, scan, desired_dist):

        if self.wall_side == 'right':
            angle_b = -np.pi / 2
            angle_a = angle_b + self.theta
        else:
            angle_b = np.pi / 2
            angle_a = angle_b - self.theta

        b = self.get_range(scan, angle_b)
        a = self.get_range(scan, angle_a)

        numerator = a * np.cos(self.theta) - b
        denominator = a * np.sin(self.theta)

        if abs(denominator) < 1e-6:
            alpha = 0.0
        else:
            alpha = np.arctan2(numerator, denominator)

        D_t = b * np.cos(alpha)
        D_t1 = D_t + self.lookahead_distance * np.sin(alpha)

        error = D_t1 - desired_dist
        return error

    def pid_control(self, error):

        current_time = self.get_clock().now()

        if self.prev_time is None:
            dt = 0.0
        else:
            dt = (current_time - self.prev_time).nanoseconds / 1e9

        self.prev_time = current_time

        if dt <= 1e-6:
            dt = 1e-6

        # PID
        P = self.kp * error
        self.integral += error * dt
        I = self.ki * self.integral
        D = self.kd * (error - self.prev_error) / dt

        angle = P + I + D
        self.prev_error = error

        pre_clamped_angle: float = angle

        # Clamp steering
        angle = np.clip(angle, -0.5, 0.5)

        # Speed control
        if abs(angle) < 0.1:
            speed = 2
        elif abs(angle) < 0.3:
            speed = 1.5
        else:
            speed = 1

        # Slight smoothing
        angle *= 0.7

        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(angle)

        self.drive_pub.publish(msg)
        
        pre_clamped_msg = Float32()
        pre_clamped_msg.data = pre_clamped_angle
        
        error_msg = Float32()
        error_msg.data = error
        
        self.pid_target_pub.publish(pre_clamped_angle)
        self.pid_error_pub.publish(error_msg)

    def scan_callback(self, msg):
        error = self.get_error(msg, self.desired_distance)
        self.pid_control(error)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
