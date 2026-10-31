
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class ReactiveFollowGap(Node):

    def __init__(self):
        super().__init__('reactive_node')

        # Declare parameters
        self.declare_parameter('bubble_radius', 0.5)
        self.declare_parameter('preprocess_conv_size', 3)
        self.declare_parameter('max_lidar_range', 3.0)
        self.declare_parameter('speed_min', 1.5)
        self.declare_parameter('speed_max', 2.5)
        self.declare_parameter('steering_gain', 1.0)

        # Get parameters
        self.bubble_radius = self.get_parameter(
            'bubble_radius').value

        self.preprocess_conv_size = self.get_parameter(
            'preprocess_conv_size').value

        self.max_lidar_range = self.get_parameter(
            'max_lidar_range').value

        self.speed_min = self.get_parameter(
            'speed_min').value

        self.speed_max = self.get_parameter(
            'speed_max').value

        self.steering_gain = self.get_parameter(
            'steering_gain').value

        # Topics
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # LiDAR subscriber
        self.lidar_sub = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.lidar_callback,
            10
        )

        # Drive publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        self.get_logger().info('Follow the Gap Node Initialized')
        self.get_logger().info(
            f'Bubble Radius: {self.bubble_radius}m'
        )
        self.get_logger().info(
            f'Max LiDAR Range: {self.max_lidar_range}m'
        )
        self.get_logger().info(
            f'Speed Range: {self.speed_min}-{self.speed_max} m/s'
        )

    def preprocess_lidar(self, ranges):
        """
        Preprocess LiDAR data:
        1. Handle NaN and infinity values.
        2. Clip ranges to maximum distance.
        3. Apply moving average filter.
        """

        proc_ranges = np.array(ranges, dtype=float)

        proc_ranges = np.nan_to_num(
            proc_ranges,
            nan=self.max_lidar_range,
            posinf=self.max_lidar_range,
            neginf=0.0
        )

        proc_ranges = np.clip(
            proc_ranges,
            0,
            self.max_lidar_range
        )

        if self.preprocess_conv_size > 1:
            kernel = (
                np.ones(self.preprocess_conv_size)
                / self.preprocess_conv_size
            )

            proc_ranges = np.convolve(
                proc_ranges,
                kernel,
                mode='same'
            )

        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """
        Find the largest continuous sequence of non-zero
        LiDAR ranges.

        Returns:
            start_idx, end_idx
        """

        mask = free_space_ranges > 0

        diff = np.diff(
            np.concatenate(([0], mask.astype(int), [0]))
        )

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) == 0:
            self.get_logger().warn(
                'No gaps found in LiDAR data!'
            )
            return -1, -1

        gap_lengths = ends - starts

        max_gap_idx = np.argmax(gap_lengths)

        start_idx = starts[max_gap_idx]
        end_idx = ends[max_gap_idx] - 1

        return start_idx, end_idx

    def find_best_point(self, start_i, end_i, ranges):
        """
        Find the best point in the gap.

        Weighted strategy:
        80% furthest point
        20% center of gap
        """

        if start_i < 0 or start_i >= end_i:
            return -1

        if end_i >= len(ranges):
            return -1

        gap_ranges = ranges[start_i:end_i + 1]

        # Furthest point
        furthest_idx_in_gap = np.argmax(gap_ranges)
        furthest_idx = start_i + furthest_idx_in_gap

        # Center of gap
        centered_idx = (start_i + end_i) // 2

        # Weighted selection
        best_idx = int(
            0.5 * furthest_idx +
            0.5 * centered_idx
        )

        return best_idx

    def get_bubble_indices(self, data, closest_idx,
                           closest_distance):
        """
        Calculate indices inside the safety bubble.
        """

        if closest_distance <= 0:
            closest_distance = 0.1

        bubble_angle = (
            self.bubble_radius / closest_distance
        )

        angle_increment = abs(data.angle_increment)

        if angle_increment <= 0:
            return np.array([closest_idx])

        bubble_idx_range = int(
            np.ceil(bubble_angle / angle_increment)
        )

        start_idx = max(
            0,
            closest_idx - bubble_idx_range
        )

        end_idx = min(
            len(data.ranges) - 1,
            closest_idx + bubble_idx_range
        )

        return np.arange(start_idx, end_idx + 1)

    def publish_drive(self, steering_angle, speed):
        """
        Publish AckermannDriveStamped message.
        """

        drive_msg = AckermannDriveStamped()

        drive_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        drive_msg.header.frame_id = 'base_link'

        drive_msg.drive.steering_angle = float(
            steering_angle
        )

        drive_msg.drive.speed = float(speed)

        self.drive_pub.publish(drive_msg)

    def lidar_callback(self, data):
        """
        Follow the Gap algorithm.
        """

        # Step 1: Preprocess LiDAR
        ranges = np.array(data.ranges)

        if len(ranges) == 0:
            self.publish_drive(0.0, 0.0)
            return

        proc_ranges = self.preprocess_lidar(ranges)

        # Step 2: Find closest point
        proc_ranges_copy = proc_ranges.copy()

        valid_ranges = proc_ranges_copy > 0

        if not np.any(valid_ranges):
            self.get_logger().warn(
                'No valid LiDAR ranges!'
            )
            self.publish_drive(0.0, 0.0)
            return

        closest_distance = np.min(
            proc_ranges_copy[valid_ranges]
        )

        # Ignore invalid zero values when finding minimum
        closest_idx = np.argmin(
            np.where(
                valid_ranges,
                proc_ranges_copy,
                np.inf
            )
        )

        # Step 3: Create safety bubble
        bubble_indices = self.get_bubble_indices(
            data,
            closest_idx,
            closest_distance
        )

        proc_ranges_copy[bubble_indices] = 0.0

        # Step 4: Find maximum gap
        start_i, end_i = self.find_max_gap(
            proc_ranges_copy
        )

        if start_i == -1:
            self.publish_drive(0.0, 0.0)
            return

        gap_length = end_i - start_i + 1

        # Step 5: Find best point
        best_idx = self.find_best_point(
            start_i,
            end_i,
            proc_ranges_copy
        )

        if best_idx == -1:
            self.publish_drive(0.0, 0.0)
            return

        # Step 6: Calculate steering angle
        angle_to_best = (
            data.angle_min +
            best_idx * data.angle_increment
        )

        steering_angle = (
            self.steering_gain * angle_to_best
        )

        # Limit steering to 30 degrees
        max_steering = np.deg2rad(30.0)

        steering_angle = np.clip(
            steering_angle,
            -max_steering,
            max_steering
        )

        # Step 7: Calculate speed
        steering_magnitude = abs(steering_angle)

        if steering_magnitude < np.deg2rad(10):
            speed = self.speed_max

        elif steering_magnitude < np.deg2rad(20):
            speed = (
                self.speed_min + self.speed_max
            ) / 2.0

        else:
            speed = self.speed_min

        # Adjust speed based on gap size
        min_safe_gap = 50

        if gap_length < min_safe_gap:
            speed = speed * (
                gap_length / min_safe_gap
            )

            speed = max(speed, self.speed_min)

        # Step 8: Publish drive command
        self.publish_drive(
            steering_angle,
            speed
        )

        # Debug information
        self.get_logger().info(
            f'Gap: [{start_i}:{end_i}] | '
            f'Length: {gap_length} | '
            f'Best: {best_idx} | '
            f'Angle: {np.rad2deg(steering_angle):.1f} deg | '
            f'Speed: {speed:.2f} m/s',
            throttle_duration_sec=0.5
        )


def main(args=None):

    rclpy.init(args=args)

    print('Follow the Gap Initialized')

    reactive_node = ReactiveFollowGap()

    try:
        rclpy.spin(reactive_node)

    except KeyboardInterrupt:
        pass

    finally:
        reactive_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
