#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: Subscribe to LIDAR
        # TODO: Publish to drive

    def preprocess_lidar(self, ranges):
      """ 
      Preprocess the LiDAR scan array.
      1. Apply mean filter over a window
      2. Clip values to max_lidar_range
      3. Handle inf and nan values

      Args:
          ranges: array of LiDAR ranges

      Returns:
          proc_ranges: processed array of ranges
      """
      # Convert to numpy array
      proc_ranges = np.array(ranges)

      # Replace inf and nan with max_lidar_range
      proc_ranges = np.nan_to_num(proc_ranges, 
                                  nan=self.max_lidar_range, 
                                  posinf=self.max_lidar_range, 
                                  neginf=0.0)

      # Clip to max range - anything beyond max_lidar_range is set to max_lidar_range
      proc_ranges = np.clip(proc_ranges, 0, self.max_lidar_range)

      # Apply mean filter (moving average) for smoothing
      # Use convolution for efficient averaging
      if self.preprocess_conv_size > 1:
          kernel = np.ones(self.preprocess_conv_size) / self.preprocess_conv_size
          proc_ranges = np.convolve(proc_ranges, kernel, mode='same')

      return proc_ranges

    def find_max_gap(self, free_space_ranges):
      """ 
      Return the start index & end index of the max gap in free_space_ranges.
      A gap is a continuous sequence of non-zero values.

      Args:
          free_space_ranges: array with 0s for obstacles and non-zero for free space

      Returns:
          start_idx: start index of max gap
          end_idx: end index of max gap (inclusive)
      """
      # Find where free space begins and ends
      # non-zero values represent free space
      mask = free_space_ranges > 0

      # Find the boundaries of all gaps
      # np.diff finds changes (0->1 or 1->0)
      diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))

      # starts: where diff changes from 0 to 1 (gap begins)
      # ends: where diff changes from 1 to 0 (gap ends)
      starts = np.where(diff == 1)[0]
      ends = np.where(diff == -1)[0]

      # If no gaps found, return invalid indices
      if len(starts) == 0:
          self.get_logger().warn('No gaps found in LiDAR data!')
          return 0, 0

      # Calculate gap lengths
      gap_lengths = ends - starts

      # Find the maximum gap
      max_gap_idx = np.argmax(gap_lengths)

      start_idx = starts[max_gap_idx]
      end_idx = ends[max_gap_idx] - 1  # -1 because ends is exclusive

      return start_idx, end_idx
    
    def find_best_point(self, start_i, end_i, ranges):
        """
        Find the best point in the gap to navigate toward.

        Strategies:
        1. Naive: furthest point in gap
        2. Centered: middle of gap
        3. Weighted average of furthest and centered

        Args:
            start_i: start index of gap
            end_i: end index of gap
            ranges: array of LiDAR ranges

        Returns:
            best_idx: index of best point in gap
        """
        # Handle edge case
        if start_i >= end_i or end_i >= len(ranges):
            return len(ranges) // 2  # Default to center

        # Extract gap ranges
        gap_ranges = ranges[start_i:end_i+1]

        # Strategy 1: Furthest point (naive)
        furthest_idx_in_gap = np.argmax(gap_ranges)
        furthest_idx = start_i + furthest_idx_in_gap

        # Strategy 2: Centered point
        centered_idx = (start_i + end_i) // 2

        # Strategy 3: Weighted average (80% furthest, 20% centered)
        # This balances safety (center) with progress (furthest)
        best_idx = int(0.8 * furthest_idx + 0.2 * centered_idx)

        # Alternative: just use furthest
        # best_idx = furthest_idx

        # Alternative: just use centered
        # best_idx = centered_idx

        return best_idx
      
    def publish_drive(self, steering_angle, speed):
        """
        Publish AckermannDriveStamped message.

        Args:
            steering_angle: desired steering angle (radians)
            speed: desired speed (m/s)
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = float(speed)

        self.drive_pub.publish(drive_msg)

    def lidar_callback(self, data):
      """ 
      Process each LiDAR scan as per the Follow Gap algorithm
      """
      # Find max length gap
      start_i, end_i = self.find_max_gap(proc_ranges_copy)

      # Debug
      gap_length = end_i - start_i + 1
      self.get_logger().info(
          f'Max gap: start={start_i}, end={end_i}, length={gap_length}',
          throttle_duration_sec=1.0)
      
      ranges = np.array(data.ranges)

      # Step 1: Preprocess the LiDAR scan
      proc_ranges = self.preprocess_lidar(ranges)

      # Step 2: Find closest point to LiDAR
      # Ignore zero values (they represent obstacles from bubble or invalid)
      # Create a copy to modify
      proc_ranges_copy = proc_ranges.copy()

      # Find the closest point (minimum non-zero distance)
      nonzero_ranges = proc_ranges_copy[proc_ranges_copy > 0]
      if len(nonzero_ranges) == 0:
          # No valid ranges, stop the car
          self.get_logger().warn('No valid LiDAR ranges detected!')
          self.publish_drive(0.0, 0.0)
          return

      closest_distance = np.min(nonzero_ranges)
      closest_idx = np.argmin(proc_ranges_copy)

      # Step 3: Eliminate all points inside 'bubble' (set them to zero)
      # Calculate bubble indices based on closest point
      bubble_indices = self.get_bubble_indices(data, closest_idx, closest_distance)
      proc_ranges_copy[bubble_indices] = 0.0

      # Debug logging
      self.get_logger().debug(
          f'Closest point: idx={closest_idx}, dist={closest_distance:.2f}m, '
          f'bubble_size={len(bubble_indices)}',
          throttle_duration_sec=1.0)

      # TODO: Continue with finding max gap...

    def get_bubble_indices(self, data, closest_idx, closest_distance):
      """
      Get indices of all points within the safety bubble around closest point.

      Args:
          data: LaserScan message (for angle info)
          closest_idx: index of closest point
          closest_distance: distance to closest point

      Returns:
          bubble_indices: array of indices within bubble
      """
      # Calculate the angle subtended by the bubble
      # Using small angle approximation: angle ≈ bubble_radius / distance
      if closest_distance == 0:
          closest_distance = 0.1  # Avoid division by zero

      bubble_angle = self.bubble_radius / closest_distance

      # Convert angle to number of indices
      angle_increment = data.angle_increment
      bubble_idx_range = int(np.ceil(bubble_angle / angle_increment))

      # Get start and end indices of bubble
      start_idx = max(0, closest_idx - bubble_idx_range)
      end_idx = min(len(data.ranges) - 1, closest_idx + bubble_idx_range)

      # Return all indices in bubble
      bubble_indices = np.arange(start_idx, end_idx + 1)

      return bubble_indices

def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()