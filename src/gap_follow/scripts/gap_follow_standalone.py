#!/usr/bin/env python3
"""Standalone ROS 2 Humble Follow-the-Gap controller.

Requires only NumPy and the ROS packages already used by the car: rclpy,
rcl_interfaces, sensor_msgs, and ackermann_msgs. No custom module, package
registration, or rebuild is needed when running this file with python3.

Dry run (after sourcing ROS and the native workspace):
    python3 gap_follow_standalone.py --ros-args -r /drive:=/gap_follow_test
Normal output through the existing Ackermann mux:
    python3 gap_follow_standalone.py --ros-args -p max_speed:=0.5
Quick offline checks, without importing ROS:
    python3 gap_follow_standalone.py --self-test

Default limits: speed 0.5 m/s, steering +/-0.35 rad, clearance radius 0.25 m,
front stop distance 0.5 m, scan timeout 0.3 s. All settings are ROS startup
parameters: restart with -p NAME:=VALUE to change them. The radius represents
half the vehicle width plus a margin; distances are measured from the LiDAR.
These are initial settings, not a measurement or calibration of this car.

Angle zero must point forward and positive scan angles/steering point left.
The algorithm is a reactive heuristic, not a collision-free path planner.
The straight-corridor check does not model the swept body during a turn.
Invalid/unknown ranges, including +inf, stay blocked. Stop commands are never
smoothed. A watchdog only works while this process/control chain is running.
Release R1 before switching controllers; run one autonomous publisher at a
time. This file does not change the existing controller/mux handoff behavior.

Algorithm reference: https://github.com/f1tenth/f1tenth_lab4_template
ROS timer API: https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/node.py
"""

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
import sys
import time

import numpy as np


@dataclass(frozen=True)
class GapConfig:
    max_speed: float = 0.5
    min_speed: float = 0.15       # Corner speed before clearance scaling.
    max_steering_angle: float = 0.35
    max_lidar_range: float = 10.0
    forward_fov_deg: float = 180.0
    bubble_radius: float = 0.25
    min_gap_distance: float = 0.75
    stop_distance: float = 0.5
    slow_distance: float = 1.5
    front_sector_deg: float = 20.0
    min_front_valid_fraction: float = 0.7
    preprocess_conv_size: int = 3
    disparity_threshold: float = 0.3
    target_window_deg: float = 10.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value)):
                raise ValueError(f'{field.name} must be a finite number')
        if not 0 <= self.min_speed <= self.max_speed:
            raise ValueError('Require 0 <= min_speed <= max_speed')
        if not 0 < self.max_steering_angle < math.pi / 2:
            raise ValueError('max_steering_angle must be between 0 and pi/2')
        if not 0 < self.front_sector_deg <= self.forward_fov_deg <= 180:
            raise ValueError('Require 0 < front_sector_deg <= forward_fov_deg <= 180')
        if not 0 < self.stop_distance < self.slow_distance <= self.max_lidar_range:
            raise ValueError('Require 0 < stop_distance < slow_distance <= max_lidar_range')
        if not 0 < self.bubble_radius < self.min_gap_distance <= self.max_lidar_range:
            raise ValueError('Require 0 < bubble_radius < min_gap_distance <= max_lidar_range')
        if not 0 < self.min_front_valid_fraction <= 1:
            raise ValueError('min_front_valid_fraction must be in (0, 1]')
        if not 0 < self.disparity_threshold <= self.max_lidar_range:
            raise ValueError('disparity_threshold must be positive and <= max_lidar_range')
        if not 0 < self.target_window_deg <= self.forward_fov_deg:
            raise ValueError('target_window_deg must be positive and <= forward_fov_deg')
        if (not isinstance(self.preprocess_conv_size, Integral)
                or not 1 <= self.preprocess_conv_size <= 101
                or self.preprocess_conv_size % 2 != 1):
            raise ValueError('preprocess_conv_size must be an odd integer from 1 to 101')


@dataclass(frozen=True)
class GapCommand:
    steering_angle: float
    speed: float
    reason: str
    target_angle: float = 0.0
    front_clearance: float = 0.0


@dataclass(frozen=True)
class ScanGeometry:
    indices: np.ndarray
    angles: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    front: np.ndarray
    jumps: np.ndarray
    step: float
    target_window: int


class GapFollower:
    """Pure NumPy controller. Geometry is cached; measurements are always fresh."""

    def __init__(self, config=GapConfig()):
        if not isinstance(config, GapConfig):
            raise TypeError('config must be a GapConfig')
        self.config = config
        self._geometry_key = None
        self._geometry = None
        self._kernel = np.ones(config.preprocess_conv_size) / config.preprocess_conv_size

    @staticmethod
    def stop(reason, clearance=0.0):
        return GapCommand(0.0, 0.0, reason, front_clearance=float(clearance))

    def _scan_geometry(self, count, angle_min, angle_increment):
        key = (count, angle_min, angle_increment)
        if key == self._geometry_key:
            return self._geometry
        cfg = self.config
        step = abs(angle_increment)
        angles = (angle_min + np.arange(count) * angle_increment + math.pi) % (
            2 * math.pi) - math.pi
        indices = np.flatnonzero(
            np.abs(angles) <= math.radians(cfg.forward_fov_deg) / 2 + 1e-9)
        indices = indices[np.argsort(angles[indices], kind='stable')]
        angles = angles[indices]
        if angles.size:
            # A full-circle scan can contain the same endpoint twice.
            keep = np.r_[True, np.diff(angles) > step / 2]
            indices, angles = indices[keep], angles[keep]
        window = max(3, int(math.ceil(math.radians(cfg.target_window_deg) / step)))
        if window % 2 == 0:
            window += 1
        geometry = ScanGeometry(
            indices, angles, np.cos(angles), np.sin(angles),
            np.abs(angles) <= math.radians(cfg.front_sector_deg) / 2 + 1e-9,
            np.diff(angles) > 1.5 * step, step, window,
        )
        self._geometry_key, self._geometry = key, geometry
        return geometry

    def _inflate_edges(self, processed, distances, valid, geometry):
        """Extend a nearby obstacle edge into its neighboring farther readings."""
        differences = np.diff(distances)
        edges = np.flatnonzero(
            valid[:-1] & valid[1:] & ~geometry.jumps
            & (np.abs(differences) >= self.config.disparity_threshold))
        for edge in edges:
            near = edge if differences[edge] > 0 else edge + 1
            distance = distances[near]
            half_angle = math.asin(min(1.0, self.config.bubble_radius / distance))
            if near == edge:
                start = near + 1
                end = np.searchsorted(
                    geometry.angles, geometry.angles[near] + half_angle, side='right')
            else:
                start = np.searchsorted(
                    geometry.angles, geometry.angles[near] - half_angle, side='left')
                end = near
            processed[start:end] = np.minimum(processed[start:end], distance)

    def _choose_target(self, processed, free, geometry):
        angles, step = geometry.angles, geometry.step
        starts = np.flatnonzero(
            free & (~np.r_[False, free[:-1]] | np.r_[False, geometry.jumps]))
        ends = np.flatnonzero(
            free & (~np.r_[free[1:], False] | np.r_[geometry.jumps, False]))
        candidates = []
        for start, end in zip(starts, ends):
            count = int(end - start + 1)
            if count < 3:
                continue
            segment = processed[start:end + 1]
            span = float(angles[end] - angles[start])
            near = float(np.min(segment))
            if 2 * near * math.sin(min(math.pi, span) / 2) < 2 * self.config.bubble_radius:
                continue
            window = min(geometry.target_window, count if count % 2 else count - 1)
            centers = np.arange(start + window // 2, end - window // 2 + 1)
            # Use the worst clearance in each complete window: one unusually
            # long ray cannot make an otherwise short opening look attractive.
            scores = np.min(np.lib.stride_tricks.sliding_window_view(segment, window), axis=1)
            margin = math.asin(min(1.0, self.config.bubble_radius / near))
            interior = ((angles[centers] >= angles[start] + margin)
                        & (angles[centers] <= angles[end] - margin))
            if not np.any(interior):
                continue
            centers, scores = centers[interior], scores[interior]
            best_score = float(np.max(scores))
            # Near-equal windows prefer forward progress, reducing noisy flips.
            eligible = centers[scores >= 0.98 * best_score]
            target = int(eligible[np.argmin(np.abs(angles[eligible]))])
            candidates.append((span, target, best_score))
        if not candidates:
            return None
        widest = max(candidate[0] for candidate in candidates)
        equivalent = [c for c in candidates if c[0] >= widest - step / 2]
        return min(equivalent, key=lambda c: abs(angles[c[1]]))[1]

    def compute(self, ranges, angle_min, angle_increment, range_min, range_max):
        cfg = self.config
        try:
            raw = np.asarray(ranges, dtype=float)
            metadata = np.asarray(
                [angle_min, angle_increment, range_min, range_max], dtype=float)
        except (TypeError, ValueError, OverflowError):
            return self.stop('invalid_scan')
        if raw.ndim != 1 or raw.size < 3:
            return self.stop('invalid_scan')
        if metadata.shape != (4,) or not np.all(np.isfinite(metadata)):
            return self.stop('invalid_scan_metadata')
        angle_min, angle_increment, range_min, range_max = metadata
        step = abs(angle_increment)
        if (step == 0 or step > math.pi or range_min < 0 or range_max <= range_min
                or (raw.size - 1) * step > 2 * math.pi + step):
            return self.stop('invalid_scan_metadata')
        geometry = self._scan_geometry(raw.size, angle_min, angle_increment)
        raw = raw[geometry.indices]
        if raw.size < 3 or np.count_nonzero(geometry.front) < 3:
            return self.stop('no_forward_data')
        valid = (np.isfinite(raw) & (raw > 0) & (raw >= range_min) & (raw <= range_max))
        if np.mean(valid[geometry.front]) < cfg.min_front_valid_fraction:
            return self.stop('insufficient_front_data')
        distances = np.where(valid, np.minimum(raw, cfg.max_lidar_range), 0.0)
        x, y = distances * geometry.cosine, distances * geometry.sine
        corridor = valid & (x > 0) & (np.abs(y) <= cfg.bubble_radius)
        observed_front = valid & geometry.front
        if not np.any(corridor):
            return self.stop('unobserved_forward_corridor')
        clearance = float(np.min(x[corridor | observed_front]))
        if clearance <= cfg.stop_distance:
            return self.stop('front_obstacle', clearance)
        closest = int(np.argmin(np.where(valid, distances, np.inf)))
        closest_distance = float(distances[closest])
        if closest_distance <= cfg.bubble_radius:
            return self.stop('obstacle_inside_bubble', clearance)

        processed = distances.copy()
        if cfg.preprocess_conv_size > 1:
            average = np.convolve(
                np.pad(distances, cfg.preprocess_conv_size // 2), self._kernel, mode='valid')
            processed = np.minimum(processed, average)
        self._inflate_edges(processed, distances, valid, geometry)
        bubble_angle = math.asin(min(1.0, cfg.bubble_radius / closest_distance))
        bubble = np.abs(geometry.angles - geometry.angles[closest]) <= bubble_angle + step / 2
        free = valid & ~bubble & (processed >= cfg.min_gap_distance)
        target_index = self._choose_target(processed, free, geometry)
        if target_index is None:
            return self.stop('no_gap', clearance)
        target = float(geometry.angles[target_index])
        steering = float(np.clip(target, -cfg.max_steering_angle, cfg.max_steering_angle))
        corner_speed = cfg.max_speed - (cfg.max_speed - cfg.min_speed) * (
            abs(steering) / cfg.max_steering_angle)
        clearance_scale = float(np.clip(
            (clearance - cfg.stop_distance) / (cfg.slow_distance - cfg.stop_distance), 0.0, 1.0))
        return GapCommand(steering, float(corner_speed * clearance_scale), 'ok', target, clearance)


def scan_timestamp_status(stamp_ns, now_ns, timeout):
    """Zero stamps use the arrival watchdog; nonzero stamps must share ROS time."""
    if stamp_ns == 0:
        return None
    if stamp_ns < 0 or now_ns <= 0:
        return 'invalid_scan_clock'
    age = (now_ns - stamp_ns) / 1e9
    if age > timeout:
        return 'stale_scan'
    if age < -timeout:
        return 'scan_clock_ahead'
    return None


def build_ros_node_class():
    """Import ROS lazily so the core and --self-test work without ROS installed."""
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from rcl_interfaces.msg import ParameterDescriptor
    from ackermann_msgs.msg import AckermannDriveStamped
    from sensor_msgs.msg import LaserScan

    class GapFollowNode(Node):
        def __init__(self):
            super().__init__('gap_follow_standalone')
            descriptor = ParameterDescriptor(
                read_only=True, description='Set at startup; restart to change.')
            defaults = GapConfig()
            settings = {
                field.name: self.declare_parameter(
                    field.name, getattr(defaults, field.name), descriptor).value
                for field in fields(defaults)
            }
            self.controller = GapFollower(GapConfig(**settings))
            self.scan_timeout = self.declare_parameter('scan_timeout', 0.3, descriptor).value
            if not math.isfinite(self.scan_timeout) or self.scan_timeout <= 0:
                raise ValueError('scan_timeout must be positive and finite')
            self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 1)
            scan_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE)
            self.subscription = self.create_subscription(LaserScan, '/scan', self.on_scan, scan_qos)
            self.last_received = None
            self.last_reason = None
            self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self.watchdog = self.create_timer(
                min(0.05, self.scan_timeout / 2), self.check_timeout, clock=self.steady_clock)
            self.get_logger().info(
                f'Gap following: max_speed={settings["max_speed"]:.2f} m/s, '
                f'steering_limit={settings["max_steering_angle"]:.2f} rad. '
                'Use /drive:=/gap_follow_test for a dry run.')

        def publish_command(self, command):
            message = AckermannDriveStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = 'base_link'
            message.drive.steering_angle = float(command.steering_angle)
            message.drive.speed = float(command.speed)
            self.publisher.publish(message)
            if command.reason != self.last_reason:
                self.last_reason = command.reason
                self.get_logger().info(f'Gap following: {command.reason}')

        def on_scan(self, scan):
            try:
                stamp_ns = scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
                status = scan_timestamp_status(
                    stamp_ns, self.get_clock().now().nanoseconds, self.scan_timeout)
                if status is not None:
                    self.publish_command(self.controller.stop(status))
                    return
                self.last_received = time.monotonic()
                command = self.controller.compute(
                    scan.ranges, scan.angle_min, scan.angle_increment,
                    scan.range_min, scan.range_max)
            except Exception as error:
                # A callback failure must not leave the previous moving command
                # as our most recent output. KeyboardInterrupt is not swallowed.
                command = self.controller.stop('scan_processing_error')
                self.get_logger().error(f'{type(error).__name__}: {error}', throttle_duration_sec=1.0)
            self.publish_command(command)

        def check_timeout(self):
            if self.last_received is None:
                self.publish_command(self.controller.stop('waiting_for_scan'))
            elif time.monotonic() - self.last_received > self.scan_timeout:
                self.publish_command(self.controller.stop('scan_timeout'))

    return GapFollowNode


def self_test():
    """Small portable smoke check; fuller synthetic tests were run during development."""
    angles = np.linspace(-3 * math.pi / 4, 3 * math.pi / 4, 1081)
    follower = GapFollower()

    def command(ranges):
        return follower.compute(ranges, angles[0], angles[1] - angles[0], 0.05, 30.0)

    corridor = np.minimum(8.0, 1.2 / np.maximum(np.abs(np.sin(angles)), 1e-6))
    result = command(corridor)
    assert 0 < result.speed <= 0.5 and abs(result.steering_angle) < 0.1
    for value in (0.0, np.nan, np.inf, -np.inf):
        assert command(np.full(angles.size, value)).speed == 0
    blocked = np.full(angles.size, 4.0)
    blocked[np.abs(angles) < math.radians(10)] = 0.35
    assert command(blocked).speed == 0
    assert scan_timestamp_status(1_000_000_000, 2_000_000_000, 0.3) == 'stale_scan'
    print('Self-test passed: corridor, invalid data, front stop, timestamp check.')


def main():
    if sys.argv[1:] == ['--self-test']:
        self_test()
        return
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.signals import SignalHandlerOptions
        node_class = build_ros_node_class()
    except ImportError as error:
        raise SystemExit(
            'ROS imports failed. Source /opt/ros/humble/setup.bash and your '
            'workspace install/setup.bash, then run this file with system python3. '
            f'Details: {error}') from error
    import signal

    def request_stop(_signum, _frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    node = None
    try:
        # Keep the ROS context alive while handling Ctrl+C, so the finally
        # block can make a best-effort zero-speed publication before shutdown.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = node_class()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                if rclpy.ok():
                    node.publish_command(node.controller.stop('shutdown'))
            finally:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == '__main__':
    main()
