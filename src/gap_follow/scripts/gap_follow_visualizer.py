#!/usr/bin/env python3
"""Read-only RViz helper for the running standalone controller.

Subscribes to live scan, navigation command, mux output, odometry and rosout.
The only application output is /gap_follow_monitor/markers at 2 Hz. It never
starts or reruns the planner, changes parameters, or publishes a drive command.

Orange dashed lines show the geometry of the latest /drive steering request,
not a collision-checked path or measured steering.
Logged front/path values are separate, throttled ~1 Hz observations.

After sourcing ROS and the workspace:
    python3 src/gap_follow/scripts/gap_follow_visualizer.py --duration 1200
    rviz2 -d src/gap_follow/config/live_monitor.rviz
"""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import re
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


NUMBER = r'([-+0-9.eE]+)'
DEBUG = re.compile(r'front=' + NUMBER + r' path=' + NUMBER
                   + r' target=' + NUMBER + r' steer=' + NUMBER + r' speed=' + NUMBER)
REASON = re.compile(r'Gap following: ([^;]+); front=')


class GapFollowVisualizer(Node):
    def __init__(self, options):
        super().__init__('gap_follow_visualizer', enable_rosout=False,
                         start_parameter_services=False)
        self.options = options
        self.started = time.monotonic()
        self.done = False
        self.scan = self.drive = self.mux = self.odom = None
        self.scan_times = deque(maxlen=256)
        self.drive_time = self.mux_time = self.odom_time = None
        self.debug = None
        self.debug_time = None
        self.reason = None
        self.reason_time = None
        self.telemetry = None
        if options.output_file is not None:
            options.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.telemetry = options.output_file.open('a', buffering=1)
        self.output = self.create_publisher(MarkerArray, '/gap_follow_monitor/markers', 1)
        self.subscriptions_held = [
            self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data),
            self.create_subscription(AckermannDriveStamped, '/drive', self.on_drive, 1),
            self.create_subscription(AckermannDriveStamped, '/ackermann_cmd', self.on_mux, 1),
            self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data),
            self.create_subscription(Log, '/rosout', self.on_log, QoSProfile(
                depth=100, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE)),
        ]
        self.timer = self.create_timer(0.5, self.render)
        print('Read-only monitor: publishing /gap_follow_monitor/markers at 2 Hz. '
              'Orange arc is an unchecked /drive curvature projection; log values are ~1 Hz.',
              flush=True)

    def on_scan(self, message):
        self.scan = message
        self.scan_times.append(time.monotonic())

    def on_drive(self, message):
        self.drive, self.drive_time = message.drive, time.monotonic()

    def on_mux(self, message):
        self.mux, self.mux_time = message.drive, time.monotonic()

    def on_odom(self, message):
        self.odom, self.odom_time = message.twist.twist, time.monotonic()

    def on_log(self, message):
        if message.name != 'gap_follow_standalone':
            return
        match = DEBUG.search(message.msg)
        if match:
            try:
                values = tuple(float(value) for value in match.groups())
            except ValueError:
                return
            if all(math.isfinite(value) for value in values):
                self.debug, self.debug_time = values, time.monotonic()
        match = REASON.search(message.msg)
        if match:
            self.reason, self.reason_time = match.group(1), time.monotonic()

    def marker(self, marker_id, kind, color):
        result = Marker()
        result.header.frame_id = self.options.frame
        result.header.stamp = self.get_clock().now().to_msg()
        result.ns, result.id, result.type = 'gap_follow_monitor', marker_id, kind
        result.action = Marker.ADD
        result.pose.orientation.w = 1.0
        result.color.r, result.color.g, result.color.b, result.color.a = color
        result.lifetime.sec = 2
        return result

    @staticmethod
    def command_text(label, command, received, now):
        if command is None:
            return f'{label}: no messages received'
        age = now - received
        stale = ' STALE' if age > 0.5 else ''
        return (f'{label}: v={command.speed:.3f} m/s, steer={command.steering_angle:+.3f} rad'
                f' (age {age:.2f}s){stale}')

    def render(self):
        now = time.monotonic()
        if self.options.duration and now - self.started >= self.options.duration:
            self.done = True
            return
        messages = []
        footprint = self.marker(0, Marker.LINE_STRIP, (1.0, 0.85, 0.1, 1.0))
        footprint.scale.x = 0.025
        front, rear, width = self.options.front, self.options.rear, self.options.half_width
        footprint.points = [Point(x=x, y=y, z=0.03) for x, y in
                            ((-rear, -width), (front, -width), (front, width),
                             (-rear, width), (-rear, -width))]
        messages.append(footprint)

        arc = self.marker(1, Marker.LINE_LIST, (1.0, 0.35, 0.0, 0.9))
        arc.scale.x = 0.035
        fresh = self.drive_time is not None and now - self.drive_time <= 0.5
        if fresh and math.isfinite(self.drive.steering_angle) and abs(self.drive.steering_angle) < 1.4:
            curvature = math.tan(self.drive.steering_angle) / self.options.wheelbase
            # A fixed one-metre illustration of requested curvature, not a
            # forecast of travelled distance or permission to enter that space.
            for index in range(0, 50, 2):
                for distance in (index / 50.0, (index + 1) / 50.0):
                    x = distance if abs(curvature) < 1e-10 else math.sin(curvature * distance) / curvature
                    y = 0.0 if abs(curvature) < 1e-10 else (1 - math.cos(curvature * distance)) / curvature
                    arc.points.append(Point(x=x, y=y, z=0.06))
        else:
            arc.action = Marker.DELETE
        messages.append(arc)

        text = self.marker(2, Marker.TEXT_VIEW_FACING, (0.95, 0.95, 0.95, 1.0))
        text.pose.position.x, text.pose.position.y, text.pose.position.z = -0.3, -2.0, 0.15
        text.scale.z = 0.13
        lines = ['READ-ONLY LIVE MONITOR',
                 'Yellow: configured padded footprint; orange: unchecked /drive curvature',
                 self.command_text('/drive request', self.drive, self.drive_time, now),
                 self.command_text('/ackermann_cmd mux', self.mux, self.mux_time, now)]
        if self.odom is not None:
            lines.append(f'/odom measured vx={self.odom.linear.x:.3f} m/s '
                         f'(age {now - self.odom_time:.2f}s)')
        if self.scan is not None:
            recent = [stamp for stamp in self.scan_times if now - stamp < 2.0]
            rate = ((len(recent) - 1) / (recent[-1] - recent[0])
                    if len(recent) > 1 and recent[-1] > recent[0] else 0.0)
            invalid = sum(not math.isfinite(value) or value <= 0
                          or value < self.scan.range_min or value > self.scan.range_max
                          for value in self.scan.ranges)
            stamp_ns = self.scan.header.stamp.sec * 1_000_000_000 + self.scan.header.stamp.nanosec
            scan_age = ((self.get_clock().now().nanoseconds - stamp_ns) / 1e9
                        if stamp_ns else None)
            sensor_age = 'unstamped' if scan_age is None else f'{scan_age:.3f}s'
            lines.append(f'/scan {rate:.1f} Hz, receipt age {now - self.scan_times[-1]:.2f}s, '
                         f'invalid={invalid}/{len(self.scan.ranges)}, frame={self.scan.header.frame_id}')
            lines.append(f'/scan timestamp age: {sensor_age}')
        else:
            lines.append('/scan: no messages received')
        if self.reason is None:
            lines.append('Planner reason: awaiting next state-change log')
        else:
            lines.append(f'Last reason log: {self.reason} (age {now - self.reason_time:.1f}s)')
        if self.debug is None:
            lines.append('Planner values: awaiting DEBUG rosout (~1 Hz)')
        else:
            front, path, target, steering, speed = self.debug
            lines.extend((f'Logged values (~1 Hz, age {now - self.debug_time:.1f}s; not scan-synchronized):',
                          f'front={front:.2f} m, checked arc={path:.2f} m',
                          f'target={target:+.3f}, steer={steering:+.3f} rad, speed={speed:.3f} m/s'))
        text.text = '\n'.join(lines)
        messages.append(text)
        self.output.publish(MarkerArray(markers=messages))
        if self.telemetry is not None:
            def number(value):
                return float(value) if math.isfinite(value) else None

            def command_state(command, received):
                if command is None:
                    return None
                return {'speed_mps': number(command.speed),
                        'steering_rad': number(command.steering_angle),
                        'receipt_age_s': now - received}

            sample = {'time_unix': time.time(),
                      'drive': command_state(self.drive, self.drive_time),
                      'mux': command_state(self.mux, self.mux_time),
                      'odom_vx_mps': None if self.odom is None else number(self.odom.linear.x),
                      'odom_receipt_age_s': None if self.odom_time is None else now - self.odom_time,
                      'last_reason_log': self.reason,
                      'reason_log_receipt_age_s': None if self.reason_time is None else now - self.reason_time,
                      'debug_log': None if self.debug is None else dict(zip(
                          ('front_m', 'checked_path_m', 'target_rad', 'steering_rad', 'speed_mps'),
                          self.debug)),
                      'debug_log_receipt_age_s': None if self.debug_time is None else now - self.debug_time,
                      'scan': None if self.scan is None else {
                          'rate_hz': rate, 'receipt_age_s': now - self.scan_times[-1],
                          'timestamp_age_s': scan_age, 'invalid_ranges': invalid,
                          'ranges': len(self.scan.ranges)},
                      'synchronization': 'latest received values; debug logs are ~1 Hz, not per-scan traces'}
            self.telemetry.write(json.dumps(sample, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', default='base_link')
    parser.add_argument('--wheelbase', type=float, default=0.33)
    parser.add_argument('--front', type=float, default=0.43)
    parser.add_argument('--rear', type=float, default=0.12)
    parser.add_argument('--half-width', type=float, default=0.25)
    parser.add_argument('--duration', type=float, default=0.0,
                        help='Exit after this many seconds; zero runs until Ctrl+C.')
    parser.add_argument('--output-file', type=Path,
                        help='Optionally append timestamped telemetry summaries as JSONL.')
    options, ros_args = parser.parse_known_args()
    for name in ('wheelbase', 'front', 'rear', 'half_width'):
        value = getattr(options, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name} must be finite and positive')
    if not math.isfinite(options.duration) or options.duration < 0:
        parser.error('duration must be finite and nonnegative')
    rclpy.init(args=ros_args)
    node = GapFollowVisualizer(options)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if node.telemetry is not None:
            node.telemetry.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
