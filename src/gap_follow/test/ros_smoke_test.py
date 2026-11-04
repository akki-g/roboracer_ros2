#!/usr/bin/env python3
"""Exercise the ROS wrapper without connecting to the car's command topics.

Run after sourcing ROS and this workspace:
    python3 src/gap_follow/test/ros_smoke_test.py

This separate, opt-in test requires ROS; the ordinary unittest suite does not.
It forces domain 217 and localhost discovery before importing ROS. Both the
controller's scan input and command output are remapped to unique test topics.
No messages are published to /scan, /drive, or the hardware command chain.
"""

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid


def post_compute_freshness():
    """Exercise delayed callbacks in a separate, isolated ROS context.

    Intercept publication before calling on_scan directly. The actual ROS
    endpoints are also remapped, so even an unexpected publication cannot
    enter the car's command chain.
    """
    os.environ['ROS_DOMAIN_ID'] = '217'
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    import rclpy
    from sensor_msgs.msg import LaserScan

    token = uuid.uuid4().hex[:12]
    prefix = f'/gap_freshness_{token}'
    scan_topic, drive_topic = prefix + '/scan', prefix + '/drive'
    controller = Path(__file__).resolve().parents[1] / 'scripts' / 'gap_follow_standalone.py'
    spec = importlib.util.spec_from_file_location(f'gap_freshness_controller_{token}', controller)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rclpy.init(args=['--ros-args',
                    '-r', f'/scan:={scan_topic}', '-r', f'/drive:={drive_topic}',
                    '-r', f'__node:=gap_freshness_{token}', '-p', 'scan_timeout:=0.05'])
    node = None
    try:
        node = module.build_ros_node_class()()
        assert node.publisher.topic_name == drive_topic, node.publisher.topic_name
        assert node.subscription.topic_name == scan_topic, node.subscription.topic_name
        published, calls = [], []
        node.publish_command = published.append

        def delayed_compute(*_args, **_kwargs):
            calls.append(True)
            time.sleep(node.scan_timeout * 2)
            return module.GapCommand(0.1, 0.2, 'ok')

        node.controller.compute = delayed_compute
        for zero_stamp, reason in ((False, 'stale_scan'), (True, 'scan_timeout')):
            published.clear()
            calls.clear()
            message = LaserScan()
            if not zero_stamp:
                message.header.stamp = node.get_clock().now().to_msg()
            node.on_scan(message)
            # The callback must reach the delayed computation; an early stop
            # would not exercise the post-compute freshness guard.
            assert len(calls) == 1, (zero_stamp, calls, published)
            assert len(published) == 1, published
            assert published[0].speed == 0.0 and published[0].reason == reason, published
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def main():
    # Apply isolation in this process and the child even if a sourced shell uses
    # the physical car's domain. Never rely on the caller remembering overrides.
    os.environ['ROS_DOMAIN_ID'] = '217'
    os.environ['ROS_LOCALHOST_ONLY'] = '1'

    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    from sensor_msgs.msg import LaserScan

    token = uuid.uuid4().hex[:12]
    prefix = f'/gap_smoke_{token}'
    scan_topic, drive_topic = prefix + '/scan', prefix + '/drive'
    controller_name = f'gap_smoke_controller_{token}'
    controller = Path(__file__).resolve().parents[1] / 'scripts' / 'gap_follow_standalone.py'
    argv = [sys.executable, str(controller), '--ros-args',
            '-r', f'/scan:={scan_topic}', '-r', f'/drive:={drive_topic}',
            '-r', f'__node:={controller_name}', '-p', 'max_speed:=0.5']

    rclpy.init(args=[])
    node = rclpy.create_node(f'gap_smoke_probe_{token}')
    received = []
    subscription = node.create_subscription(
        AckermannDriveStamped, drive_topic,
        lambda message: received.append(message), 10)
    publisher = node.create_publisher(LaserScan, scan_topic, 1)
    client = node.create_client(SetParameters, f'/{controller_name}/set_parameters')
    process = None

    def spin_until(predicate, timeout=5.0, publish=None):
        deadline = time.monotonic() + timeout
        next_publication = 0.0
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise AssertionError(f'Controller exited early: {process.returncode}')
            if publish is not None and time.monotonic() >= next_publication:
                publisher.publish(publish())
                next_publication = time.monotonic() + 0.1
            rclpy.spin_once(node, timeout_sec=0.02)
            if predicate():
                return
        raise AssertionError('Timed out waiting for the expected ROS behavior')

    def scan(invalid=False, stale=False):
        message = LaserScan()
        now = node.get_clock().now().nanoseconds - (2_000_000_000 if stale else 0)
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(now, 1_000_000_000)
        message.header.frame_id = 'laser'
        message.angle_min = -3 * math.pi / 4
        message.angle_max = 3 * math.pi / 4
        message.angle_increment = (message.angle_max - message.angle_min) / 1080
        message.range_min, message.range_max = 0.05, 30.0
        message.ranges = [float('nan') if invalid else min(
            8.0, 1.2 / max(abs(math.sin(message.angle_min + i * message.angle_increment)), 1e-6))
            for i in range(1081)]
        return message

    def expect_speed(moving, publish=None, timeout=5.0):
        start = len(received)
        spin_until(lambda: any((message.drive.speed > 0) == moving
                               for message in received[start:]),
                   timeout=timeout, publish=publish)

    with tempfile.TemporaryFile(mode='w+') as log:
        def read_log():
            # pread leaves the child's shared stdout file position unchanged.
            return os.pread(log.fileno(), 6000, max(0, os.fstat(log.fileno()).st_size - 6000)).decode(
                'utf-8', errors='replace')

        try:
            process = subprocess.Popen(argv, env=os.environ.copy(), stdout=log,
                                       stderr=subprocess.STDOUT)
            spin_until(lambda: publisher.get_subscription_count() == 1
                       and node.count_publishers(drive_topic) == 1
                       and (controller_name, '/') in node.get_node_names_and_namespaces())
            # Verify the child's graph endpoints as a second check on remapping.
            outputs = node.get_publisher_names_and_types_by_node(controller_name, '/')
            inputs = node.get_subscriber_names_and_types_by_node(controller_name, '/')
            assert drive_topic in dict(outputs) and '/drive' not in dict(outputs), outputs
            assert scan_topic in dict(inputs) and '/scan' not in dict(inputs), inputs

            expect_speed(True, scan)
            assert all(message.drive.speed <= 0.5 for message in received)
            expect_speed(False, lambda: scan(invalid=True))
            expect_speed(True, scan)
            expect_speed(False, lambda: scan(stale=True))
            expect_speed(True, scan)
            # Stop publishing and allow only the wrapper's arrival watchdog to fire.
            expect_speed(False, timeout=2.0)

            assert client.wait_for_service(timeout_sec=5.0), 'Parameter service unavailable'
            request = SetParameters.Request(parameters=[Parameter(
                name='max_speed', value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE, double_value=0.75))])
            future = client.call_async(request)
            spin_until(future.done)
            result = future.result()
            assert len(result.results) == 1 and not result.results[0].successful, result
            output = read_log()
            for reason in ('insufficient_front_data', 'stale_scan', 'scan_timeout'):
                assert f'Gap following: {reason};' in output, output
            guard = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), '--post-compute-freshness'],
                env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=15.0)
            assert guard.returncode == 0, guard.stdout
            print('ROS smoke test passed: isolated remaps, fresh scans, invalid/stale '
                  'stops, arrival watchdog, read-only parameters, and post-compute '
                  'freshness for stamped and unstamped scans.')
        except BaseException:
            print(read_log(), file=sys.stderr)
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.0)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    if sys.argv[1:] == ['--post-compute-freshness']:
        post_compute_freshness()
    else:
        main()
