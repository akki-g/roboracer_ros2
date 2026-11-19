"""ROS transport only: monotonic watchdogs and a live context during zero-on-exit."""
from dataclasses import fields
import json
import signal
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from vesc_msgs.msg import VescStateStamped

from .core import Engine, Params


class TestNode(Node):
    def __init__(self, test):
        super().__init__(test.replace("-", "_") + "_test")
        values = {}
        for f in fields(Params):
            default = getattr(Params(), f.name)
            self.declare_parameter(f.name, list(default) if isinstance(default, tuple) else default)
            value = self.get_parameter(f.name).value
            values[f.name] = tuple(value) if isinstance(default, tuple) else value
        self.engine = Engine(test, Params(**values))
        self.pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.events = self.create_publisher(String, "/car_test/events", 10)
        self.recording_ready = False
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Joy, "/joy", self.on_joy, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", lambda m: self.engine.odom(m.twist.twist.linear.x, time.monotonic()), qos_profile_sensor_data)
        self.create_subscription(VescStateStamped, "/sensors/core",
                                 lambda m: self.engine.sensor(m.state.speed, m.state.fault_code, time.monotonic()), qos_profile_sensor_data)
        if test == "policy-sweep":
            self.create_subscription(AckermannDriveStamped, "/car_test/policy_drive",
                                     lambda m: self.engine.policy(m.drive.speed, m.drive.steering_angle, time.monotonic()), 10)
        self.get_logger().info("Waiting for recorder subscriptions. Keep all buttons released.")

    def on_joy(self, msg):
        if not self.recording_ready:
            # A press before discovery must not become a delayed automatic start.
            self.engine.rb = len(msg.buttons) > 5 and msg.buttons[5] == 1
            self.engine.released = False
            self.engine.joy_t = time.monotonic()
            return
        self.engine.joy(msg.buttons, time.monotonic())

    def check_recording(self):
        if self.recording_ready:
            return
        names = []
        for topic in ("/drive", "/car_test/events"):
            names.append({x.node_name for x in self.get_subscriptions_info_by_topic(topic)
                          if x.node_name.startswith(("car_tests_recorder_", "rosbag2_recorder"))})
        if names[0] & names[1]:
            self.recording_ready = True
            self.get_logger().info("Recorder ready. Release all buttons, then hold RB. Release RB to ABORT; restart required after a fault.")

    def on_scan(self, msg):
        self.engine.scan(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max, time.monotonic())

    def publish(self, command):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed, msg.drive.steering_angle = map(float, command)
        self.pub.publish(msg)

    def flush_events(self):
        while self.engine.events:
            event = self.engine.events.pop(0)
            self.events.publish(String(data=json.dumps(event, allow_nan=False)))
            self.get_logger().info(f"run {event['run']}: {event['phase']} {event['stop_reason']}")

    def heartbeat(self, now):
        # Repeat current params/targets for a recorder discovering this new topic
        # after the initial waiting marker. 'heartbeat' never denotes a transition.
        from dataclasses import asdict
        event = dict(test=self.engine.test, phase="heartbeat", state=self.engine.phase,
                     time=now, run=self.engine.run, targets=dict(speed=self.engine.target,
                     steering=self.engine.last_output[1]), stop_reason=self.engine.reason,
                     params=asdict(self.engine.p))
        self.events.publish(String(data=json.dumps(event, allow_nan=False)))


def main(test):
    # rclpy's default SIGINT shuts down DDS before a final stop can be delivered.
    # Handle INT/TERM ourselves and keep publishing zero for 0.5 s before shutdown.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    interrupted = []
    previous = {s: signal.signal(s, lambda sig, frame: interrupted.append(sig)) for s in (signal.SIGINT, signal.SIGTERM)}
    reason, result = "completed", 0
    try:
        node = TestNode(test)
        period = 1.0 / node.engine.p.rate_hz
        due = time.monotonic()
        heartbeat_due = due
        discovery_due = due
        while rclpy.ok() and not interrupted:
            rclpy.spin_once(node, timeout_sec=max(0.0, min(period, due - time.monotonic())))
            now = time.monotonic()
            if now >= discovery_due:
                node.check_recording()
                discovery_due = now + .1
            if now >= due:
                node.publish(node.engine.tick(now))
                if node.recording_ready:
                    node.flush_events()
                if node.recording_ready and now >= heartbeat_due:
                    node.heartbeat(now)
                    heartbeat_due = now + 1.0
                due = now + period
                if node.engine.terminal:
                    reason = node.engine.reason or "completed"
                    result = 2 if node.engine.phase == "aborted" else 0
                    break
        if interrupted:
            reason = "SIGINT" if interrupted[0] == signal.SIGINT else "SIGTERM"
    except BaseException:
        reason = "exception"
        raise
    finally:
        if node is not None and rclpy.ok():
            for command in node.engine.exit_commands(reason, time.monotonic()):
                node.publish(command)
                node.flush_events()
                time.sleep(1.0 / node.engine.p.rate_hz)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return result


def grip_main():
    return main("grip")


def straight_main():
    return main("straight")


def steer_main():
    return main("steer")


def policy_main():
    return main("policy-sweep")
