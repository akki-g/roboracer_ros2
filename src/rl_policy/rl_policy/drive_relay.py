"""Relay drive commands from the ROS sim to the real car over UDP (car on a stand: hardware-in-the-loop).

In the sim (VM / container), forward the sim policy's /drive:
    ros2 run rl_policy drive_relay --ros-args -p mode:=send -p host:=<car ip>
On the car, with the bringup (joystick, mux, VESC) running and the policy node NOT running:
    ros2 run rl_policy drive_relay --ros-args -p mode:=receive

The receiver publishes the latest command on /drive (the mux's navigation input) only while it is fresh
(timeout 0.3 s) and joystick button deadman_button (5 = RB) is held; otherwise speed 0, steering 0.
Speed is clamped to [0, max_speed], steering to +-STEER_LIMIT. tools/sim_gru.sh starts the sender when
CAR_HOST is set. UDP rather than ROS discovery: VMware's NAT does not pass DDS multicast to the car.
"""

import math
import socket
import struct
import time

PACKET = struct.Struct("!Idd")  # sequence number, speed (m/s), steering angle (rad)
DEFAULT_PORT = 5555


def encode(seq, speed, steer):
    return PACKET.pack(seq & 0xFFFFFFFF, float(speed), float(steer))


def decode(data):
    """(seq, speed, steer), or None for a malformed or non-finite packet."""
    if len(data) != PACKET.size:
        return None
    seq, speed, steer = PACKET.unpack(data)
    if not (math.isfinite(speed) and math.isfinite(steer)):
        return None
    return seq, speed, steer


def relay_command(last, age, timeout, deadman_ok, max_speed, steer_limit):
    """(speed, steer) to publish: the last received (speed, steer) when fresh and the deadman is held,
    clamped; otherwise a stop."""
    if last is None or age > timeout or not deadman_ok:
        return 0.0, 0.0
    speed, steer = last
    return min(max(speed, 0.0), max_speed), min(max(steer, -steer_limit), steer_limit)


def main():
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from rclpy.node import Node
    from sensor_msgs.msg import Joy

    from .safety import STEER_LIMIT, deadman_held

    class DriveRelay(Node):
        def __init__(self):
            super().__init__("drive_relay")
            self.mode = self.declare_parameter("mode", "receive").value
            self.port = int(self.declare_parameter("port", DEFAULT_PORT).value)
            topic = self.declare_parameter("drive_topic", "/drive").value
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.mode == "send":
                self.host = self.declare_parameter("host", "").value
                if not self.host:
                    raise ValueError("mode:=send needs host:=<car ip>")
                self.seq = 0
                self.create_subscription(AckermannDriveStamped, topic, self.on_drive, 10)
                self.get_logger().info(f"forwarding {topic} to {self.host}:{self.port}")
            elif self.mode == "receive":
                self.timeout = float(self.declare_parameter("timeout", 0.3).value)
                self.max_speed = float(self.declare_parameter("max_speed", 10.0).value)
                self.deadman_button = int(self.declare_parameter("deadman_button", 5).value)
                self.joy_timeout = float(self.declare_parameter("joy_timeout", 0.5).value)
                self.sock.bind(("0.0.0.0", self.port))
                self.sock.setblocking(False)
                self.last, self.last_t, self.joy, self.packets = None, -math.inf, None, 0
                self.pub = self.create_publisher(AckermannDriveStamped, topic, 10)
                if self.deadman_button >= 0:
                    self.create_subscription(Joy, self.declare_parameter("joy_topic", "/joy").value, self.on_joy, 10)
                self.create_timer(0.02, self.tick)
                self.create_timer(2.0, self.report)
                self.get_logger().info(f"receiving on UDP {self.port}, publishing {topic} while button "
                                       f"{self.deadman_button} is held")
            else:
                raise ValueError("mode must be send or receive")

        def on_drive(self, msg):
            self.sock.sendto(encode(self.seq, msg.drive.speed, msg.drive.steering_angle), (self.host, self.port))
            self.seq += 1

        def on_joy(self, msg):
            self.joy = (time.monotonic(), list(msg.buttons))

        def tick(self):
            while True:
                try:
                    data, _ = self.sock.recvfrom(64)
                except BlockingIOError:
                    break
                pkt = decode(data)
                if pkt is not None:
                    self.last, self.last_t = pkt[1:], time.monotonic()
                    self.packets += 1
            now = time.monotonic()
            joy_age = now - self.joy[0] if self.joy else math.inf
            ok = deadman_held(self.deadman_button, self.joy[1] if self.joy else [], joy_age, self.joy_timeout)
            speed, steer = relay_command(self.last, now - self.last_t, self.timeout, ok, self.max_speed, STEER_LIMIT)
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.drive.speed, msg.drive.steering_angle = float(speed), float(steer)
            self.pub.publish(msg)

        def report(self):
            age = time.monotonic() - self.last_t
            self.get_logger().info(f"{self.packets} packets in 2 s, last {age:.2f} s ago, last command "
                                   f"{self.last if self.last else '-'}")
            self.packets = 0

    rclpy.init()
    node = DriveRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
