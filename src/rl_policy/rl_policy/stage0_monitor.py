"""Stage-0 monitor for the ROS sim: laps, stalls, /drive rate and safety
overrides, printed as JSON at the end.

    ros2 run rl_policy stage0_monitor --ros-args -p centerline:=/root/ros2_ws/sim/maps/Duct1077_centerline.csv -p duration:=120.0

Laps come from projecting odometry onto the map's centerline and
accumulating progress. The classic gym zeroes the car's speed on a crash, so
a stall (speed < 0.05 m/s for 1 s while the published command is > 0.3 m/s)
counts as a crash.
"""

import collections
import json
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class Stage0Monitor(Node):
    def __init__(self):
        super().__init__("stage0_monitor")
        self.declare_parameter("centerline", "")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("debug_topic", "/policy_node/debug")
        self.declare_parameter("duration", 120.0)
        cl = np.loadtxt(self.get_parameter("centerline").value, delimiter=",")[:, :2]
        self.cl = cl
        seg = np.hypot(*np.diff(np.vstack([cl, cl[:1]]), axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        self.length = float(seg.sum())
        self.progress = 0.0
        self.last_s = None
        self.max_speed = 0.0
        self.speeds = []
        self.drive_times = []
        self.last_cmd = 0.0
        self.stall_since = None
        self.stalls = 0
        self.overrides = collections.Counter()
        self.debug_msgs = 0
        self.active_ticks = 0
        self.t0 = time.time()
        self.duration = float(self.get_parameter("duration").value)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self.on_odom, 10)
        self.create_subscription(AckermannDriveStamped, self.get_parameter("drive_topic").value, self.on_drive, 10)
        self.create_subscription(String, self.get_parameter("debug_topic").value, self.on_debug, 10)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        i = int(np.argmin(np.hypot(self.cl[:, 0] - p.x, self.cl[:, 1] - p.y)))
        s = self.s[i]
        if self.last_s is not None:
            ds = s - self.last_s
            ds -= self.length * np.round(ds / self.length)
            self.progress += ds
        self.last_s = s
        v = float(msg.twist.twist.linear.x)
        self.speeds.append(v)
        self.max_speed = max(self.max_speed, v)
        now = time.time()
        if abs(v) < 0.05 and self.last_cmd > 0.3:
            if self.stall_since is None:
                self.stall_since = now
            elif now - self.stall_since > 1.0:
                self.stalls += 1
                self.stall_since = None
                self.get_logger().warn(f"stall/crash at ({p.x:.2f}, {p.y:.2f})")
        else:
            self.stall_since = None

    def on_drive(self, msg):
        self.drive_times.append(time.time())
        self.last_cmd = float(msg.drive.speed)

    def on_debug(self, msg):
        d = json.loads(msg.data)
        self.debug_msgs += 1
        self.active_ticks += bool(d.get("active"))
        self.overrides.update(d.get("overrides", []))

    def summary(self):
        dt = np.diff(self.drive_times) if len(self.drive_times) > 1 else np.array([np.nan])
        return {
            "seconds": round(time.time() - self.t0, 1),
            "track_length_m": round(self.length, 2),
            "laps": round(self.progress / self.length, 3),
            "stalls_or_crashes": self.stalls,
            "mean_speed": round(float(np.mean(self.speeds)), 3) if self.speeds else None,
            "max_speed": round(self.max_speed, 3),
            "drive_msgs": len(self.drive_times),
            "drive_rate_hz": round(float(1.0 / np.median(dt)), 2) if len(dt) and np.isfinite(dt).all() else None,
            "drive_period_p99_ms": round(float(np.percentile(dt, 99) * 1000), 1) if len(dt) > 10 else None,
            "debug_msgs": self.debug_msgs,
            "active_ticks": self.active_ticks,
            "overrides": dict(self.overrides),
        }


def main(args=None):
    rclpy.init(args=args)
    node = Stage0Monitor()
    try:
        while rclpy.ok() and time.time() - node.t0 < node.duration:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    print(json.dumps(node.summary(), indent=2), flush=True)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
