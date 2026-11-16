"""ROS loop test for a new car computer (written for the Kria KV260): the real policy_node,
fed recorded car scans at the LiDAR's rate. No car hardware, and nothing reaches the motor:
every topic is under /board_test, not /scan, /odom or /drive.

    ros2 run rl_policy board_loop_test                    # 20 s
    ros2 run rl_policy board_loop_test --seconds 60 --policy gru1025_obs_u0750.npz
    tools/kv260_test.sh                                   # board_test, then this

Starts policy_node (node name policy_board_test, the car's settings) as a child process,
publishes data/car_replay.npz's 1081-beam scans on /board_test/scan at 40 Hz and its speeds
on /board_test/odom at 50 Hz, and watches /board_test/drive and the node's debug topic.

Pass: commands come out at the policy's rate (within 2%), under 1% of them more than half a
tick late, the node is active with no watchdog stop on 99% of ticks, and its whole tick takes
under half a tick at the 99th percentile and never more than a tick.
"""

import argparse
import array
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import numpy as np

from .board_test import find_share, load_replay

SCAN_HZ = 40.0  # UST-10LX
ODOM_HZ = 50.0  # vesc_to_odom on the car
WARMUP_S = 2.0  # after the node's first command, before measuring
START_TIMEOUT_S = 60.0
NS = "/board_test"
NODE = "policy_board_test"


def _cpu_seconds(pid):
    """User + system CPU time of a process (Linux), or None."""
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20.0, help="measuring time (default 20)")
    ap.add_argument("--policy", default="", help="file in the package's models/ or a full path "
                                                 "(default: the one that drove the recording)")
    a = ap.parse_args(argv)

    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String

    rep = load_replay()
    policy = a.policy or rep["policy"]
    policy_path = pathlib.Path(policy) if os.sep in policy else find_share("models", policy)
    if not policy_path.exists():
        print(f"policy not found: {policy_path}", file=sys.stderr)
        return 2
    with np.load(policy_path, allow_pickle=False) as d:
        tick_s = 1.0 / float(json.loads(str(d["config"]))["policy_rate_hz"])

    class LoopTest(Node):
        def __init__(self):
            super().__init__("board_loop_test")
            n = rep["scans"].shape[1]
            self.scans = []
            for s in rep["scans"]:
                m = LaserScan()
                m.header.frame_id = "laser"
                m.angle_min = rep["angle_min"]
                m.angle_increment = rep["angle_increment"]
                m.angle_max = rep["angle_min"] + (n - 1) * rep["angle_increment"]
                m.scan_time = 1.0 / SCAN_HZ
                m.range_min, m.range_max = rep["range_min"], rep["range_max"]
                m.ranges = array.array("f", s.tobytes())
                self.scans.append(m)
            self.i = 0
            self.odom = Odometry()
            self.odom.header.frame_id, self.odom.child_frame_id = "odom", "base_link"
            self.scan_pub = self.create_publisher(LaserScan, f"{NS}/scan", qos_profile_sensor_data)
            self.odom_pub = self.create_publisher(Odometry, f"{NS}/odom", 10)
            self.create_subscription(AckermannDriveStamped, f"{NS}/drive", self.on_drive, 50)
            self.create_subscription(String, f"/{NODE}/debug", self.on_debug, 50)
            self.create_timer(1.0 / SCAN_HZ, self.send_scan)
            self.create_timer(1.0 / ODOM_HZ, self.send_odom)
            self.drive_t, self.debug, self.measuring = [], [], False

        def send_scan(self):
            m = self.scans[self.i % len(self.scans)]
            m.header.stamp = self.get_clock().now().to_msg()
            self.scan_pub.publish(m)
            self.i += 1

        def send_odom(self):
            self.odom.header.stamp = self.get_clock().now().to_msg()
            self.odom.twist.twist.linear.x = float(rep["speed"][self.i % len(self.scans)])
            self.odom_pub.publish(self.odom)

        def on_drive(self, _msg):
            self.drive_t.append(time.monotonic())

        def on_debug(self, msg):
            if self.measuring:
                self.debug.append(json.loads(msg.data))

    rclpy.init()
    node = LoopTest()
    cmd = [sys.executable, "-m", "rl_policy.policy_node", "--ros-args", "-r", f"__node:={NODE}",
           "-p", f"policy_path:={policy_path}", "-p", f"scan_topic:={NS}/scan", "-p", f"odom_topic:={NS}/odom",
           "-p", f"drive_topic:={NS}/drive", "-p", "max_speed:=10.0", "-p", "ttc_half_angle_deg:=0.0",
           "-p", "scan_resample:=false"]
    print(f"== 3. ROS loop: policy_node ({policy_path.name}) on recorded scans at {SCAN_HZ:g} Hz, "
          f"{a.seconds:g} s (topics under {NS}, nothing on /drive)")
    child = subprocess.Popen(cmd)
    status = 1
    try:
        t0 = time.monotonic()
        while not node.drive_t:  # the node's first command
            rclpy.spin_once(node, timeout_sec=0.05)
            if child.poll() is not None:
                print(f"  FAIL: policy_node exited with code {child.returncode} before publishing")
                return 1
            if time.monotonic() - t0 > START_TIMEOUT_S:
                print(f"  FAIL: no command from policy_node within {START_TIMEOUT_S:g} s")
                return 1
        print(f"  policy_node up after {time.monotonic() - t0:.1f} s")
        t_end = time.monotonic() + WARMUP_S
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.drive_t.clear()
        node.measuring = True
        cpu0, wall0 = _cpu_seconds(child.pid), time.monotonic()
        t_end = wall0 + a.seconds
        while time.monotonic() < t_end and child.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        cpu1, wall1 = _cpu_seconds(child.pid), time.monotonic()
        node.measuring = False
        if child.poll() is not None:
            print(f"  FAIL: policy_node exited with code {child.returncode} during the test")
            return 1

        t = np.asarray(node.drive_t)
        dbg = node.debug
        if len(t) < 10 or len(dbg) < 10:
            print(f"  FAIL: only {len(t)} commands and {len(dbg)} debug messages in {a.seconds:g} s")
            return 1
        periods = np.diff(t) * 1e3
        rate = (len(t) - 1) / (t[-1] - t[0])
        late = float(np.mean(periods > 1.5 * tick_s * 1e3))
        good = float(np.mean([bool(m["active"]) and not any(o.startswith(("watchdog", "invalid")) for o in m["overrides"])
                              for m in dbg]))
        age = np.array([m["scan_age"] for m in dbg if m["scan_age"] is not None]) * 1e3
        last = dbg[-1]
        tick, fwd = last.get("tick_ms"), last.get("policy_ms")
        brakes = sum("emergency_brake" in m["overrides"] for m in dbg)
        print(f"  commands        {rate:6.2f} Hz (policy rate {1 / tick_s:g} Hz), period p50 {np.percentile(periods, 50):.1f} ms, "
              f"p99 {np.percentile(periods, 99):.1f} ms, max {periods.max():.1f} ms; {late:.1%} over {1.5 * tick_s * 1e3:g} ms")
        print(f"  node active     {good:.1%} of {len(dbg)} ticks (no watchdog stop); emergency brake on {brakes}")
        if len(age):
            print(f"  scan age        p50 {np.percentile(age, 50):.1f} ms, p99 {np.percentile(age, 99):.1f} ms at the tick "
                  f"(a scan every {1e3 / SCAN_HZ:g} ms; over 100 ms the watchdog stops the car)")
        if fwd:
            print(f"  forward pass    p50 {fwd['p50']:.3f} ms, p99 {fwd['p99']:.3f} ms (last 500 ticks)")
        if tick:
            print(f"  whole tick      p50 {tick['p50']:.3f} ms, p99 {tick['p99']:.3f} ms, max {tick['max']:.3f} ms "
                  f"(budget: p99 < {tick_s * 500:g} ms, none over {tick_s * 1e3:g} ms)")
        if cpu0 is not None and cpu1 is not None:
            print(f"  policy_node CPU {100 * (cpu1 - cpu0) / (wall1 - wall0):.0f}% of one core ({os.cpu_count()} cores)")
        problems = []
        if abs(rate * tick_s - 1.0) > 0.02:
            problems.append(f"command rate {rate:.2f} Hz is not the policy's {1 / tick_s:g} Hz")
        if late > 0.01:
            problems.append(f"{late:.1%} of commands more than half a tick late")
        if good < 0.99:
            problems.append(f"node active on only {good:.1%} of ticks")
        if not tick:
            problems.append("no tick_ms in the debug topic (old policy_node?)")
        elif tick["p99"] >= tick_s * 500 or tick["max"] > tick_s * 1e3:
            problems.append(f"tick time p99 {tick['p99']} ms, max {tick['max']} ms")
        for p in problems:
            print(f"  FAIL: {p}")
        if not problems:
            print("  PASS")
        status = 1 if problems else 0
        print("== ROS loop test: " + ("PASS" if status == 0 else "FAIL"))
        return status
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
