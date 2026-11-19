"""Process supervisor behind the bash entry points (Linux car, standard library + ROS).

Each child owns a process group. Ctrl-C ends the current run, keeps the recorder
alive for the guard's zero flush, then sends SIGINT to rosbag for metadata closure.
"""
import argparse
from dataclasses import fields
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .core import Params

TOPICS = ["/scan", "/odom", "/drive", "/ackermann_cmd", "/joy", "/sensors/core",
          "/commands/motor/speed", "/commands/servo/position", "/sensors/servo_position_command",
          "/car_test/events", "/car_test/policy_drive", "/policy_node/debug"]
EXECUTABLES = dict(grip="grip_circle", straight="straight_runs", steer="steer_steps", **{"policy-sweep": "policy_guard"})


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("test", choices=EXECUTABLES)
    ap.add_argument("--bag-root", type=Path, default=Path.home() / "bags")
    ap.add_argument("--model", help="default: installed rl_policy/models/car_v2_25hz_u1064.npz")
    ap.add_argument("--caps", nargs="+", type=float, default=[1.5, 2.0, 2.5, 3.0])
    for f in fields(Params):
        value = getattr(Params(), f.name)
        name = "--" + f.name.replace("_", "-")
        ap.add_argument(name, type=float if isinstance(value, tuple) else type(value),
                        **({"nargs": "+"} if isinstance(value, tuple) else {}), default=value)
    return ap


def stop_process(proc, timeout=10.0):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"WARNING: PID {proc.pid} ignored SIGINT; sending TERM (bag may need ros2 bag reindex)", flush=True)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def start(command):
    return subprocess.Popen(command, start_new_session=True)


class Probe:
    def __init__(self):
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan, Joy
        from nav_msgs.msg import Odometry
        from vesc_msgs.msg import VescStateStamped
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.ros = rclpy
        self.node = rclpy.create_node("car_tests_preflight")
        self.received, self.rb = {}, True
        def receive(topic, message):
            self.received[topic] = time.monotonic()
            if topic == "/joy":
                self.rb = len(message.buttons) <= 5 or any(message.buttons)
        self.subs = [self.node.create_subscription(cls, topic, lambda m, topic=topic: receive(topic, m), qos_profile_sensor_data)
                     for topic, cls in (("/scan", LaserScan), ("/odom", Odometry), ("/sensors/core", VescStateStamped), ("/joy", Joy))]

    def spin(self):
        self.ros.spin_once(self.node, timeout_sec=0.02)

    def check(self):
        end = time.monotonic() + 8
        while time.monotonic() < end:
            self.spin()
            now = time.monotonic()
            if len(self.received) == 4 and all(now - t < 0.25 for t in self.received.values()) and not self.rb:
                if self.node.count_publishers("/drive"):
                    raise RuntimeError("/drive already has a publisher; stop other autonomy/test nodes first")
                if self.node.count_publishers("/car_test/policy_drive"):
                    raise RuntimeError("private policy topic already has a publisher")
                if not self.node.count_subscribers("/drive"):
                    raise RuntimeError("no /drive subscriber (bringup/mux missing)")
                return
        raise RuntimeError("preflight needs fresh /scan, /odom, /sensors/core, /joy and ALL buttons released")

    def recorder_ready(self, recorder, recorder_name):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            self.spin()
            if recorder.poll() is not None:
                raise RuntimeError("rosbag recorder exited before start")
            # The event publisher appears only when the test starts. /scan already
            # exists; a recorder subscription confirms discovery and startup.
            if any(x.node_name == recorder_name for x in self.node.get_subscriptions_info_by_topic("/scan")):
                return
        raise RuntimeError("recorder did not discover /scan")

    def close(self):
        self.node.destroy_node()
        self.ros.shutdown()


def run_one(args, p, model=None):
    p.validate(args.test)
    key = (f"{p.direction}_d{p.delta}_v{p.v_cap}" if args.test == "grip" else
           "v" + "-".join(map(str, p.targets if args.test == "straight" else p.speeds)) if args.test != "policy-sweep" else
           f"{Path(model).stem}_cap{p.max_speed}")
    if args.test == "steer":
        key += "_d" + "-".join(map(str, p.deltas))
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    bag = args.bag_root.expanduser() / f"{args.test}_{key}_{datetime.now():%Y%m%d_%H%M%S}"
    if bag.exists():
        raise RuntimeError(f"bag already exists: {bag}")
    args.bag_root.expanduser().mkdir(parents=True, exist_ok=True)
    probe = Probe()
    recorder = node = policy = None
    interrupted = []
    handlers = {s: signal.signal(s, lambda sig, frame: interrupted.append(sig)) for s in (signal.SIGINT, signal.SIGTERM)}
    result = 1
    try:
        probe.check()
        recorder_name = f"car_tests_recorder_{os.getpid()}_{time.monotonic_ns()}"
        recorder = start(["ros2", "bag", "record", "--storage", "sqlite3", "--max-cache-size", "8388608",
                          "--node-name", recorder_name, "-o", str(bag)] + TOPICS)
        probe.recorder_ready(recorder, recorder_name)
        if interrupted:
            return 130
        if model:
            policy = start(["ros2", "launch", "rl_policy", "policy.launch.py", "mode:=car",
                            f"max_speed:={p.max_speed}", f"policy_path:={model}", "record:=false",
                            "drive_topic:=/car_test/policy_drive"])
        params = []
        for f in fields(Params):
            params.extend(["-p", f"{f.name}:={json.dumps(getattr(p, f.name))}"])
        # Run the installed console script directly so its PID is the actual
        # publisher: ros2 run's parent could exit before the child's zero tail.
        prefix = subprocess.check_output(["ros2", "pkg", "prefix", "car_tests"], text=True).strip()
        executable = str(Path(prefix) / "lib/car_tests" / EXECUTABLES[args.test])
        node = start([executable, "--ros-args"] + params)
        print(f"Recording {bag}\nRelease, then hold RB. Ctrl-C ends this run.", flush=True)
        while not interrupted and node.poll() is None:
            if recorder.poll() is not None or (policy is not None and policy.poll() is not None):
                raise RuntimeError("recorder or policy exited unexpectedly; stopping test")
            time.sleep(0.05)
        result = node.returncode if node.returncode is not None else 130
    finally:
        # Stop the guarded publisher first, preserving its 0.5 s zero tail in bag.
        stop_process(node)
        stop_process(policy)
        stop_process(recorder)
        probe.close()
        for s, handler in handlers.items():
            signal.signal(s, handler)
        print(f"Bag: {bag}", flush=True)
    if signal.SIGTERM in interrupted:
        raise SystemExit(143)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    values = {f.name: getattr(args, f.name) for f in fields(Params)}
    p = Params(**values).validate(args.test)
    if args.test != "policy-sweep":
        return run_one(args, p)
    if any(not math.isfinite(c) or not 0 < c <= p.max_speed for c in args.caps):
        raise ValueError("all caps must be positive, finite, and <= --max-speed")
    if args.caps != sorted(set(args.caps)):
        raise ValueError("caps must be strictly increasing")
    if args.model:
        model = str(Path(args.model).expanduser().resolve())
    else:
        share = subprocess.check_output(["ros2", "pkg", "prefix", "--share", "rl_policy"], text=True).strip()
        model = str(Path(share) / "models/car_v2_25hz_u1064.npz")
    if not Path(model).is_file():
        raise ValueError(f"model does not exist: {model}")
    launch_args = subprocess.check_output(["ros2", "launch", "rl_policy", "policy.launch.py", "--show-args"], text=True)
    if "'drive_topic'" not in launch_args and '"drive_topic"' not in launch_args:
        raise RuntimeError("installed policy launch lacks drive_topic; rebuild before using the guarded sweep")
    for cap in args.caps:
        response = input(f"\n{Path(model).name}, cap {cap} m/s. Reposition, release ALL buttons. Enter to start; q to quit: ")
        if response.strip().lower() == "q":
            break
        run_one(args, Params(**{**values, "max_speed": cap}), model)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}")
