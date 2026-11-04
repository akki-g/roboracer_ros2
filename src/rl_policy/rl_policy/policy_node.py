"""Phase-1 RL policy node (spec section 7.2).

Subscribes to a LaserScan and Odometry (keeps the latest of each). A 25 Hz
timer builds the observation with the shared obs.py, runs the NumPy actor
(tanh(mean)), scales the action, applies the safety layer (safety.py) and
publishes AckermannDriveStamped. On the car /drive feeds ackermann_mux at
navigation priority (10); the joystick (priority 100) always wins and the
mux timeout stops the car if this node dies.

The published (post-safety) command becomes the next observation's previous
action. The frame stack and previous action reset whenever the node
(re)activates: at start-up, after any watchdog stop, and when the
``enabled`` parameter is switched back on.
"""

import json
import math
import os

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from . import obs as ob
from .numpy_policy import NumpyPolicy
from .safety import DEFAULT_MAX_SPEED, STEER_LIMIT, SafetyConfig, apply_safety, min_ttc

SCAN_ANGLE_TOL = 0.01  # rad, on angle_min
SCAN_INCREMENT_TOL = 2e-5  # rad; f1tenth_gym_ros publishes fov/N instead of fov/(N-1)


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("debug_topic", "~/debug")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("max_speed", DEFAULT_MAX_SPEED)
        self.declare_parameter("steer_limit", STEER_LIMIT)
        self.declare_parameter("ttc_threshold", 0.5)
        self.declare_parameter("ttc_half_angle_deg", 30.0)
        self.declare_parameter("scan_timeout", 0.1)
        self.declare_parameter("odom_timeout", 0.2)
        self.declare_parameter("enabled", True)

        self.safety = self._safety_config()
        self.safety.validate()
        self.enabled = bool(self.get_parameter("enabled").value)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        path = os.path.expanduser(self.get_parameter("policy_path").value)
        self.policy = None
        try:
            self.policy = NumpyPolicy(path)
            self._check_policy_config(self.policy.config)
            self.get_logger().info(
                f"loaded {path} (checkpoint {self.policy.config['checkpoint_sha256'][:12]}, "
                f"update {self.policy.config['checkpoint_update']})"
            )
        except Exception as exc:  # stay up and publish speed 0: the mux will not see a dead node
            self.policy = None
            self.get_logger().error(f"no usable policy at '{path}': {exc}; commanding speed 0")

        rate = self.policy.config["policy_rate_hz"] if self.policy else 25.0
        self.v_cmd_max = self.policy.config["v_cmd_max"] if self.policy else ob.V_CMD_MAX
        self.steer_max = self.policy.config["steer_max"] if self.policy else ob.S_MAX

        self.scan = None  # (receive time, ranges ndarray, angles ndarray, range_min)
        self.scan_ok = False
        self.odom = None  # (receive time, speed)
        self.stack = None
        self.prev_action = ob.reset_action()
        self.active = False
        self.brake_events = 0

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter("drive_topic").value, 10)
        self.debug_pub = self.create_publisher(String, self.get_parameter("debug_topic").value, 10)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value,
                                 self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10)
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"policy_node: {rate:.0f} Hz, max_speed {self.safety.max_speed} m/s, "
            f"scan {self.get_parameter('scan_topic').value}, odom {self.get_parameter('odom_topic').value}, "
            f"drive {self.get_parameter('drive_topic').value}")

    # ---------------------------------------------------------------- params
    def _safety_config(self, overrides=None):
        get = (lambda n: overrides[n]) if overrides else (lambda n: self.get_parameter(n).value)
        return SafetyConfig(
            max_speed=float(get("max_speed")),
            steer_limit=float(get("steer_limit")),
            ttc_threshold=float(get("ttc_threshold")),
            ttc_half_angle=math.radians(float(get("ttc_half_angle_deg"))),
            scan_timeout=float(get("scan_timeout")),
            odom_timeout=float(get("odom_timeout")),
        )

    def _on_set_parameters(self, params):
        names = ("max_speed", "steer_limit", "ttc_threshold", "ttc_half_angle_deg", "scan_timeout", "odom_timeout")
        current = {n: self.get_parameter(n).value for n in names}
        enabled = self.enabled
        for p in params:
            if p.name in names:
                current[p.name] = p.value
            elif p.name == "enabled":
                enabled = bool(p.value)
            elif p.name in ("scan_topic", "odom_topic", "drive_topic", "debug_topic", "policy_path"):
                return SetParametersResult(successful=False, reason=f"{p.name} is fixed at start-up")
        try:
            cfg = self._safety_config(current)
            cfg.validate()
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        self.safety = cfg
        if enabled != self.enabled:
            self.enabled = enabled
            self.active = False  # re-enabling resets the frame stack and previous action
            self.get_logger().info(f"policy {'enabled' if enabled else 'disabled'}")
        return SetParametersResult(successful=True)

    def _check_policy_config(self, cfg):
        if cfg["obs_dim"] != ob.OBS_DIM or cfg["frame_stack"] != ob.FRAME_STACK:
            raise ValueError("policy observation layout differs from obs.py")
        if list(cfg["beam_indices"]) != ob.CAR_BEAM_INDICES.tolist():
            raise ValueError("policy beam indices differ from obs.py")

    # ------------------------------------------------------------- callbacks
    def _on_scan(self, msg: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = len(ranges)
        ok = (
            n == ob.CAR_SCAN_BEAMS
            and abs(msg.angle_min + ob.FOV / 2) < SCAN_ANGLE_TOL
            and abs(msg.angle_increment - ob.FOV / (ob.CAR_SCAN_BEAMS - 1)) < SCAN_INCREMENT_TOL
        )
        if not ok:
            self.get_logger().error(
                f"scan layout {n} beams, angle_min {msg.angle_min:.4f}, increment {msg.angle_increment:.6f} "
                f"!= expected {ob.CAR_SCAN_BEAMS} beams over {ob.FOV} rad; commanding speed 0",
                throttle_duration_sec=2.0)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        self.scan = (now, ranges, angles, float(msg.range_min))
        self.scan_ok = ok

    def _on_odom(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.odom = (now, float(msg.twist.twist.linear.x))

    # ------------------------------------------------------------------ tick
    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        scan_age = now - self.scan[0] if self.scan else math.inf
        odom_age = now - self.odom[0] if self.odom else math.inf
        speed_meas = self.odom[1] if self.odom else 0.0
        fresh = scan_age <= self.safety.scan_timeout and odom_age <= self.safety.odom_timeout
        usable = self.policy is not None and self.scan_ok and self.enabled

        raw_action = None
        speed_cmd, steer_cmd = 0.0, 0.0
        ttc = math.inf
        if self.scan is not None:
            _, ranges, angles, range_min = self.scan
            ttc = min_ttc(ranges, angles, speed_meas, self.safety.ttc_half_angle, range_min)

        if usable and fresh:
            _, ranges, _, _ = self.scan
            scan_norm = ob.normalize_scan(ob.subsample_scan(ranges))
            if not self.active:  # (re)activation: fresh stack, stopped previous action
                self.stack = ob.init_stack(scan_norm)
                self.prev_action = ob.reset_action()
                self.active = True
                self.get_logger().info("policy active: frame stack and previous action reset")
            else:
                self.stack = ob.push_scan(self.stack, scan_norm)
            observation = ob.build_obs(self.stack, speed_meas, self.prev_action)
            raw_action = self.policy(observation)
            speed_cmd, steer_cmd = ob.action_to_command(raw_action, self.v_cmd_max, self.steer_max)
        elif self.active:
            self.active = False
            self.get_logger().warn(
                f"policy inactive (scan age {scan_age:.3f}s, odom age {odom_age:.3f}s, "
                f"scan ok {self.scan_ok}, enabled {self.enabled}); commanding speed 0")

        result = apply_safety(speed_cmd, steer_cmd, self.safety, scan_age=scan_age, odom_age=odom_age,
                              ttc=ttc, valid=usable)
        if "emergency_brake" in result.overrides and raw_action is not None:
            self.brake_events += 1
            self.get_logger().warn(f"emergency brake: TTC {ttc:.2f}s < {self.safety.ttc_threshold}s "
                                   f"at {speed_meas:.2f} m/s (event {self.brake_events})")

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(result.speed)
        msg.drive.steering_angle = float(result.steer)
        self.drive_pub.publish(msg)

        if self.active:
            self.prev_action = ob.command_to_action(result.speed, result.steer, self.v_cmd_max, self.steer_max)

        debug = {
            "t": now,
            "active": self.active,
            "speed_meas": speed_meas,
            "scan_age": None if math.isinf(scan_age) else round(scan_age, 4),
            "odom_age": None if math.isinf(odom_age) else round(odom_age, 4),
            "raw_action": None if raw_action is None else [float(x) for x in raw_action],
            "policy_cmd": [float(speed_cmd), float(steer_cmd)],
            "published": [float(result.speed), float(result.steer)],
            "overrides": result.overrides,
            "min_ttc": None if math.isinf(ttc) else round(ttc, 3),
            "obs_summary": None if self.stack is None else {
                "scan_min_m": float(self.stack[0].min() * ob.MAX_RANGE),
                "scan_front_m": float(self.stack[0][ob.NUM_BEAMS // 2] * ob.MAX_RANGE),
                "scan_mean_m": float(self.stack[0].mean() * ob.MAX_RANGE),
                "speed_norm": speed_meas / ob.V_CMD_MAX,
                "prev_action": [float(x) for x in self.prev_action],
            },
        }
        self.debug_pub.publish(String(data=json.dumps(debug)))

    def stop(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = 0.0
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
