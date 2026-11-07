"""Phase-1 RL policy node (spec section 7.2).

Subscribes to a LaserScan and Odometry (keeps the latest of each). A 25 Hz
timer builds the observation with the shared obs.py, runs the NumPy actor
(tanh(mean)), scales the action, applies the safety layer (safety.py) and
publishes AckermannDriveStamped. On the car /drive feeds ackermann_mux at
navigation priority (10); the joystick (priority 100) always wins and the
mux timeout stops the car if this node dies.

The published (post-safety) command becomes the next observation's previous
action, also with scale_speed: the previous action then agrees with the measured
speed (feeding back the unscaled request paired an 8 m/s request with a ~3 m/s
speed on every tick, which the policy never saw in training, and the car crashed). The frame stack and previous action reset whenever the node
(re)activates: at start-up, after any watchdog stop, and when the
``enabled`` parameter is switched back on.
"""

import json
import math
import os
import time
from collections import deque

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String

from . import obs as ob
from .numpy_policy import NumpyPolicy
from .safety import (DEFAULT_MAX_SPEED, STEER_LIMIT, SafetyConfig, apply_safety, deadman_held, floor_speed, min_ttc,
                     request_speed)

# Training ends an episode after 2 s below 0.2 m/s; the node mirrors that by
# resetting the stack, previous action and recurrent state (a fresh episode),
# e.g. while the joystick holds the car before the autonomy deadman is pressed.
STALL_SPEED = 0.2  # m/s
STALL_TICKS = 50  # 2 s at 25 Hz
SCAN_ANGLE_TOL = 0.01  # rad, on angle_min
SCAN_INCREMENT_TOL = 2e-5  # rad; f1tenth_gym_ros publishes fov/N instead of fov/(N-1)
def policy_angles(num_beams=ob.NUM_BEAMS):
    """The policy's beam angles: num_beams evenly over [-FOV/2, FOV/2]."""
    return -ob.FOV / 2 + np.arange(num_beams) * (ob.FOV / (num_beams - 1))


def resample_indices(angle_min, angle_increment, n, angles=None):
    """Nearest beam of an n-beam scan to each policy angle, and the worst angle error."""
    angles = policy_angles() if angles is None else angles
    idx = np.clip(np.rint((angles - angle_min) / angle_increment), 0, n - 1).astype(int)
    err = np.max(np.abs(angle_min + idx * angle_increment - angles))
    return idx, float(err)


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
        self.declare_parameter("ttc_brake", True)
        # Map the policy's 0..v_cmd_max speed range onto 0..max_speed instead of clamping.
        self.declare_parameter("scale_speed", False)
        self.declare_parameter("scale_knee", 0.7)  # scale_speed: unchanged below this fraction of max_speed
        self.declare_parameter("speed_gain", 1.0)  # multiplies the policy's speed request
        self.declare_parameter("min_speed", 0.0)  # m/s: floor for moving requests (0 = off)
        self.declare_parameter("speed_smoothing", 1.0)  # fraction of the way to a new speed request per tick (1 = off)
        self.declare_parameter("scan_timeout", 0.1)
        self.declare_parameter("odom_timeout", 0.2)
        self.declare_parameter("enabled", True)
        # >= 0: drive only while this /joy button is held (RB = 5, joy_teleop's autonomous deadman); -1: off
        self.declare_parameter("deadman_button", -1)
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("joy_timeout", 0.5)  # s: an older /joy counts as released
        # Off on the car: anything but the UST-10LX's 1081 beams stops the car.
        # On in the ROS sim, whose LiDAR is fixed at 1080 beams over 4.7 rad:
        # take the nearest beam to each policy angle if within resample_tol_deg.
        self.declare_parameter("scan_resample", False)
        self.declare_parameter("resample_tol_deg", 0.5)

        self.safety = self._safety_config()
        self.safety.validate()
        self.enabled = bool(self.get_parameter("enabled").value)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # observation layout: obs.py defaults until a policy says otherwise
        self.beam_idx, self.num_beams, self.frame_stack = ob.CAR_BEAM_INDICES, ob.NUM_BEAMS, ob.FRAME_STACK
        self.policy_angles = policy_angles(self.num_beams)
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
        self.scan_idx = self.beam_idx
        self.scan_resample = bool(self.get_parameter("scan_resample").value)
        self.resample_tol = math.radians(float(self.get_parameter("resample_tol_deg").value))
        self._layout = None
        self.odom = None  # (receive time, speed)
        self.stack = None
        self.prev_action = ob.reset_action()
        self.active = False
        self.brake_events = 0
        self.forward_ms = deque(maxlen=500)  # policy forward-pass times, for the debug topic
        self.stopped_ticks = 0
        self.stall_resets = 0
        self.speed_request = 0.0  # speed_smoothing state: the previous smoothed request
        self.deadman_button = int(self.get_parameter("deadman_button").value)
        self.joy_timeout = float(self.get_parameter("joy_timeout").value)
        self.joy = None  # (receive time, buttons)
        self.deadman_ok = self.deadman_button < 0

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter("drive_topic").value, 10)
        self.debug_pub = self.create_publisher(String, self.get_parameter("debug_topic").value, 10)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value,
                                 self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10)
        if self.deadman_button >= 0:
            self.create_subscription(Joy, self.get_parameter("joy_topic").value, self._on_joy, 10)
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"policy_node: {rate:.0f} Hz, max_speed {self.safety.max_speed} m/s "
            f"({'scaled' if self.safety.scale_speed else 'clamped'}), "
            f"speed gain {self.safety.speed_gain:g}, min speed {self.safety.min_speed:g} m/s, "
            f"speed smoothing {self.safety.speed_smoothing:g}, "
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
            ttc_brake=bool(get("ttc_brake")),
            scale_speed=bool(get("scale_speed")),
            scale_knee=float(get("scale_knee")),
            speed_gain=float(get("speed_gain")),
            min_speed=float(get("min_speed")),
            speed_smoothing=float(get("speed_smoothing")),
            scan_timeout=float(get("scan_timeout")),
            odom_timeout=float(get("odom_timeout")),
        )

    def _on_set_parameters(self, params):
        names = ("max_speed", "steer_limit", "ttc_threshold", "ttc_half_angle_deg", "ttc_brake",
                 "scale_speed", "scale_knee", "speed_gain", "min_speed", "speed_smoothing", "scan_timeout",
                 "odom_timeout")
        current = {n: self.get_parameter(n).value for n in names}
        enabled = self.enabled
        for p in params:
            if p.name in names:
                current[p.name] = p.value
            elif p.name == "enabled":
                enabled = bool(p.value)
            elif p.name in ("scan_topic", "odom_topic", "drive_topic", "debug_topic", "policy_path",
                            "scan_resample", "resample_tol_deg"):
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
        """Take the observation layout from the policy (phase-1 files: 109 beams,
        stack 3; phase 1c: 271 beams) and check it is one obs.py can build."""
        arch = cfg.get("arch", "mlp")
        if arch not in ("mlp", "cnn_gru"):
            raise ValueError(f"policy arch {arch!r} is not supported by this node")
        if arch == "cnn_gru" and int(cfg["frame_stack"]) != 1:
            raise ValueError("a cnn_gru policy takes one scan per step (frame_stack 1)")
        idx = np.asarray(cfg["beam_indices"], dtype=int)
        if idx.tolist() != ob.beam_indices(len(idx)).tolist():
            raise ValueError("policy beam indices are not evenly spaced over the car's 1081 beams")
        frame_stack = int(cfg["frame_stack"])
        if cfg["obs_dim"] != ob.obs_dim(frame_stack, len(idx)):
            raise ValueError("policy obs_dim does not match its beams and frame stack")
        self.beam_idx, self.num_beams, self.frame_stack = idx, len(idx), frame_stack
        self.policy_angles = policy_angles(self.num_beams)

    # ------------------------------------------------------------- callbacks
    def _on_scan(self, msg: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = len(ranges)
        layout = (n, msg.angle_min, msg.angle_increment)
        if layout != self._layout:  # validate once per layout
            self._layout = layout
            ust10lx = (
                n == ob.CAR_SCAN_BEAMS
                and abs(msg.angle_min + ob.FOV / 2) < SCAN_ANGLE_TOL
                and abs(msg.angle_increment - ob.FOV / (ob.CAR_SCAN_BEAMS - 1)) < SCAN_INCREMENT_TOL
            )
            if ust10lx:
                self.scan_idx, self._layout_ok = self.beam_idx, True
            elif self.scan_resample and n > 1 and msg.angle_increment > 0:
                self.scan_idx, err = resample_indices(msg.angle_min, msg.angle_increment, n, self.policy_angles)
                self._layout_ok = err <= self.resample_tol
                log = self.get_logger().warn if self._layout_ok else self.get_logger().error
                log(f"scan has {n} beams from {msg.angle_min:.4f} rad: resampling to the policy's "
                    f"{self.num_beams} angles, worst error {math.degrees(err):.2f} deg "
                    f"({'ok' if self._layout_ok else 'too large, commanding speed 0'})")
            else:
                self._layout_ok = False
            if not self._layout_ok and not self.scan_resample:
                self.get_logger().error(
                    f"scan layout {n} beams, angle_min {msg.angle_min:.4f}, increment {msg.angle_increment:.6f} "
                    f"!= expected {ob.CAR_SCAN_BEAMS} beams over {ob.FOV} rad; commanding speed 0")
        ok = self._layout_ok
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        self.scan = (now, ranges, angles, float(msg.range_min))
        self.scan_ok = ok

    def _on_joy(self, msg: Joy):
        self.joy = (self.get_clock().now().nanoseconds * 1e-9, list(msg.buttons))

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
        joy_age = now - self.joy[0] if self.joy else math.inf
        deadman_ok = deadman_held(self.deadman_button, self.joy[1] if self.joy else [], joy_age, self.joy_timeout)
        if deadman_ok != self.deadman_ok:
            self.deadman_ok = deadman_ok
            self.get_logger().info(f"deadman (button {self.deadman_button}) "
                                   f"{'held: policy drives' if deadman_ok else 'released: speed 0'}")
        usable = self.policy is not None and self.scan_ok and self.enabled and deadman_ok

        raw_action = None
        speed_cmd, steer_cmd = 0.0, 0.0
        policy_speed = 0.0
        ttc = math.inf
        if self.scan is not None:
            _, ranges, angles, range_min = self.scan
            ttc = min_ttc(ranges, angles, speed_meas, self.safety.ttc_half_angle, range_min)

        if usable and fresh:
            _, ranges, _, _ = self.scan
            scan_norm = ob.normalize_scan(ranges[self.scan_idx])
            self.stopped_ticks = self.stopped_ticks + 1 if abs(speed_meas) < STALL_SPEED else 0
            stall_reset = self.active and self.stopped_ticks >= STALL_TICKS
            if stall_reset:  # as training's stall end: a new episode
                self.active = False
                self.stopped_ticks = 0
                self.stall_resets += 1
                if self.stall_resets <= 3 or self.stall_resets % 50 == 0:
                    self.get_logger().info(f"stopped for {STALL_TICKS / 25:.0f} s: starting a fresh episode "
                                           f"(reset {self.stall_resets})")
            if not self.active:  # (re)activation: fresh stack, stopped previous action, zero GRU state
                self.stack = ob.init_stack(scan_norm, self.frame_stack)
                self.prev_action = ob.reset_action()
                self.policy.reset()  # recurrent policies: start-up, watchdog recovery, enabled toggled on
                self.speed_request = 0.0  # the car starts from rest
                self.active = True
                if not stall_reset:
                    self.get_logger().info("policy active: frame stack, previous action and recurrent state reset")
            else:
                self.stack = ob.push_scan(self.stack, scan_norm)
            observation = ob.build_obs(self.stack, speed_meas, self.prev_action)
            t0 = time.perf_counter()
            raw_action = self.policy(observation)  # steps a GRU even on ticks the safety layer overrides
            self.forward_ms.append((time.perf_counter() - t0) * 1e3)
            speed_cmd, steer_cmd = ob.action_to_command(raw_action, self.v_cmd_max, self.steer_max)
            policy_speed = float(speed_cmd)
            raw_request = request_speed(policy_speed, self.safety, self.v_cmd_max)  # gain, scale_speed
            self.speed_request = request_speed(policy_speed, self.safety, self.v_cmd_max, self.speed_request)  # + smoothing
            speed_cmd = floor_speed(self.speed_request, self.safety, raw_request)  # + min_speed
        elif self.active:
            self.active = False
            self.get_logger().warn(
                f"policy inactive (scan age {scan_age:.3f}s, odom age {odom_age:.3f}s, "
                f"scan ok {self.scan_ok}, enabled {self.enabled}, deadman {self.deadman_ok}); commanding speed 0")

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
            "deadman": self.deadman_ok,
            "speed_meas": speed_meas,
            "scan_age": None if math.isinf(scan_age) else round(scan_age, 4),
            "odom_age": None if math.isinf(odom_age) else round(odom_age, 4),
            "raw_action": None if raw_action is None else [float(x) for x in raw_action],
            "policy_cmd": [policy_speed, float(steer_cmd)],
            "speed_mapped": float(speed_cmd),
            "published": [float(result.speed), float(result.steer)],
            "overrides": result.overrides,
            "min_ttc": None if math.isinf(ttc) else round(ttc, 3),
            "stall_resets": self.stall_resets,
            "policy_ms": None if not self.forward_ms else {
                "last": round(self.forward_ms[-1], 3),
                "p50": round(float(np.percentile(self.forward_ms, 50)), 3),
                "p99": round(float(np.percentile(self.forward_ms, 99)), 3)},
            "gru_h_norm": (round(float(np.linalg.norm(self.policy.h)), 3)
                           if self.policy is not None and getattr(self.policy, "arch", "mlp") == "cnn_gru" else None),
            "obs_summary": None if self.stack is None else {
                "scan_min_m": float(self.stack[0].min() * ob.MAX_RANGE),
                "scan_front_m": float(self.stack[0][self.num_beams // 2] * ob.MAX_RANGE),
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
