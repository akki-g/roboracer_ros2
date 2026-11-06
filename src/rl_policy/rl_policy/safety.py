"""Safety layer for the RL policy node (spec section 7.3), free of ROS so it
can be unit-tested. Applied after the policy, in this order:

1. Speed cap         clamp the speed command to max_speed (default 1.5 m/s).
                     With scale_speed, the node first scales the policy's speed
                     range [0, v_cmd_max] onto [0, max_speed] (speed_scale()), so
                     the policy's slow-downs survive a low cap; the clamp stays
                     as a backstop.
2. Steering clamp    +-steer_limit (0.4189 rad)
3. Emergency brake   time-to-collision over the beams within +-30 deg of
                     straight ahead, TTC = range / max(v cos(theta), 0.01);
                     below ttc_threshold (0.5 s) command speed 0.
                     ttc_brake=False switches it off: it fires on normal
                     cornering lines in narrow tracks, and training has no
                     brake (results note, D23). Use it on early car runs.
4. Watchdog          scan older than 100 ms or odometry older than 200 ms
                     (or no valid scan / odometry / policy) -> speed 0
5. Deadman           not in this node: the joystick (mux priority 100)
                     overrides it, and the mux timeout stops the car if this
                     node dies.
"""

import math
from dataclasses import dataclass, field

import numpy as np

DEFAULT_MAX_SPEED = 1.5  # m/s; do not raise the default
STEER_LIMIT = 0.4189  # rad
TTC_THRESHOLD = 0.5  # s
TTC_HALF_ANGLE = math.radians(30.0)
SCAN_TIMEOUT = 0.1  # s
ODOM_TIMEOUT = 0.2  # s


@dataclass
class SafetyConfig:
    max_speed: float = DEFAULT_MAX_SPEED
    steer_limit: float = STEER_LIMIT
    ttc_threshold: float = TTC_THRESHOLD
    ttc_half_angle: float = TTC_HALF_ANGLE
    ttc_brake: bool = True
    scale_speed: bool = False
    scan_timeout: float = SCAN_TIMEOUT
    odom_timeout: float = ODOM_TIMEOUT

    def validate(self):
        if not (0.0 <= self.max_speed):
            raise ValueError("max_speed must be >= 0")
        if not (0.0 < self.steer_limit <= STEER_LIMIT):
            raise ValueError(f"steer_limit must be in (0, {STEER_LIMIT}]")
        if not (self.ttc_threshold >= 0.0 and self.scan_timeout > 0.0 and self.odom_timeout > 0.0):
            raise ValueError("ttc_threshold must be >= 0 and timeouts > 0")


@dataclass
class SafetyResult:
    speed: float
    steer: float
    overrides: list = field(default_factory=list)  # names of the rules that fired
    min_ttc: float = math.inf


def speed_scale(cfg, v_cmd_max):
    """Factor that maps the policy's speed range [0, v_cmd_max] onto [0, max_speed]:
    max_speed / v_cmd_max with scale_speed on and a cap below the policy's range,
    else 1.0 (the cap clamps)."""
    if cfg.scale_speed and 0.0 < cfg.max_speed < v_cmd_max:
        return cfg.max_speed / v_cmd_max
    return 1.0


def min_ttc(ranges, angles, speed, half_angle=TTC_HALF_ANGLE, range_min=0.0):
    """Smallest TTC over beams with |angle| <= half_angle.

    NaN, +-inf (no return) and readings below the sensor's range_min are
    ignored. TTC = r / max(v cos(theta), 0.01).
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    sel = (np.abs(angles) <= half_angle) & np.isfinite(ranges) & (ranges >= range_min)
    if not np.any(sel):
        return math.inf
    closing = np.maximum(speed * np.cos(angles[sel]), 0.01)
    return float(np.min(ranges[sel] / closing))


def apply_safety(speed_cmd, steer_cmd, cfg, *, scan_age, odom_age, ttc, valid=True):
    """Return the command to publish and which rules changed it.

    scan_age / odom_age: seconds since the latest message (inf if none yet).
    ttc: min_ttc() of the latest scan at the current speed.
    valid: False if the policy or scan layout is unusable.
    """
    overrides = []
    speed, steer = float(speed_cmd), float(steer_cmd)
    if not (math.isfinite(speed) and math.isfinite(steer)):
        speed, steer = 0.0, 0.0
        overrides.append("non_finite_action")
    # 1. speed cap (also keeps reverse out: the policy range is [0, v_cmd_max])
    capped = min(max(speed, 0.0), cfg.max_speed)
    if capped != speed:
        overrides.append("speed_cap")
    speed = capped
    # 2. steering clamp
    clamped = min(max(steer, -cfg.steer_limit), cfg.steer_limit)
    if clamped != steer:
        overrides.append("steer_clamp")
    steer = clamped
    # 3. emergency brake
    if cfg.ttc_brake and ttc < cfg.ttc_threshold:
        speed = 0.0
        overrides.append("emergency_brake")
    # 4. watchdog
    if scan_age > cfg.scan_timeout:
        speed = 0.0
        overrides.append("watchdog_scan")
    if odom_age > cfg.odom_timeout:
        speed = 0.0
        overrides.append("watchdog_odom")
    if not valid:
        speed = 0.0
        overrides.append("invalid_input")
    return SafetyResult(speed, steer, overrides, ttc)
