"""Safety layer for the RL policy node (spec section 7.3), free of ROS so it
can be unit-tested. The policy's speed request is first shaped (shape_speed):

0. speed_gain        multiplies it (default 1.0, off);
   scale_speed       optional knee mapping onto [0, max_speed] (map_speed());
   speed_smoothing   moving average with the previous request: each tick the
                     request moves this fraction of the way to the new one
                     (default 1.0, off; 0.5 cuts the policy's brake pulses
                     ~3-4x in the sim with no loss in completion or lap time);
   min_speed         raises requests between MIN_SPEED_FROM and min_speed to
                     min_speed (default 0.0, off): the car's motor cannot hold
                     crawling speeds, so a policy asking for 0.1-0.3 m/s sits
                     still and can stay stuck. Requests at or below
                     MIN_SPEED_FROM (the policy stopping) stay as they are.

Then, in this order (each can only lower the speed, down to the brakes' 0):

1. Speed cap         clamp the speed command to max_speed (default 1.5 m/s).
                     With scale_speed, the node first maps the policy's speed
                     onto [0, max_speed] (map_speed()): unchanged below a knee
                     (scale_knee x max_speed), the rest of [0, v_cmd_max]
                     compressed above it, so corner speeds stay as asked and
                     only fast requests ease toward the cap; the clamp stays
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
5. Deadman           the joystick (mux priority 100) overrides this node, and
                     the mux timeout stops the car if it dies. With
                     deadman_button >= 0 the node also commands speed 0 unless
                     that /joy button is held (and /joy is fresh): the policy
                     only drives while the autonomous deadman (RB) is held.
"""

import math
from dataclasses import dataclass, field

import numpy as np

DEFAULT_MAX_SPEED = 1.5  # m/s; do not raise the default
MIN_SPEED_FROM = 0.05  # m/s: min_speed lifts requests above this (lower ones mean "stop")
SPEED_GAIN_RANGE = (0.5, 2.0)
MIN_SPEED_MAX = 2.0  # m/s
STEER_LIMIT = 0.4422  # rad: the car's smaller full lock (servo end stop with gain -0.9, offset 0.552; was 0.4189)
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
    scale_knee: float = 0.7  # fraction of max_speed below which scale_speed leaves the speed unchanged
    speed_gain: float = 1.0  # multiplies the policy's speed request
    min_speed: float = 0.0  # m/s: floor for moving requests (0 = off)
    speed_smoothing: float = 1.0  # fraction of the way to the new request per tick (1 = off)
    scan_timeout: float = SCAN_TIMEOUT
    odom_timeout: float = ODOM_TIMEOUT

    def validate(self):
        if not (0.0 <= self.max_speed):
            raise ValueError("max_speed must be >= 0")
        if not (0.0 < self.steer_limit <= STEER_LIMIT):
            raise ValueError(f"steer_limit must be in (0, {STEER_LIMIT}]")
        if not (self.ttc_threshold >= 0.0 and self.scan_timeout > 0.0 and self.odom_timeout > 0.0):
            raise ValueError("ttc_threshold must be >= 0 and timeouts > 0")
        if not (SPEED_GAIN_RANGE[0] <= self.speed_gain <= SPEED_GAIN_RANGE[1]):
            raise ValueError(f"speed_gain must be in [{SPEED_GAIN_RANGE[0]}, {SPEED_GAIN_RANGE[1]}]")
        if not (0.0 <= self.min_speed <= MIN_SPEED_MAX):
            raise ValueError(f"min_speed must be in [0, {MIN_SPEED_MAX}] m/s")
        if not (0.0 < self.speed_smoothing <= 1.0):
            raise ValueError("speed_smoothing must be in (0, 1]")


@dataclass
class SafetyResult:
    speed: float
    steer: float
    overrides: list = field(default_factory=list)  # names of the rules that fired
    min_ttc: float = math.inf


def map_speed(speed, cfg, v_cmd_max):
    """The policy's speed request mapped onto [0, max_speed] (scale_speed): unchanged up to the
    knee k = scale_knee * max_speed, then [k, v_cmd_max] compressed linearly onto [k, max_speed].
    Identity with scale_speed off or a cap at or above the policy's range (the cap then clamps)."""
    if not (cfg.scale_speed and 0.0 < cfg.max_speed < v_cmd_max):
        return float(speed)
    k = min(max(cfg.scale_knee, 0.0), 1.0) * cfg.max_speed
    if speed <= k:
        return float(speed)
    return float(k + (speed - k) * (cfg.max_speed - k) / (v_cmd_max - k))


def deadman_held(button, buttons, joy_age, timeout):
    """True when the deadman check passes: off (button < 0), or that button held on a /joy message
    no older than ``timeout`` s (no /joy at all, or a stale one, counts as released)."""
    if button < 0:
        return True
    return joy_age <= timeout and button < len(buttons) and bool(buttons[button])


def request_speed(speed, cfg, v_cmd_max, prev=None):
    """The policy's speed request with speed_gain, the scale_speed mapping and speed_smoothing (a
    moving average with ``prev``, the previous value this returned; None: no smoothing)."""
    speed = map_speed(float(speed) * cfg.speed_gain, cfg, v_cmd_max)
    if prev is not None and cfg.speed_smoothing < 1.0:
        speed = prev + cfg.speed_smoothing * (speed - prev)
    return float(speed)


def floor_speed(speed, cfg, raw=None):
    """min_speed: moving requests (> MIN_SPEED_FROM) below it are raised to it. ``raw``: the request
    before smoothing; when the policy itself asks to stop (raw <= MIN_SPEED_FROM) there is no floor,
    so a smoothed request slows the car on its way down instead of being held at min_speed."""
    if raw is not None and raw <= MIN_SPEED_FROM:
        return float(speed)
    if cfg.min_speed > 0.0 and MIN_SPEED_FROM < speed < cfg.min_speed:
        return float(cfg.min_speed)
    return float(speed)


def shape_speed(speed, cfg, v_cmd_max, prev=None):
    """request_speed then floor_speed: what goes to the cap and the brakes (apply_safety)."""
    return floor_speed(request_speed(speed, cfg, v_cmd_max, prev), cfg, request_speed(speed, cfg, v_cmd_max))


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
