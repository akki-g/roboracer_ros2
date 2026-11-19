"""Deterministic maneuvers and fail-closed safety, using monotonic seconds only.

No ROS or NumPy dependencies. A fault latches until this Engine is discarded.
Completed straight/steer runs need a release and NEW RB edge for the next run.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Params:
    rate_hz: float = 50.0
    max_speed: float = 3.5
    steer_limit: float = 0.42
    max_duration: float = 120.0
    stale_s: float = 0.25
    cone_deg: float = 10.0
    latency: float = 0.25
    a_brake: float = 3.0
    margin: float = 0.5
    min_clearance: float = 0.35
    short_reject: float = 0.05  # m: readings below this are LiDAR noise (scan_distances)
    cluster_beams: int = 3  # an obstacle must span this many adjacent beams (0.25 deg each)
    lat_acc_cap: float = 5.0
    wheelbase: float = 0.33
    lidar_offset: float = 0.265
    erpm_gain: float = 4675.0
    erpm_offset: float = 220.0
    v_start: float = 1.0
    ramp: float = 0.05
    v_cap: float = 3.5
    delta: float = 0.30
    direction: str = "left"
    targets: tuple = (1.5, 2.0, 2.5, 3.0)
    speeds: tuple = (1.0, 1.5, 2.0)
    deltas: tuple = (0.1, 0.2, 0.3, 0.42)
    hold_s: float = 1.0
    cruise_s: float = 0.0
    accel_estimate: float = 2.0
    reach_timeout: float = 4.0
    step_s: float = 0.8
    settle_s: float = 0.8
    stop_timeout: float = 5.0
    stopped_speed: float = 0.10

    def validate(self, test):
        if test not in ("grip", "straight", "steer", "policy-sweep"):
            raise ValueError("unknown test")
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        positive = ("rate_hz", "max_speed", "steer_limit", "max_duration", "stale_s", "cone_deg",
                    "a_brake", "margin", "min_clearance", "lat_acc_cap", "wheelbase", "erpm_gain",
                    "ramp", "accel_estimate", "reach_timeout", "step_s", "settle_s", "stop_timeout",
                    "stopped_speed")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("latency", "lidar_offset", "hold_s", "cruise_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0 <= self.short_reject < self.min_clearance:
            raise ValueError("short_reject must be in [0, min_clearance)")
        if not isinstance(self.cluster_beams, int) or not 1 <= self.cluster_beams <= 8:
            raise ValueError("cluster_beams must be an integer 1..8")
        if not 10 <= self.rate_hz <= 100 or self.stale_s > 0.25:
            raise ValueError("rate_hz must be 10..100; stale_s must be <= 0.25")
        if self.steer_limit > 0.42 or self.cone_deg > 90 or self.direction not in ("left", "right"):
            raise ValueError("invalid steering limit, cone, or direction")
        if test == "grip" and not (0 < self.delta <= self.steer_limit and
                                   0 < self.v_start <= self.v_cap <= self.max_speed):
            raise ValueError("grip requires 0 < v_start <= v_cap <= max_speed and valid delta")
        if test == "grip" and lateral(self.v_start, self.delta, self.wheelbase) > self.lat_acc_cap:
            raise ValueError("grip starting speed exceeds lateral acceleration cap")
        for name in (("targets",) if test == "straight" else ("speeds", "deltas") if test == "steer" else ()):
            values = getattr(self, name)
            cap = self.steer_limit if name == "deltas" else self.max_speed
            if not values or any(not math.isfinite(x) or not 0 < x <= cap for x in values):
                raise ValueError(f"{name} must be nonempty, finite, positive, and <= {cap}")
        if test == "steer" and any(not any(lateral(v, d, self.wheelbase) <= self.lat_acc_cap
                                           for d in self.deltas) for v in self.speeds):
            raise ValueError("each steering speed needs at least one allowed delta")
        return self


def lateral(v, delta, wheelbase=0.33):
    return v * v * abs(math.tan(delta)) / wheelbase


def stop_distance(v, p):
    """Required distance FROM THE LIDAR, not the rear axle. Margin includes nose overhang.

    Rear axle wall distance is lidar_distance + lidar_offset: the offset cancels
    when enforcing the same clearance at the sensor, so do NOT add it as free space.
    """
    v = abs(v)
    return v * p.latency + v * v / (2 * p.a_brake) + p.margin


def _cluster_min(values, k):
    """Smallest distance seen by k ADJACENT valid beams (max over each k-window, then min).

    The car's UST-10LX returns isolated spurious short readings: 0.07-0.29% of readings under
    10 cm in the 2026-11-17 recordings (f1tenth_gym_jax rl/car_params.py HR_SHORT_PROB), about
    3 per scan, which made a plain minimum trip the clearance stop in 20% of scans of
    rosbag2_2026_11_17-00_51_17. A real obstacle spans several 0.25 deg beams. Invalid beams are
    None and break a window. None when no full window of valid beams exists (fails closed).
    """
    best = None
    for i in range(len(values) - k + 1):
        window = values[i:i + k]
        if any(v is None for v in window):
            continue
        worst = max(window)
        best = worst if best is None else min(best, worst)
    return best


def scan_distances(ranges, angle_min, angle_increment, range_min, range_max, p):
    if not ranges or not all(math.isfinite(x) for x in (angle_min, angle_increment, range_min, range_max)) or angle_increment <= 0:
        return None, None
    # Readings below max(range_min, short_reject) are sensor noise, not obstacles; NaN/Inf are not
    # usable free-space evidence either. Absence of valid returns fails closed (None).
    lo = max(range_min, p.short_reject)
    valid = [r if math.isfinite(r) and lo <= r <= range_max else None for r in ranges]
    cone = math.radians(p.cone_deg)
    forward = [v for i, v in enumerate(valid) if abs(angle_min + i * angle_increment) <= cone]
    return _cluster_min(forward, p.cluster_beams), _cluster_min(valid, p.cluster_beams)


def steer_schedule(v, p):
    steps, skipped = [("settle", p.settle_s, 0.0)], []
    for d in p.deltas:
        if lateral(v, d, p.wheelbase) > p.lat_acc_cap:
            skipped.append(d)
            continue
        steps.extend([("left", p.step_s, d), ("center", p.settle_s, 0.0),
                      ("right", p.step_s, -d), ("center", p.settle_s, 0.0)])
    return steps, skipped


class Engine:
    def __init__(self, test, params=Params()):
        self.test, self.p = test, params.validate(test)
        self.phase, self.reason = "waiting", ""
        self.events = []
        self.rb = False
        self.released = False  # require observing release after startup
        self.scan_t = self.sensor_t = self.joy_t = self.odom_t = None
        self.forward = self.near = None
        self.speed = self.odom_speed = 0.0
        self.fault = 0
        self.run = -1
        self.started = self.phase_t = self.session_start = None
        self.stopped_since = None
        self.request = (0.0, 0.0)
        self.request_t = None
        self.last_output = (0.0, 0.0)
        self.schedule = []
        self.step_index = 0
        self.target = 0.0
        self._event("waiting", 0.0)

    @property
    def terminal(self):
        return self.phase in ("aborted", "done", "exit")

    def _event(self, phase, now, **extra):
        self.phase, self.phase_t = phase, now
        self.events.append(dict(test=self.test, phase=phase, time=now, run=self.run,
                                targets=dict(speed=self.target, steering=self.last_output[1]),
                                stop_reason=self.reason, params=asdict(self.p), **extra))

    def abort(self, reason, now):
        if not self.terminal:
            self.reason = reason
            self.last_output = (0.0, 0.0)
            self._event("aborted", now)
        return (0.0, 0.0)

    def joy(self, buttons, now):
        was = self.rb
        self.rb = len(buttons) > 5 and buttons[5] == 1
        self.joy_t = now
        if len(buttons) > 4 and buttons[4] and (self.rb or self.phase not in ("waiting", "between")):
            self.abort("LB override", now)
        if not self.rb:
            self.released = True
            if self.phase not in ("waiting", "between") and not self.terminal:
                self.abort("RB released", now)
        elif not was and self.released and self.phase in ("waiting", "between"):
            self.released = False
            self._start(now)

    def scan(self, ranges, angle_min, angle_increment, range_min, range_max, now):
        self.forward, self.near = scan_distances(ranges, angle_min, angle_increment, range_min, range_max, self.p)
        self.scan_t = now

    def sensor(self, erpm, fault, now):
        # Conservatively cover both offset signs; the checked-in odom uses +offset.
        self.speed = max(abs((erpm - self.p.erpm_offset) / self.p.erpm_gain),
                         abs((erpm + self.p.erpm_offset) / self.p.erpm_gain))
        self.fault, self.sensor_t = fault, now

    def odom(self, speed, now):
        self.odom_speed, self.odom_t = abs(speed), now

    def policy(self, speed, steer, now):
        self.request, self.request_t = (speed, steer), now

    def _health(self, now, v, steer=0.0):
        for name, stamp in (("scan", self.scan_t), ("sensors/core", self.sensor_t),
                            ("joy", self.joy_t), ("odom", self.odom_t)):
            if stamp is None or not 0 <= now - stamp <= self.p.stale_s:
                return f"stale /{name}"
        if not self.rb:
            return "RB released"
        if self.fault:
            return "VESC fault"
        if not all(math.isfinite(x) for x in (self.speed, self.odom_speed, v, steer)):
            return "nonfinite speed/steering"
        effective_v = max(abs(v), self.speed, self.odom_speed)
        # policy-sweep drives a walled track: the policy passes ducts closer than min_clearance and
        # has walls inside the stopping distance in every corner (2026-11-17 KV260 policy bags,
        # moving scans: clearance tripped in 14%, forward distance in up to 75%), and its crashes
        # are what the test records. It keeps RB, caps, staleness and the lateral limit only.
        if self.test != "policy-sweep":
            if self.forward is None or self.near is None:
                return "invalid forward scan"
            if self.near < self.p.min_clearance:
                return "minimum clearance"
            if self.forward <= stop_distance(effective_v, self.p):
                return "forward stopping distance"
        if max(self.speed, self.odom_speed) > self.p.max_speed + 0.10:
            return "measured overspeed"
        if lateral(effective_v, steer, self.p.wheelbase) > self.p.lat_acc_cap + 1e-8:
            return "lateral acceleration cap"
        return ""

    def _start(self, now):
        reason = self._health(now, 0.0)
        if reason:
            self.abort(reason, now)
            return
        if max(self.speed, self.odom_speed) > self.p.stopped_speed:
            self.abort("start requires stopped car", now)
            return
        self.run += 1
        self.reason = ""
        self.started = now
        if self.session_start is None:
            self.session_start = now
        self.stopped_since = None
        if self.test == "straight":
            self.target = self.p.targets[self.run]
            required = (self.target ** 2 / (2 * self.p.accel_estimate) +
                        self.target * (self.p.hold_s + self.p.cruise_s) + stop_distance(self.target, self.p))
            if self.forward <= required:
                self.abort(f"unreachable target: need {required:.2f} m from LiDAR, have {self.forward:.2f} m", now)
                return
            self._event("accelerate", now, start_wall_lidar_m=self.forward,
                        start_wall_rear_axle_m=self.forward + self.p.lidar_offset, required_m=required)
        elif self.test == "steer":
            self.target = self.p.speeds[self.run]
            self.schedule, skipped = steer_schedule(self.target, self.p)
            self.step_index = 0
            self._event("accelerate", now, skipped_deltas=skipped)
        elif self.test == "grip":
            self.target = self.p.v_start
            self._event("ramp", now)
        else:
            self._event("policy", now)

    def _brake(self, now, reason):
        self.reason = reason
        self.last_output = (0.0, 0.0)
        self._event("brake", now)

    def tick(self, now):
        if self.terminal or self.phase in ("waiting", "between"):
            self.last_output = (0.0, 0.0)
            return self.last_output
        if now - self.started >= self.p.max_duration:
            return self.abort("maximum duration", now)
        v, d = 0.0, 0.0
        if self.phase == "brake":
            if max(self.speed, self.odom_speed) <= self.p.stopped_speed:
                if self.stopped_since is None:
                    self.stopped_since = now
                if now - self.stopped_since >= 0.5:
                    count = len(self.p.targets) if self.test == "straight" else len(self.p.speeds) if self.test == "steer" else 1
                    self.released = False
                    self._event("done" if self.run + 1 == count else "between", now)
            else:
                self.stopped_since = None
            if self.phase == "brake" and now - self.phase_t > self.p.stop_timeout:
                return self.abort("failed to stop", now)
        elif self.test == "grip":
            d = self.p.delta * (1 if self.p.direction == "left" else -1)
            limit = min(self.p.v_cap, math.sqrt(self.p.lat_acc_cap * self.p.wheelbase / abs(math.tan(d))))
            v = min(limit, self.p.v_start + self.p.ramp * (now - self.started))
            self.target = v
            if v >= limit:
                self._brake(now, "lateral acceleration cap" if limit < self.p.v_cap else "speed cap")
                v = d = 0.0
        elif self.test == "policy-sweep":
            if self.request_t is None or now - self.request_t > self.p.stale_s:
                return self.abort("stale policy command", now)
            v, d = self.request
            if not math.isfinite(v) or not math.isfinite(d):
                return self.abort("nonfinite policy command", now)
            v = max(0.0, min(v, self.p.max_speed))
            d = max(-self.p.steer_limit, min(d, self.p.steer_limit))
            allowed = math.atan(self.p.lat_acc_cap * self.p.wheelbase / max(v, self.speed, self.odom_speed, 0.01) ** 2)
            d = max(-allowed, min(d, allowed))
            self.target = v
        else:
            v = self.target
            if self.phase == "accelerate":
                if self.odom_speed >= 0.9 * self.target:
                    self._event("hold" if self.test == "straight" else "settle", now)
                elif now - self.phase_t > self.p.reach_timeout:
                    return self.abort("target not reached", now)
            if self.test == "straight":
                if self.phase == "hold" and now - self.phase_t >= self.p.hold_s:
                    if self.p.cruise_s:
                        self._event("cruise", now)
                    else:
                        self._brake(now, "hold complete")
                elif self.phase == "cruise" and now - self.phase_t >= self.p.cruise_s:
                    self._brake(now, "cruise complete")
            elif self.phase != "accelerate":
                _, duration, d = self.schedule[self.step_index]
                if now - self.phase_t >= duration:
                    self.step_index += 1
                    if self.step_index == len(self.schedule):
                        self._brake(now, "steps complete")
                    else:
                        phase, _, d = self.schedule[self.step_index]
                        self.last_output = (v, d)
                        self._event(phase, now, step=self.step_index)
            if self.phase == "brake":
                v = d = 0.0
        reason = self._health(now, v, d)
        if reason:
            return self.abort(reason, now)
        if not (0 <= v <= self.p.max_speed and abs(d) <= self.p.steer_limit):
            return self.abort("profile exceeded hard cap", now)
        self.last_output = (v, d)
        return self.last_output

    def exit_commands(self, reason, now):
        """Latch zero and supply at least 0.5 s of zero publications to the wrapper."""
        self.reason = reason
        self.last_output = (0.0, 0.0)
        self._event("exit", now)
        return [(0.0, 0.0)] * (math.ceil(0.5 * self.p.rate_hz) + 1)
