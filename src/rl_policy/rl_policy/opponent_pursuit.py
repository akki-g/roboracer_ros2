"""Opponent car for the ROS sim, without ROS: pure pursuit along a map's centerline at a set
speed, or one that changes every few seconds within a range, slowing for corners.

Used by opponent_node.py (the ROS node). Centerline: the map's *_centerline.csv
(x, y, w_left, w_right; the driving direction is the file's order, the same as the sim's
start heading in the sim_rl_<MAP>.yaml configs).
"""

import math

import numpy as np

WHEELBASE = 0.3302  # m, the classic gym's car (lf + lr)
STEER_LIMIT = 0.4189  # rad, the classic gym's car


def parse_speed(text):
    """'2.0' -> (2.0, 2.0); '1.0-4.0' -> (1.0, 4.0) (m/s)."""
    parts = [float(p) for p in str(text).replace(":", "-").split("-") if p.strip()]
    if len(parts) == 1:
        parts *= 2
    lo, hi = sorted(parts[:2])
    if lo < 0 or len(parts) != 2:
        raise ValueError(f"speed must be like 2.0 or 1.0-4.0, got {text!r}")
    return lo, hi


class Track:
    """A closed centerline: arc length, heading and curvature at each point."""

    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=float)[:, :2]
        seg = np.hypot(*np.diff(np.vstack([self.xy, self.xy[:1]]), axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        self.length = float(seg.sum())
        self.spacing = self.length / len(self.xy)
        t = np.roll(self.xy, -1, axis=0) - np.roll(self.xy, 1, axis=0)
        self.yaw = np.arctan2(t[:, 1], t[:, 0])
        k = 3  # curvature over ~0.6 m at 0.1 m spacing
        dyaw = np.angle(np.exp(1j * (np.roll(self.yaw, -k) - np.roll(self.yaw, k))))
        self.kappa = np.abs(dyaw) / (2 * k * self.spacing)

    @classmethod
    def from_csv(cls, path):
        return cls(np.loadtxt(path, delimiter=",", comments="#"))

    def nearest(self, x, y):
        return int(np.argmin(np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)))

    def ahead(self, i, dist):
        """Index of the centerline point ``dist`` m ahead of point i."""
        return int((i + round(dist / self.spacing)) % len(self.xy))

    def pose_ahead(self, dist):
        """(x, y, yaw) on the centerline ``dist`` m after its first point."""
        j = self.ahead(0, dist)
        return float(self.xy[j, 0]), float(self.xy[j, 1]), float(self.yaw[j])


def lookahead(v):
    return min(max(0.4 + 0.3 * v, 0.6), 1.5)


def pursuit_steer(track, x, y, yaw, v, wheelbase=WHEELBASE, steer_limit=STEER_LIMIT):
    """Pure-pursuit steering toward the centerline point a speed-dependent lookahead ahead."""
    ld = lookahead(v)
    i = track.nearest(x, y)
    tx, ty = track.xy[track.ahead(i, ld)]
    dx, dy = tx - x, ty - y
    lx = math.cos(yaw) * dx + math.sin(yaw) * dy
    ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
    alpha = math.atan2(ly, lx)
    dist = max(math.hypot(lx, ly), 1e-3)
    steer = math.atan2(2.0 * wheelbase * math.sin(alpha), dist)
    return float(np.clip(steer, -steer_limit, steer_limit)), i


def corner_speed(track, i, a_lat, horizon=2.0):
    """Highest speed that keeps lateral acceleration under a_lat over the next ``horizon`` m."""
    n = max(1, round(horizon / track.spacing))
    idx = (i + np.arange(n)) % len(track.xy)
    k = float(track.kappa[idx].max())
    return math.sqrt(a_lat / k) if k > 1e-6 else float("inf")


class SpeedSchedule:
    """Target speed: fixed (lo == hi), or a new uniform draw in [lo, hi] every ``period`` s."""

    def __init__(self, lo, hi, period=6.0, seed=0):
        self.lo, self.hi, self.period = lo, hi, period
        self.rng = np.random.default_rng(seed)
        self.target = lo if lo == hi else float(self.rng.uniform(lo, hi))
        self.next_change = period

    def __call__(self, t):
        if self.lo != self.hi and t >= self.next_change:
            self.target = float(self.rng.uniform(self.lo, self.hi))
            self.next_change = t + self.period
        return self.target


def command(track, schedule, t, x, y, yaw, v, a_lat=4.0):
    """(speed, steer, target speed) for the opponent at time t (s) and pose (x, y, yaw), speed v."""
    steer, i = pursuit_steer(track, x, y, yaw, v)
    target = schedule(t)
    return min(target, corner_speed(track, i, a_lat)), steer, target
