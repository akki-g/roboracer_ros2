#!/usr/bin/env python3
"""Standalone ROS 2 Humble Follow-the-Gap controller.

Requires only NumPy and the ROS packages already used by the car: rclpy,
rcl_interfaces, sensor_msgs, and ackermann_msgs. No custom module, package
registration, or rebuild is needed when running this file with python3.

Dry run (after sourcing ROS and the native workspace):
    python3 gap_follow_standalone.py --ros-args -r /drive:=/gap_follow_test
Normal output through the existing Ackermann mux:
    python3 gap_follow_standalone.py --ros-args -p max_speed:=0.5
Quick offline checks, without importing ROS:
    python3 gap_follow_standalone.py --self-test

Default limits: speed 1.5 m/s, steering +/-0.35 rad, half-width including
margin 0.25 m, scan timeout 0.3 s. Start physical testing with max_speed:=0.5.
All settings are ROS startup parameters: restart with -p NAME:=VALUE to change
them. Vehicle dimensions and lidar_offset are measured from the rear axle.
These are initial settings, not a measurement or calibration of this car.

Angle zero must point forward and positive scan angles/steering point left.
Gap preference guides sampled Ackermann arcs, checked against LiDAR returns
with a swept rectangular footprint. Each scan selects one steering angle;
the arc's checked clearance bounds speed. Leading-body samples must be in
observed free space. The default 260-degree usable scan excludes the outer 5 degrees
at either rear edge of this car's 270-degree scan, where close peripheral
returns vetoed every path. Excluded sectors are unobserved, not free space.
The rear blind spot, tire slip, steering dynamics, and moving
obstacles are not predicted: this is still a reactive local planner.
Invalid/unknown ranges stay blocked; +inf is free only with inf_is_free:=true
after confirming the sensor's no-return convention. Speed uses curvature and
braking-distance limits, assuming the configured deceleration and latency.
Stop commands are never smoothed. A watchdog only works while this process
and control chain are running.
Release R1 before switching controllers; run one autonomous publisher at a
time. This file does not change the existing controller/mux handoff behavior.

Algorithm reference: https://github.com/f1tenth/f1tenth_lab4_template
ROS timer API: https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/node.py
"""

from dataclasses import dataclass, fields, replace
import math
from numbers import Integral, Real
import sys
import time

import numpy as np


@dataclass(frozen=True)
class GapConfig:
    max_speed: float = 1.5
    min_speed: float = 0.15       # Nominal cruise floor, never overrides safety limits.
    max_steering_angle: float = 0.35
    max_lidar_range: float = 10.0
    forward_fov_deg: float = 260.0  # Keep side walls; exclude the obstructed rear scan edges.
    bubble_radius: float = 0.25
    min_gap_distance: float = 0.60
    slow_distance: float = 0.10
    front_sector_deg: float = 20.0
    min_front_valid_fraction: float = 0.7
    preprocess_conv_size: int = 3
    disparity_threshold: float = 0.3
    target_window_deg: float = 6.0
    wheelbase: float = 0.33
    lidar_offset: float = 0.27   # Matches the physical car's bringup laser transform.
    vehicle_front: float = 0.43
    vehicle_rear: float = 0.12
    path_horizon: float = 3.0
    path_step: float = 0.10
    steering_samples: int = 41
    max_lateral_accel: float = 4.0
    braking_decel: float = 2.5
    reaction_time: float = 0.30
    max_acceleration: float = 1.0
    inf_is_free: bool = False
    comfort_clearance: float = 0.35
    inner_wall_bias: float = 1.0  # Optional inside-line preference; 0 disables it.
    inner_wall_clearance: float = 0.10  # Desired extra gap beyond the padded side.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == 'inf_is_free':
                if not isinstance(value, bool):
                    raise ValueError('inf_is_free must be a boolean')
                continue
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value)):
                raise ValueError(f'{field.name} must be a finite number')
        if not 0 <= self.min_speed <= self.max_speed:
            raise ValueError('Require 0 <= min_speed <= max_speed')
        if not 0 < self.max_steering_angle < math.pi / 2:
            raise ValueError('max_steering_angle must be between 0 and pi/2')
        if not 0 < self.front_sector_deg <= 90 <= self.forward_fov_deg <= 360:
            raise ValueError('Require 0 < front_sector_deg <= 90 <= forward_fov_deg <= 360')
        if not 0 < self.slow_distance <= self.max_lidar_range:
            raise ValueError('Require 0 < slow_distance <= max_lidar_range')
        if not 0 < self.bubble_radius < self.min_gap_distance <= self.max_lidar_range:
            raise ValueError('Require 0 < bubble_radius < min_gap_distance <= max_lidar_range')
        if not 0 < self.min_front_valid_fraction <= 1:
            raise ValueError('min_front_valid_fraction must be in (0, 1]')
        if not 0 < self.disparity_threshold <= self.max_lidar_range:
            raise ValueError('disparity_threshold must be positive and <= max_lidar_range')
        if not 0 < self.target_window_deg <= self.forward_fov_deg:
            raise ValueError('target_window_deg must be positive and <= forward_fov_deg')
        if (not isinstance(self.preprocess_conv_size, Integral)
                or not 1 <= self.preprocess_conv_size <= 101
                or self.preprocess_conv_size % 2 != 1):
            raise ValueError('preprocess_conv_size must be an odd integer from 1 to 101')
        for name in ('wheelbase', 'vehicle_front', 'vehicle_rear', 'max_lateral_accel',
                     'braking_decel', 'reaction_time', 'max_acceleration'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0 <= self.lidar_offset < self.vehicle_front:
            raise ValueError('Require 0 <= lidar_offset < vehicle_front')
        if not 0 < self.path_step < self.path_horizon <= self.max_lidar_range:
            raise ValueError('Require 0 < path_step < path_horizon <= max_lidar_range')
        if self.comfort_clearance <= 0:
            raise ValueError('comfort_clearance must be positive')
        if not 0 <= self.inner_wall_bias <= 1:
            raise ValueError('inner_wall_bias must be between 0 and 1')
        if not 0 <= self.inner_wall_clearance <= min(1.0, self.max_lidar_range):
            raise ValueError('inner_wall_clearance must be between 0 and 1 metre, within LiDAR range')
        if (not isinstance(self.steering_samples, Integral)
                or not 5 <= self.steering_samples <= 101 or self.steering_samples % 2 != 1):
            raise ValueError('steering_samples must be an odd integer from 5 to 101')


@dataclass(frozen=True)
class GapCommand:
    steering_angle: float
    speed: float
    reason: str
    target_angle: float = 0.0
    front_clearance: float = 0.0
    path_clearance: float = 0.0


@dataclass(frozen=True)
class ScanGeometry:
    indices: np.ndarray
    angles: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    front: np.ndarray
    jumps: np.ndarray
    step: float
    target_window: int


@dataclass(frozen=True)
class PathGeometry:
    travel: np.ndarray
    x: np.ndarray
    y: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    curvature: np.ndarray
    probe_range: np.ndarray
    probe_angle: np.ndarray
    probe_inside: np.ndarray
    scan_left: np.ndarray = None
    scan_right: np.ndarray = None
    scan_observed: np.ndarray = None

    def select(self, indices):
        return PathGeometry(self.travel, *(None if getattr(self, field.name) is None
                            else getattr(self, field.name)[indices]
                            for field in fields(self) if field.name != 'travel'))


class GapFollower:
    """Pure NumPy controller. Geometry is cached; measurements are always fresh."""

    def __init__(self, config=GapConfig()):
        if not isinstance(config, GapConfig):
            raise TypeError('config must be a GapConfig')
        self.config = config
        self._geometry_key = None
        self._geometry = None
        self._kernel = np.ones(config.preprocess_conv_size) / config.preprocess_conv_size
        self._previous_steering = 0.0
        self._previous_speed = 0.0
        self._build_paths()

    def stop(self, reason, clearance=0.0, path_clearance=0.0, preserve_steering=False):
        self._previous_speed = 0.0
        if not preserve_steering:
            self._previous_steering = 0.0
        return GapCommand(self._previous_steering, 0.0, reason,
                          front_clearance=float(clearance), path_clearance=float(path_clearance))

    def _build_paths(self):
        """Cache constant-steering arcs in the current vehicle frame."""
        cfg = self.config
        self._steering = np.linspace(-cfg.max_steering_angle, cfg.max_steering_angle,
                                     cfg.steering_samples)
        self._curvature = np.tan(self._steering) / cfg.wheelbase
        count = int(math.ceil(cfg.path_horizon / cfg.path_step))
        travel = np.linspace(0.0, cfg.path_horizon, count + 1)
        yaw = self._curvature[:, None] * travel
        x = travel * np.sinc(yaw / math.pi)
        y = travel * np.sin(yaw / 2) * np.sinc(yaw / (2 * math.pi))
        self._paths = self._make_paths(travel, x, y, yaw, self._curvature)

    def _make_paths(self, travel, x, y, yaw, curvature):
        cfg = self.config
        cosine, sine = np.cos(yaw), np.sin(yaw)
        lateral = np.linspace(-cfg.bubble_radius, cfg.bubble_radius,
                              max(3, int(math.ceil(2 * cfg.bubble_radius / 0.05)) + 1))
        bx, by = np.meshgrid(np.linspace(cfg.lidar_offset, cfg.vehicle_front, 3), lateral)
        px = x[..., None] + cosine[..., None] * bx.ravel() - sine[..., None] * by.ravel() - cfg.lidar_offset
        py = y[..., None] + sine[..., None] * bx.ravel() + cosine[..., None] * by.ravel()
        inside = ((px + cfg.lidar_offset >= -cfg.vehicle_rear)
                  & (px + cfg.lidar_offset <= cfg.vehicle_front)
                  & (np.abs(py) <= cfg.bubble_radius))
        return PathGeometry(travel, x, y, cosine, sine, curvature, np.hypot(px, py),
                            np.arctan2(py, px), inside)

    def _unknown_path_blocks(self, valid, geometry, paths):
        """Intersect unknown rays with the leading body, including between probes."""
        cfg = self.config
        blocked = np.zeros(paths.x.shape, dtype=bool)
        unknown = np.flatnonzero(~valid)
        if not unknown.size:
            return blocked
        cosine, sine = paths.cosine[..., None], paths.sine[..., None]
        dx = cfg.lidar_offset - paths.x[..., None]
        dy = -paths.y[..., None]
        origin_x = dx * cosine + dy * sine
        origin_y = -dx * sine + dy * cosine

        def slab(origin, direction, low, high):
            parallel = np.abs(direction) < 1e-12
            with np.errstate(divide='ignore', invalid='ignore'):
                first = (low - origin) / direction
                last = (high - origin) / direction
            entry = np.where(parallel, -np.inf, np.minimum(first, last))
            leave = np.where(parallel, np.inf, np.maximum(first, last))
            missed = parallel & ((origin < low) | (origin > high))
            return np.where(missed, np.inf, entry), np.where(missed, -np.inf, leave)

        for start in range(0, unknown.size, 64):
            indices = unknown[start:start + 64]
            ray_x, ray_y = geometry.cosine[indices], geometry.sine[indices]
            # Ignore only the part of each ray inside the current body. Beyond
            # its box exit, an invalid reading supplies no evidence of free space.
            with np.errstate(divide='ignore', invalid='ignore'):
                exit_x = np.where(ray_x >= 0, cfg.vehicle_front - cfg.lidar_offset,
                                  -cfg.vehicle_rear - cfg.lidar_offset) / ray_x
                exit_y = cfg.bubble_radius / np.abs(ray_y)
            exit_x = np.where(np.abs(ray_x) < 1e-12, np.inf, exit_x)
            exit_y = np.where(np.abs(ray_y) < 1e-12, np.inf, exit_y)
            known_exit = np.minimum(exit_x, exit_y)
            local_x = ray_x * cosine + ray_y * sine
            local_y = -ray_x * sine + ray_y * cosine
            enter_x, leave_x = slab(origin_x, local_x, cfg.lidar_offset, cfg.vehicle_front)
            enter_y, leave_y = slab(origin_y, local_y, -cfg.bubble_radius, cfg.bubble_radius)
            entry = np.maximum(np.maximum(enter_x, enter_y), known_exit)
            leave = np.minimum(leave_x, leave_y)
            # Strictly extend past the known body; numerical contact at s=0
            # must not make every candidate appear occupied.
            blocked |= np.any((entry <= leave) & (leave > known_exit + 1e-9), axis=2)
        return blocked

    def _obstacle_points(self, distances, hits, geometry):
        """Enclose every return in tight per-cell measured bounds.

        The grouping grid limits work, but the whole unmeasured 3 cm cell is
        not occupied. Its axis-aligned bounds enclose all measured returns,
        including singleton/thin obstacles; no return is averaged away.
        """
        cfg = self.config
        raw = np.column_stack((distances[hits] * geometry.cosine[hits] + cfg.lidar_offset,
                               distances[hits] * geometry.sine[hits]))
        cells, group = np.unique(np.floor(raw / 0.03), axis=0, return_inverse=True)
        low = np.full((len(cells), 2), np.inf)
        high = np.full((len(cells), 2), -np.inf)
        np.minimum.at(low, group, raw)
        np.maximum.at(high, group, raw)
        center = (low + high) / 2
        # Cover roundoff as well as both measured endpoints.
        half = np.nextafter(np.maximum(center - low, high - center), np.inf)
        return center[:, 0], center[:, 1], half.T


    @staticmethod
    def _obstacle_padding(cell_pad, cosine, sine):
        """Project measured AABB support onto body axes; retain scalar callers."""
        if np.ndim(cell_pad) == 0:
            return cell_pad, cell_pad
        half_x, half_y = cell_pad
        cosine, sine = np.abs(cosine), np.abs(sine)
        return cosine * half_x + sine * half_y, sine * half_x + cosine * half_y


    def _obstacle_clearance(self, local_x, local_y, cosine, sine, cell_pad):
        """Lower-bound body separation from all represented returns.

        An AABB contains every raw point in its group. Its projection radii
        give lower bounds on signed separation along both body axes. Taking
        the larger axis bound remains a lower bound on Euclidean separation;
        this preserves the interval-motion certificate used below.
        """
        cfg = self.config
        pad_x, pad_y = self._obstacle_padding(cell_pad, cosine, sine)
        long_gap = np.maximum(-cfg.vehicle_rear - local_x,
                              local_x - cfg.vehicle_front) - pad_x
        side_gap = np.abs(local_y) - cfg.bubble_radius - pad_y
        return np.min(np.maximum(long_gap, side_gap), axis=-1)

    @staticmethod
    def _cache_path_scan(paths, geometry):
        """Cache beam lookups only; scan measurements are always evaluated fresh."""
        if paths.scan_left is not None:
            return paths
        angles, probes = geometry.angles, paths.probe_angle
        right = np.searchsorted(angles, probes)
        left = np.clip(right - 1, 0, angles.size - 1)
        right = np.clip(right, 0, angles.size - 1)
        observed = ((probes >= angles[0]) & (probes <= angles[-1])
                    & (angles[right] - angles[left] <= 1.5 * geometry.step))
        return replace(paths, scan_left=left, scan_right=right, scan_observed=observed)

    def _refine_collision_intervals(self, paths, wall_clearance, px, py, cell_pad,
                                    counts, distances, valid, geometry):
        """Certify intervals and refine ambiguous near-term collision bounds.

        Each positive clearance is a lower bound on Euclidean distance from
        the entire padded body to an occupied cell. Body motion changes that
        separation by at most 1 + corner_radius * abs(curvature) per metre of
        rear-axle travel. Endpoint bounds c0,c1 cover their whole interval
        when c0+c1 > rate*ds. A positive initial clearance alone is insufficient.

        Up to five batched bisections resolve ambiguous first-arc intervals
        in the first 0.3 m. Unresolved intervals remain blocked. Inserted poses
        must also pass the existing observation and unknown-ray checks.
        """
        cfg = self.config
        if not px.size:
            return np.zeros(wall_clearance.shape, dtype=bool)
        out = wall_clearance <= 0.0
        corner = math.hypot(max(cfg.vehicle_front, cfg.vehicle_rear), cfg.bubble_radius)
        ds = np.diff(paths.travel)
        # Retain the generating curvature rather than recovering it from
        # wrapped headings; use actual sample spacing, not configured maxima.
        curvature = np.broadcast_to(paths.curvature[:, None], (len(paths.x), len(ds)))
        lipschitz = 1.0 + corner * np.abs(curvature)
        pair_clear = ((wall_clearance[:, :-1] > 0.0)
                      & (wall_clearance[:, 1:] > 0.0)
                      & (wall_clearance[:, :-1] + wall_clearance[:, 1:]
                         > lipschitz * ds[None, :] + 1e-9))
        out[:, 1:] = ~pair_clear
        eligible = (~pair_clear & (wall_clearance[:, :-1] > 0.0)
                    & (wall_clearance[:, 1:] > 0.0)
                    & (paths.travel[None, 1:] <= 0.3 + 1e-9)
                    & (np.arange(1, paths.travel.size)[None, :] < counts[:, None]))
        rows, starts = np.where(eligible)
        if not rows.size:
            return out
        # Every subdivision carries its original interval id. Any failed
        # child rejects that complete interval, including its other children.
        failed = np.zeros(rows.size, dtype=bool)
        root = np.arange(rows.size)
        x0, y0 = paths.x[rows, starts], paths.y[rows, starts]
        a0 = np.arctan2(paths.sine[rows, starts], paths.cosine[rows, starts])
        c0, c1 = wall_clearance[rows, starts], wall_clearance[rows, starts + 1]
        length, kappa = ds[starts], curvature[rows, starts]
        for depth in range(5):
            half = length / 2
            da = kappa * half
            local_x = half * np.sinc(da / math.pi)
            local_y = half * np.sin(da / 2) * np.sinc(da / (2 * math.pi))
            xm = x0 + np.cos(a0) * local_x - np.sin(a0) * local_y
            ym = y0 + np.sin(a0) * local_x + np.cos(a0) * local_y
            am = a0 + da
            cm, sm = np.cos(am), np.sin(am)
            dx, dy = px[None, :] - xm[:, None], py[None, :] - ym[:, None]
            lx, ly = dx * cm[:, None] + dy * sm[:, None], -dx * sm[:, None] + dy * cm[:, None]
            gap = self._obstacle_clearance(lx, ly, cm[:, None], sm[:, None], cell_pad)
            pose = self._make_paths(np.array([0.0]), xm[:, None], ym[:, None],
                                    am[:, None], kappa)
            pose = self._cache_path_scan(pose, geometry)
            left, right = pose.scan_left, pose.scan_right
            observed = pose.scan_observed & valid[left] & valid[right]
            visible = observed & (pose.probe_range <= np.minimum(distances[left], distances[right]))
            observed_pose = (np.all(visible | pose.probe_inside, axis=(1, 2))
                             & ~np.any(self._unknown_path_blocks(valid, geometry, pose), axis=1))
            failed[root[(gap <= 0.0) | ~observed_pose]] = True
            bound = (1.0 + corner * np.abs(kappa)) * half
            left_pending = (c0 + gap <= bound + 1e-9) & ~failed[root]
            right_pending = (gap + c1 <= bound + 1e-9) & ~failed[root]
            if depth == 4:
                failed[root[left_pending | right_pending]] = True
                break
            if not np.any(left_pending | right_pending):
                break
            selected = np.r_[np.flatnonzero(left_pending), np.flatnonzero(right_pending)]
            x0 = np.r_[x0[left_pending], xm[right_pending]]
            y0 = np.r_[y0[left_pending], ym[right_pending]]
            a0 = np.r_[a0[left_pending], am[right_pending]]
            c0 = np.r_[c0[left_pending], gap[right_pending]]
            c1 = np.r_[gap[left_pending], c1[right_pending]]
            length, kappa, root = half[selected], kappa[selected], root[selected]
        out[rows, starts + 1] = failed
        return out

    def _evaluate_paths(self, distances, valid, hits, geometry, paths=None, points=None):
        """Return checked travel and wall clearance within observed free space."""
        cfg = self.config
        if paths is None:
            self._paths = self._cache_path_scan(self._paths, geometry)
            paths = self._paths
        else:
            paths = self._cache_path_scan(paths, geometry)
        left, right = paths.scan_left, paths.scan_right
        observed = paths.scan_observed & valid[left] & valid[right]
        visible_range = np.minimum(distances[left], distances[right])
        visible = observed & (paths.probe_range <= visible_range)
        blocked = ~np.all(visible | paths.probe_inside, axis=2)
        unknown_blocked = self._unknown_path_blocks(valid, geometry, paths)
        blocked |= unknown_blocked
        visibility_end = np.where(np.any(blocked, axis=1),
                                  np.argmax(blocked, axis=1), paths.travel.size)
        # All returns, including side/rear returns, constrain the entire body.
        # No-return max-range beams describe free space, not physical walls.
        px, py, cell_pad = (self._obstacle_points(distances, hits, geometry)
                            if points is None else points)
        wall_clearance = np.full(paths.x.shape, cfg.max_lidar_range)
        if px.size:
            # Each grouped obstacle encloses all of its actual returns.
            # Project its tight bounds into each body pose; do not mark the
            # rest of the grouping cell occupied or discard singleton returns.
            # Check each arc only as far as its observed free space extends.
            reach = float(np.max(np.hypot(paths.x, paths.y))) + math.hypot(
                max(cfg.vehicle_front, cfg.vehicle_rear), cfg.bubble_radius) + 1.0
            radius = (cell_pad if np.ndim(cell_pad) == 0
                      else np.hypot(cell_pad[0], cell_pad[1]))
            nearby = np.hypot(px, py) <= reach + radius
            px, py = px[nearby], py[nearby]
            if np.ndim(cell_pad) != 0:
                cell_pad = cell_pad[:, nearby]
            for row, count in enumerate(visibility_end):
                if not px.size:
                    break
                if count == 0:
                    continue
                dx = px - paths.x[row, :count, None]
                dy = py - paths.y[row, :count, None]
                cosine, sine = paths.cosine[row, :count, None], paths.sine[row, :count, None]
                local_x = dx * cosine + dy * sine
                local_y = -dx * sine + dy * cosine
                wall_clearance[row, :count] = self._obstacle_clearance(
                    local_x, local_y, cosine, sine, cell_pad)
            blocked |= self._refine_collision_intervals(
                paths, wall_clearance, px, py, cell_pad, visibility_end,
                distances, valid, geometry)
        first_blocked = np.where(np.any(blocked, axis=1), np.argmax(blocked, axis=1),
                                 paths.travel.size)
        last_free = np.maximum(0, first_blocked - 1)
        travel = paths.travel[last_free]
        # Measure clearance at a shared near-term distance. Including the same
        # initial pose in a minimum hides the benefit of steering away from a wall.
        near_end = np.minimum(travel, 1.0)
        near_mask = ((paths.travel[None, :] >= np.minimum(0.25, near_end[:, None] / 2))
                     & (paths.travel[None, :] <= near_end[:, None]))
        comfort = np.sum(np.where(near_mask, np.clip(wall_clearance, 0, cfg.comfort_clearance), 0), axis=1)
        comfort /= np.maximum(1, np.sum(near_mask, axis=1))
        return travel, comfort

    def _scan_geometry(self, count, angle_min, angle_increment):
        key = (count, angle_min, angle_increment)
        if key == self._geometry_key:
            return self._geometry
        cfg = self.config
        step = abs(angle_increment)
        angles = (angle_min + np.arange(count) * angle_increment + math.pi) % (
            2 * math.pi) - math.pi
        indices = np.flatnonzero(
            np.abs(angles) <= math.radians(cfg.forward_fov_deg) / 2 + 1e-9)
        indices = indices[np.argsort(angles[indices], kind='stable')]
        angles = angles[indices]
        if angles.size:
            # A full-circle scan can contain the same endpoint twice.
            keep = np.r_[True, np.diff(angles) > step / 2]
            indices, angles = indices[keep], angles[keep]
        window = max(3, int(math.ceil(math.radians(cfg.target_window_deg) / step)))
        if window % 2 == 0:
            window += 1
        geometry = ScanGeometry(
            indices, angles, np.cos(angles), np.sin(angles),
            np.abs(angles) <= math.radians(cfg.front_sector_deg) / 2 + 1e-9,
            np.diff(angles) > 1.5 * step, step, window,
        )
        self._geometry_key, self._geometry = key, geometry
        # The scan's beam count, ordering or angle increment changed. Cached
        # indices from the previous layout must not address the new scan.
        self._paths = replace(self._paths, scan_left=None, scan_right=None,
                              scan_observed=None)
        return geometry

    def _prefer_inner_wall(self, scores, travel, points):
        """Favor a supported inside wall among comparable checked steering arcs."""
        cfg = self.config
        feasible = np.isfinite(scores) & (travel > 1e-9)
        if cfg.inner_wall_bias == 0 or not np.any(feasible):
            return scores
        baseline = int(np.argmax(np.where(feasible, scores, -np.inf)))
        if abs(self._steering[baseline]) < 0.10:
            return scores
        direction = float(np.sign(self._steering[baseline]))
        px, py, cell_pad = points
        if px.size < 3:
            return scores

        def wall_error(root):
            gaps = []
            bank = self._paths
            samples = np.array([0.3, 0.6, 0.9])
            samples = samples[samples <= travel[root] + 1e-9]
            indices = np.unique(np.searchsorted(bank.travel, samples + 1e-9, side='right') - 1)
            for index in indices[indices > 0]:
                cosine, sine = bank.cosine[root, index], bank.sine[root, index]
                dx, dy = px - bank.x[root, index], py - bank.y[root, index]
                local_x = dx * cosine + dy * sine
                local_y = direction * (-dx * sine + dy * cosine)
                side = ((local_x >= -cfg.vehicle_rear)
                        & (local_x <= cfg.vehicle_front)
                        & (local_y >= cfg.bubble_radius)
                        & (local_y <= cfg.bubble_radius + 1.0))
                # An isolated return or a distant outer wall is not an inside wall.
                if np.count_nonzero(side) < 3 or np.ptp(local_x[side]) < 0.15:
                    continue
                _, side_pad = self._obstacle_padding(cell_pad, cosine, sine)
                gaps.append(float(np.min((local_y - cfg.bubble_radius - side_pad)[side])))
            if len(gaps) < 2:
                return None
            gaps = np.asarray(gaps)
            error = np.where(gaps < cfg.inner_wall_clearance,
                             2 * (cfg.inner_wall_clearance - gaps),
                             gaps - cfg.inner_wall_clearance)
            return float(np.mean(error))

        baseline_error = wall_error(baseline)
        if baseline_error is None:
            return scores
        # Preserve braking room for the original arc's steady speed ceiling;
        # a recent stop must not let the preference accept a much shorter path.
        at = cfg.braking_decel * cfg.reaction_time
        reference_speed = min(
            cfg.max_speed,
            math.sqrt(at * at + 2 * cfg.braking_decel * travel[baseline]) - at,
            math.sqrt(cfg.max_lateral_accel / max(abs(self._curvature[baseline]), 1e-6)))
        required_travel = (reference_speed * cfg.reaction_time
                           + reference_speed ** 2 / (2 * cfg.braking_decel))
        comparable = (feasible & (direction * self._steering >= 0.10)
                      & (travel >= travel[baseline] - 0.10 - 1e-9)
                      & (travel >= required_travel - 1e-9))
        biased = scores.copy()
        for root in np.flatnonzero(comparable):
            error = baseline_error if root == baseline else wall_error(int(root))
            if error is not None:
                improvement = np.clip((baseline_error - error) / 0.10, 0.0, 1.0)
                biased[root] += 0.4 * cfg.inner_wall_bias * improvement
        return biased

    def _inflate_edges(self, processed, distances, valid, geometry):
        """Extend a nearby obstacle edge into its neighboring farther readings."""
        differences = np.diff(distances)
        edges = np.flatnonzero(
            valid[:-1] & valid[1:] & ~geometry.jumps
            & (np.abs(differences) >= self.config.disparity_threshold))
        for edge in edges:
            near = edge if differences[edge] > 0 else edge + 1
            distance = distances[near]
            half_angle = math.asin(min(1.0, self.config.bubble_radius / distance))
            if near == edge:
                start = near + 1
                end = np.searchsorted(
                    geometry.angles, geometry.angles[near] + half_angle, side='right')
            else:
                start = np.searchsorted(
                    geometry.angles, geometry.angles[near] - half_angle, side='left')
                end = near
            processed[start:end] = np.minimum(processed[start:end], distance)

    def _choose_target(self, processed, free, geometry):
        angles, step = geometry.angles, geometry.step
        starts = np.flatnonzero(
            free & (~np.r_[False, free[:-1]] | np.r_[False, geometry.jumps]))
        ends = np.flatnonzero(
            free & (~np.r_[free[1:], False] | np.r_[geometry.jumps, False]))
        candidates = []
        for start, end in zip(starts, ends):
            count = int(end - start + 1)
            if count < 3:
                continue
            segment = processed[start:end + 1]
            span = float(angles[end] - angles[start])
            near = float(np.min(segment))
            if 2 * near * math.sin(min(math.pi, span) / 2) < 2 * self.config.bubble_radius:
                continue
            window = min(geometry.target_window, count if count % 2 else count - 1)
            centers = np.arange(start + window // 2, end - window // 2 + 1)
            # Use the worst clearance in each complete window: one unusually
            # long ray cannot make an otherwise short opening look attractive.
            scores = np.min(np.lib.stride_tricks.sliding_window_view(segment, window), axis=1)
            margin = math.asin(min(1.0, self.config.bubble_radius / near))
            interior = ((angles[centers] >= angles[start] + margin)
                        & (angles[centers] <= angles[end] - margin))
            if not np.any(interior):
                continue
            centers, scores = centers[interior], scores[interior]
            # Cap depth: a distant ray at a corner is not a better racing line
            # than a broad opening with adequate lookahead and wall clearance.
            centrality = np.minimum(angles[centers] - angles[start],
                                    angles[end] - angles[centers]) / max(span, step)
            scores = (np.minimum(scores / self.config.path_horizon, 1.0)
                      + 0.35 * centrality - 0.1 * np.abs(angles[centers]))
            best = int(np.argmax(scores))
            candidates.append((float(scores[best]), int(centers[best])))
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c[0], -abs(angles[c[1]])))[1]

    def compute(self, ranges, angle_min, angle_increment, range_min, range_max, dt=0.025):
        cfg = self.config
        try:
            raw = np.asarray(ranges, dtype=float)
            metadata = np.asarray(
                [angle_min, angle_increment, range_min, range_max], dtype=float)
        except (TypeError, ValueError, OverflowError):
            return self.stop('invalid_scan')
        if raw.ndim != 1 or raw.size < 3:
            return self.stop('invalid_scan')
        if not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
            return self.stop('invalid_control_interval')
        if metadata.shape != (4,) or not np.all(np.isfinite(metadata)):
            return self.stop('invalid_scan_metadata')
        angle_min, angle_increment, range_min, range_max = metadata
        step = abs(angle_increment)
        if (step == 0 or step > math.pi or range_min < 0 or range_max <= range_min
                or (raw.size - 1) * step > 2 * math.pi + step):
            return self.stop('invalid_scan_metadata')
        geometry = self._scan_geometry(raw.size, angle_min, angle_increment)
        raw = raw[geometry.indices]
        if raw.size < 3 or np.count_nonzero(geometry.front) < 3:
            return self.stop('no_forward_data')
        finite = (np.isfinite(raw) & (raw > 0) & (raw >= range_min) & (raw <= range_max))
        no_return = np.isposinf(raw) & cfg.inf_is_free
        valid = finite | no_return
        if np.mean(valid[geometry.front]) < cfg.min_front_valid_fraction:
            return self.stop('insufficient_front_data')
        distances = np.where(valid, np.minimum(raw, min(range_max, cfg.max_lidar_range)), 0.0)
        hits = finite & (raw < min(range_max, cfg.max_lidar_range))
        x, y = distances * geometry.cosine, distances * geometry.sine
        corridor = valid & (x > 0) & (np.abs(y) <= cfg.bubble_radius)
        observed_front = valid & geometry.front
        clearance = float(np.min(x[corridor | observed_front]))
        processed = distances.copy()
        if cfg.preprocess_conv_size > 1:
            average = np.convolve(
                np.pad(distances, cfg.preprocess_conv_size // 2), self._kernel, mode='valid')
            processed = np.minimum(processed, average)
        self._inflate_edges(processed, distances, valid, geometry)
        # Full-body checks handle nearby walls. Erasing the angular bubble of
        # one globally closest ray biases symmetric openings as their width changes.
        free = (valid & (processed >= cfg.min_gap_distance)
                & (np.abs(geometry.angles) <= math.pi / 2))
        target_index = self._choose_target(processed, free, geometry)
        target = float(geometry.angles[target_index]) if target_index is not None else 0.0
        # Convert a gap bearing to a feasible steering preference using the
        # bicycle/pure-pursuit geometry. It is a preference, never a safety gate.
        lookahead = 1.0 if target_index is None else float(np.clip(processed[target_index], 1.0, 2.5))
        tx = lookahead * math.cos(target) + cfg.lidar_offset
        ty = lookahead * math.sin(target)
        preferred = math.atan2(2 * cfg.wheelbase * ty, tx * tx + ty * ty)
        points = self._obstacle_points(distances, hits, geometry)
        travel, comfort = self._evaluate_paths(distances, valid, hits, geometry, points=points)
        feasible = travel > 1e-9
        if not np.any(feasible):
            return self.stop('no_safe_path', clearance, float(np.max(travel)), preserve_steering=True)
        # Each arc is a complete candidate, so reward its full checked travel.
        score = (2.0 * travel / cfg.path_horizon + 1.2 * comfort
                 - 0.35 * np.abs(self._steering - preferred) / cfg.max_steering_angle
                 - 0.15 * np.abs(self._steering - self._previous_steering) / cfg.max_steering_angle
                 - 0.05 * np.abs(self._steering) / cfg.max_steering_angle)
        score = self._prefer_inner_wall(score, travel, points)
        best = int(np.argmax(np.where(feasible, score, -np.inf)))
        steering, path_clearance = float(self._steering[best]), float(travel[best])
        # Solve v*latency + v^2/(2*deceleration) <= available path distance.
        at = cfg.braking_decel * cfg.reaction_time
        brake_speed = math.sqrt(at * at + 2 * cfg.braking_decel * path_clearance) - at
        lateral_speed = math.sqrt(cfg.max_lateral_accel / max(abs(self._curvature[best]), 1e-6))
        cruise_speed = max(cfg.min_speed, cfg.max_speed * min(1.0, path_clearance / cfg.slow_distance))
        speed = min(cfg.max_speed, cruise_speed, brake_speed, lateral_speed,
                    self._previous_speed + cfg.max_acceleration * min(dt, 0.1))
        self._previous_speed, self._previous_steering = float(speed), steering
        return GapCommand(steering, float(speed), 'ok', target, clearance, path_clearance)


def scan_timestamp_status(stamp_ns, now_ns, timeout):
    """Zero stamps use the arrival watchdog; nonzero stamps must share ROS time."""
    if stamp_ns == 0:
        return None
    if stamp_ns < 0 or now_ns <= 0:
        return 'invalid_scan_clock'
    age = (now_ns - stamp_ns) / 1e9
    if age > timeout:
        return 'stale_scan'
    if age < -timeout:
        return 'scan_clock_ahead'
    return None


def build_ros_node_class():
    """Import ROS lazily so the core and --self-test work without ROS installed."""
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from rcl_interfaces.msg import ParameterDescriptor
    from ackermann_msgs.msg import AckermannDriveStamped
    from sensor_msgs.msg import LaserScan

    class GapFollowNode(Node):
        def __init__(self):
            super().__init__('gap_follow_standalone')
            descriptor = ParameterDescriptor(
                read_only=True, description='Set at startup; restart to change.')
            defaults = GapConfig()
            settings = {
                field.name: self.declare_parameter(
                    field.name, getattr(defaults, field.name), descriptor).value
                for field in fields(defaults)
            }
            self.controller = GapFollower(GapConfig(**settings))
            self.scan_timeout = self.declare_parameter('scan_timeout', 0.3, descriptor).value
            if not math.isfinite(self.scan_timeout) or self.scan_timeout <= 0:
                raise ValueError('scan_timeout must be positive and finite')
            self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 1)
            scan_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE)
            self.subscription = self.create_subscription(LaserScan, '/scan', self.on_scan, scan_qos)
            self.last_received = None
            self.last_reason = None
            self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self.watchdog = self.create_timer(
                min(0.05, self.scan_timeout / 2), self.check_timeout, clock=self.steady_clock)
            self.get_logger().info(
                f'Gap following: max_speed={settings["max_speed"]:.2f} m/s, '
                f'steering_limit={settings["max_steering_angle"]:.2f} rad. '
                'Use /drive:=/gap_follow_test for a dry run.')

        def publish_command(self, command):
            message = AckermannDriveStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = 'base_link'
            message.drive.steering_angle = float(command.steering_angle)
            message.drive.speed = float(command.speed)
            self.publisher.publish(message)
            if command.reason != self.last_reason:
                self.last_reason = command.reason
                self.get_logger().info(
                    f'Gap following: {command.reason}; front={command.front_clearance:.2f} m, '
                    f'path={command.path_clearance:.2f} m, speed={command.speed:.2f} m/s')
            self.get_logger().debug(
                f'front={command.front_clearance:.2f} path={command.path_clearance:.2f} '
                f'target={command.target_angle:.3f} steer={command.steering_angle:.3f} '
                f'speed={command.speed:.2f}', throttle_duration_sec=1.0)

        def on_scan(self, scan):
            try:
                stamp_ns = scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
                status = scan_timestamp_status(
                    stamp_ns, self.get_clock().now().nanoseconds, self.scan_timeout)
                if status is not None:
                    self.publish_command(self.controller.stop(status))
                    return
                received = time.monotonic()
                dt = 0.025 if self.last_received is None else max(1e-4, received - self.last_received)
                self.last_received = received
                command = self.controller.compute(
                    scan.ranges, scan.angle_min, scan.angle_increment,
                    scan.range_min, scan.range_max, dt=dt)
                # A long callback can outlive the scan's freshness budget; the
                # timer cannot interrupt it. Recheck before publishing motion.
                status = scan_timestamp_status(
                    stamp_ns, self.get_clock().now().nanoseconds, self.scan_timeout)
                if status is not None:
                    command = self.controller.stop(status)
                elif time.monotonic() - received > self.scan_timeout:
                    command = self.controller.stop('scan_timeout')
            except Exception as error:
                # A callback failure must not leave the previous moving command
                # as our most recent output. KeyboardInterrupt is not swallowed.
                command = self.controller.stop('scan_processing_error')
                self.get_logger().error(f'{type(error).__name__}: {error}', throttle_duration_sec=1.0)
            self.publish_command(command)

        def check_timeout(self):
            if self.last_received is None:
                self.publish_command(self.controller.stop('waiting_for_scan'))
            elif time.monotonic() - self.last_received > self.scan_timeout:
                self.publish_command(self.controller.stop('scan_timeout'))

    return GapFollowNode


def self_test():
    """Small portable smoke check; fuller synthetic tests were run during development."""
    angles = np.linspace(-3 * math.pi / 4, 3 * math.pi / 4, 1081)
    follower = GapFollower()

    def command(ranges):
        return follower.compute(ranges, angles[0], angles[1] - angles[0], 0.05, 30.0)

    corridor = np.minimum(8.0, 1.2 / np.maximum(np.abs(np.sin(angles)), 1e-6))
    result = command(corridor)
    assert 0 < result.speed <= follower.config.max_speed and abs(result.steering_angle) < 0.1
    for value in (0.0, np.nan, np.inf, -np.inf):
        assert command(np.full(angles.size, value)).speed == 0
    blocked = np.full(angles.size, 4.0)
    # Occupy the current body so there is no checked travel toward the wall.
    blocked[np.abs(angles) < math.radians(10)] = 0.10
    assert command(blocked).speed == 0
    assert scan_timestamp_status(1_000_000_000, 2_000_000_000, 0.3) == 'stale_scan'
    print('Self-test passed: corridor, invalid data, front stop, timestamp check.')


def main():
    if sys.argv[1:] == ['--self-test']:
        self_test()
        return
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.signals import SignalHandlerOptions
        node_class = build_ros_node_class()
    except ImportError as error:
        raise SystemExit(
            'ROS imports failed. Source /opt/ros/humble/setup.bash and your '
            'workspace install/setup.bash, then run this file with system python3. '
            f'Details: {error}') from error
    import signal

    def request_stop(_signum, _frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    node = None
    try:
        # Keep the ROS context alive while handling Ctrl+C, so the finally
        # block can make a best-effort zero-speed publication before shutdown.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = node_class()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                if rclpy.ok():
                    node.publish_command(node.controller.stop('shutdown'))
            finally:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == '__main__':
    main()
