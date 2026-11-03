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
Gap preference guides sampled Ackermann paths, checked against LiDAR returns
with a swept rectangular footprint. Selected arcs branch into turn/straighten
continuations; only the first arc's clearance authorizes speed. Leading-body samples must be in observed
free space. The default 260-degree usable scan excludes the outer 5 degrees
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

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
import sys
import time

import numpy as np


@dataclass(frozen=True)
class GapConfig:
    max_speed: float = 1.5
    min_speed: float = 0.15       # Nominal crawl speed, never overrides safety limits.
    max_steering_angle: float = 0.35
    max_lidar_range: float = 10.0
    forward_fov_deg: float = 260.0  # Keep side walls; exclude the obstructed rear scan edges.
    bubble_radius: float = 0.25
    min_gap_distance: float = 0.60
    stop_distance: float = 0.15  # Reserve along the checked path, NOT front-ray range.
    slow_distance: float = 1.5
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
    min_path_distance: float = 0.30
    steering_samples: int = 41
    max_lateral_accel: float = 1.5
    braking_decel: float = 2.5
    reaction_time: float = 0.30
    max_acceleration: float = 1.0
    inf_is_free: bool = False
    planning_horizon: float = 5.0
    branch_distance: float = 0.9
    comfort_clearance: float = 0.35
    crawl_speed: float = 0.15

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
        if not 0 < self.stop_distance < self.slow_distance <= self.max_lidar_range:
            raise ValueError('Require 0 < stop_distance < slow_distance <= max_lidar_range')
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
        if not 0 < self.path_step <= self.min_path_distance < self.path_horizon <= self.max_lidar_range:
            raise ValueError('Require 0 < path_step <= min_path_distance < path_horizon <= max_lidar_range')
        if not self.path_horizon <= self.planning_horizon <= self.max_lidar_range:
            raise ValueError('Require path_horizon <= planning_horizon <= max_lidar_range')
        if not self.min_path_distance <= self.branch_distance < self.path_horizon:
            raise ValueError('Require min_path_distance <= branch_distance < path_horizon')
        if self.comfort_clearance <= 0 or self.crawl_speed <= 0:
            raise ValueError('comfort_clearance and crawl_speed must be positive')
        if not self.stop_distance < self.min_path_distance:
            raise ValueError('stop_distance must be smaller than min_path_distance')
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
    planning_clearance: float = 0.0


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
    sweep_pad: np.ndarray
    probe_range: np.ndarray
    probe_angle: np.ndarray
    probe_inside: np.ndarray

    def select(self, indices):
        return PathGeometry(self.travel, *(getattr(self, field.name)[indices]
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
        """Cache first arcs and turn/straighten continuations in the current frame."""
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
        # Keep looking for a change of steering as room contracts. A fixed
        # 0.9 m switch otherwise disappears exactly when entering a tight turn.
        branch_indices = sorted(set((1, int(np.searchsorted(travel, min(0.3, cfg.branch_distance))),
                                     int(np.searchsorted(travel, cfg.branch_distance)))))
        self._branch_travel = float(travel[branch_indices[-1]])
        self._branch_choices = np.array([travel[index] for index in branch_indices])
        self._next_steering = np.linspace(-cfg.max_steering_angle, cfg.max_steering_angle, 5)
        next_curvature = np.tile(np.tan(self._next_steering) / cfg.wheelbase,
                                 cfg.steering_samples)
        self._continuations = {}
        for index in branch_indices:
            branch = float(travel[index])
            length = cfg.planning_horizon - branch
            tail_travel = np.linspace(0.0, length, int(math.ceil(length / cfg.path_step)) + 1)
            tail_yaw = next_curvature[:, None] * tail_travel
            tx = tail_travel * np.sinc(tail_yaw / math.pi)
            ty = tail_travel * np.sin(tail_yaw / 2) * np.sinc(tail_yaw / (2 * math.pi))
            start_yaw = np.repeat(yaw[:, index], 5)[:, None]
            start_x = np.repeat(x[:, index], 5)[:, None]
            start_y = np.repeat(y[:, index], 5)[:, None]
            tail_x = start_x + tx * np.cos(start_yaw) - ty * np.sin(start_yaw)
            tail_y = start_y + tx * np.sin(start_yaw) + ty * np.cos(start_yaw)
            self._continuations[branch] = self._make_paths(
                tail_travel, tail_x, tail_y, tail_yaw + start_yaw, next_curvature)

    def _make_paths(self, travel, x, y, yaw, curvature):
        cfg = self.config
        cosine, sine = np.cos(yaw), np.sin(yaw)
        corner = math.hypot(max(cfg.vehicle_front, cfg.vehicle_rear), cfg.bubble_radius)
        pad = cfg.path_step / 2 * (1 + corner * np.abs(curvature))
        lateral = np.linspace(-cfg.bubble_radius, cfg.bubble_radius,
                              max(3, int(math.ceil(2 * cfg.bubble_radius / 0.05)) + 1))
        bx, by = np.meshgrid(np.linspace(cfg.lidar_offset, cfg.vehicle_front, 3), lateral)
        px = x[..., None] + cosine[..., None] * bx.ravel() - sine[..., None] * by.ravel() - cfg.lidar_offset
        py = y[..., None] + sine[..., None] * bx.ravel() + cosine[..., None] * by.ravel()
        inside = ((px + cfg.lidar_offset >= -cfg.vehicle_rear)
                  & (px + cfg.lidar_offset <= cfg.vehicle_front)
                  & (np.abs(py) <= cfg.bubble_radius))
        return PathGeometry(travel, x, y, cosine, sine, pad, np.hypot(px, py),
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
        """Consolidate returns once; keep every occupied cell with bounded error."""
        cfg = self.config
        px = distances[hits] * geometry.cosine[hits] + cfg.lidar_offset
        py = distances[hits] * geometry.sine[hits]
        cell_size = 0.03
        cells = np.unique(np.floor(np.column_stack((px, py)) / cell_size), axis=0)
        px, py = ((cells + 0.5) * cell_size).T
        return px, py, cell_size / math.sqrt(2)

    def _evaluate_paths(self, distances, valid, hits, geometry, paths=None, points=None):
        """Return observed travel and wall clearance for every steering sample."""
        cfg = self.config
        paths = self._paths if paths is None else paths
        angles = geometry.angles
        right = np.searchsorted(angles, paths.probe_angle)
        left = np.clip(right - 1, 0, angles.size - 1)
        right = np.clip(right, 0, angles.size - 1)
        observed = ((paths.probe_angle >= angles[0]) & (paths.probe_angle <= angles[-1])
                    & (angles[right] - angles[left] <= 1.5 * geometry.step)
                    & valid[left] & valid[right])
        visible_range = np.minimum(distances[left], distances[right])
        visible = observed & (paths.probe_range <= visible_range)
        blocked = ~np.all(visible | paths.probe_inside, axis=2)
        blocked |= self._unknown_path_blocks(valid, geometry, paths)

        # All returns, including side/rear returns, constrain the entire body.
        # No-return max-range beams describe free space, not physical walls.
        px, py, cell_pad = (self._obstacle_points(distances, hits, geometry)
                            if points is None else points)
        wall_clearance = np.full(paths.x.shape, cfg.max_lidar_range)
        if px.size:
            # Consolidate returns into 3 cm cells; retaining each occupied cell
            # preserves thin obstacles. Expand every collision check by the cell
            # half-diagonal so this is conservative, not beam subsampling.
            # A path ends at its first unknown-space sample. Checking obstacles
            # beyond that point cannot extend it, so omit that work.
            reach = float(np.max(np.hypot(paths.x, paths.y))) + math.hypot(
                max(cfg.vehicle_front, cfg.vehicle_rear), cfg.bubble_radius) + 1.0 + cell_pad
            nearby = np.hypot(px, py) <= reach
            px, py = px[nearby], py[nearby]
            visible_count = np.where(np.any(blocked, axis=1),
                                     np.argmax(blocked, axis=1), paths.travel.size)
            for row, count in enumerate(visible_count):
                if not px.size:
                    break
                if count == 0:
                    continue
                dx = px - paths.x[row, :count, None]
                dy = py - paths.y[row, :count, None]
                cosine, sine = paths.cosine[row, :count, None], paths.sine[row, :count, None]
                local_x = dx * cosine + dy * sine
                local_y = -dx * sine + dy * cosine
                long_distance = np.maximum(-cfg.vehicle_rear - local_x,
                                           local_x - cfg.vehicle_front)
                side_distance = np.abs(local_y) - cfg.bubble_radius
                wall_clearance[row, :count] = np.min(
                    np.maximum(long_distance, side_distance), axis=1) - cell_pad
            # Padding at s=0 would forbid departures beside a safe nearby wall.
            pad = np.broadcast_to(paths.sweep_pad[:, None], wall_clearance.shape).copy()
            at_current_pose = np.hypot(paths.x[:, 0], paths.y[:, 0]) < 1e-9
            pad[at_current_pose, 0] = 0.0
            blocked |= wall_clearance <= pad
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
        return geometry

    def _score_continuations(self, distances, valid, hits, geometry, travel, base_score, points):
        """Look beyond each first arc, using a shorter switch in confined space."""
        cfg = self.config
        switches = np.full(travel.shape, self._branch_choices[0])
        for branch in self._branch_choices:
            switches = np.where(travel >= branch + cfg.path_step / 2, branch, switches)
        eligible = travel >= switches + cfg.path_step / 2
        planned_distance = travel.copy()
        next_curvature = np.abs(self._curvature).copy()
        if not np.any(eligible):
            return base_score, planned_distance, next_curvature, switches
        roots = []
        previous = int(np.argmin(np.abs(self._steering - self._previous_steering)))
        for index in (int(np.argmax(np.where(eligible, base_score, -np.inf))),
                      previous, 0, len(travel) - 1, len(travel) // 2):
            if eligible[index] and index not in roots:
                roots.append(index)
        for index in np.argsort(-np.where(eligible, base_score, -np.inf)):
            if len(roots) >= 5:
                break
            if eligible[index] and all(abs(self._steering[index] - self._steering[root]) >= 0.10
                                       for root in roots):
                roots.append(int(index))
        scores = base_score.copy()
        advantage = np.full(travel.shape, -np.inf)
        for branch in self._branch_choices:
            # In contracting space, compare an earlier steering change even
            # when the nominal switch is still reachable. Otherwise a weak
            # first arc can conceal the need to tighten the turn before 0.9 m.
            group = [root for root in roots
                     if travel[root] >= branch + cfg.path_step / 2
                     and (switches[root] == branch
                          or (np.max(travel) < 2.5 and branch >= min(0.3, cfg.branch_distance)))]
            if not group:
                continue
            indices = (np.array(group)[:, None] * 5 + np.arange(5)).ravel()
            paths = self._continuations[float(branch)].select(indices)
            onward, _ = self._evaluate_paths(distances, valid, hits, geometry,
                                             paths=paths, points=points)
            last = np.searchsorted(paths.travel, onward, side='right') - 1
            row = np.arange(len(onward))
            displacement = np.hypot(paths.x[row, last], paths.y[row, last])
            effective = np.minimum(onward + branch, displacement * 1.5 + 0.5)
            onward_score = (3.0 * effective / cfg.planning_horizon
                            - 0.10 * np.abs(np.tile(self._next_steering, len(group)))
                            / cfg.max_steering_angle)
            for i, root in enumerate(group):
                choice = int(np.argmax(onward_score[i * 5:(i + 1) * 5]))
                value = float(onward_score[i * 5 + choice])
                if value > advantage[root]:
                    advantage[root] = value
                    scores[root] = base_score[root] + max(0.0, value)
                    planned_distance[root] = branch + onward[i * 5 + choice]
                    next_curvature[root] = abs(math.tan(self._next_steering[choice]) / cfg.wheelbase)
                    switches[root] = branch
        return scores, planned_distance, next_curvature, switches

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
        feasible = travel > cfg.stop_distance + 1e-9
        if not np.any(feasible):
            return self.stop('no_safe_path', clearance, float(np.max(travel)), preserve_steering=True)
        score = (1.0 * np.minimum(travel / 1.2, 1.0) + 1.2 * comfort
                 - 0.35 * np.abs(self._steering - preferred) / cfg.max_steering_angle
                 - 0.15 * np.abs(self._steering - self._previous_steering) / cfg.max_steering_angle
                 - 0.05 * np.abs(self._steering) / cfg.max_steering_angle)
        score, planned_distance, next_curvature, switches = self._score_continuations(
            distances, valid, hits, geometry, travel, score, points)
        best = int(np.argmax(np.where(feasible, score, -np.inf)))
        steering, path_clearance = float(self._steering[best]), float(travel[best])
        available = max(0.0, path_clearance - cfg.stop_distance)
        # Solve v*latency + v^2/(2*deceleration) <= available path distance.
        at = cfg.braking_decel * cfg.reaction_time
        brake_speed = math.sqrt(at * at + 2 * cfg.braking_decel * available) - at
        lateral_speed = math.sqrt(cfg.max_lateral_accel / max(abs(self._curvature[best]), 1e-6))
        # Begin slowing before the planned tighter turn. This only lowers the
        # first-arc speed bound; future clearance can never authorize more speed.
        future_lateral_speed = math.sqrt(cfg.max_lateral_accel / max(next_curvature[best], 1e-6))
        entry_speed = math.sqrt(at * at + future_lateral_speed ** 2
                                + 2 * cfg.braking_decel * switches[best]) - at
        cruise_speed = max(cfg.min_speed, cfg.max_speed * min(1.0, path_clearance / cfg.slow_distance))
        speed = min(cfg.max_speed, cruise_speed, brake_speed, lateral_speed, entry_speed,
                    self._previous_speed + cfg.max_acceleration * min(dt, 0.1))
        if path_clearance < cfg.min_path_distance + 1e-9:
            speed = min(speed, cfg.crawl_speed)
        self._previous_speed, self._previous_steering = float(speed), steering
        return GapCommand(steering, float(speed), 'ok', target, clearance,
                          path_clearance, float(planned_distance[best]))


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
                f'plan={command.planning_clearance:.2f} '
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
    blocked[np.abs(angles) < math.radians(10)] = 0.35
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
