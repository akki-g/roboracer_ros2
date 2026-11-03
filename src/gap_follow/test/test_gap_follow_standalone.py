"""Portable geometric regressions; run with ``python3 -m unittest discover
-s src/gap_follow/test -v``. These scans model static walls and ideal steering,
not tire dynamics, actuator lag, reflective surfaces, or real-car validation.
"""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'gap_follow_standalone.py'
SPEC = importlib.util.spec_from_file_location('gap_follow_standalone', MODULE_PATH)
CONTROLLER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTROLLER
SPEC.loader.exec_module(CONTROLLER)
GapConfig = CONTROLLER.GapConfig
GapFollower = CONTROLLER.GapFollower

ANGLES = np.linspace(-3 * math.pi / 4, 3 * math.pi / 4, 1081)
ANGLE_STEP = float(ANGLES[1] - ANGLES[0])
RANGE_MAX = 30.0
LIDAR_OFFSET = GapConfig().lidar_offset
WHEELBASE = 0.33


def raycast(segments, pose=(0.0, 0.0, 0.0), angles=ANGLES):
    """Return exact ray/line-segment intersections in the moving LiDAR frame."""
    x, y, yaw = pose
    origin = np.array([x + LIDAR_OFFSET * math.cos(yaw),
                       y + LIDAR_OFFSET * math.sin(yaw)])
    directions = np.stack((np.cos(angles + yaw), np.sin(angles + yaw)), axis=1)
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    vectors = segments[:, 1] - segments[:, 0]
    offsets = segments[:, 0] - origin
    denominator = (directions[:, None, 0] * vectors[None, :, 1]
                   - directions[:, None, 1] * vectors[None, :, 0])
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = (offsets[:, 0] * vectors[:, 1]
                    - offsets[:, 1] * vectors[:, 0])[None, :] / denominator
        fraction = (offsets[None, :, 0] * directions[:, None, 1]
                    - offsets[None, :, 1] * directions[:, None, 0]) / denominator
    hit = ((np.abs(denominator) > 1e-12) & (distance >= 0)
           & (fraction >= 0) & (fraction <= 1))
    return np.minimum(RANGE_MAX, np.min(np.where(hit, distance, np.inf), axis=1))


def corridor(half_width=1.0, end=12.0):
    return np.array([[[-10.0, -half_width], [end, -half_width]],
                     [[-10.0, half_width], [end, half_width]]])


def hairpin(direction=1, inner_radius=0.5, outer_radius=2.1):
    """A U-shaped corridor: drive east, turn 180 degrees, then drive west."""
    segments = []
    for radius in (inner_radius, outer_radius):
        points = np.stack((radius * np.cos(np.linspace(-math.pi / 2, math.pi / 2, 81)),
                           radius * np.sin(np.linspace(-math.pi / 2, math.pi / 2, 81))), axis=1)
        segments.extend(zip(points[:-1], points[1:]))
        segments.extend((([-12.0, -radius], [0.0, -radius]),
                         ([0.0, radius], [-12.0, radius])))
    result = np.asarray(segments)
    result[:, :, 1] *= direction
    return result


def rectangular_hairpin(direction=1):
    """A squared U bend that traps bearing-only gap following near its end wall."""
    result = np.array([[[-12.0, -1.8], [2.2, -1.8]],
                       [[2.2, -1.8], [2.2, 1.8]],
                       [[2.2, 1.8], [-12.0, 1.8]],
                       [[-12.0, -0.3], [0.0, -0.3]],
                       [[0.0, -0.3], [0.0, 0.3]],
                       [[0.0, 0.3], [-12.0, 0.3]]])
    result[:, :, 1] *= direction
    return result


def advance(pose, command, dt):
    """Integrate the commanded bicycle arc exactly, without dynamic slip."""
    x, y, yaw = pose
    curvature = math.tan(command.steering_angle) / WHEELBASE
    distance = command.speed * dt
    if abs(curvature) < 1e-10:
        return np.array([x + distance * math.cos(yaw),
                         y + distance * math.sin(yaw), yaw])
    next_yaw = yaw + curvature * distance
    return np.array([x + (math.sin(next_yaw) - math.sin(yaw)) / curvature,
                     y - (math.cos(next_yaw) - math.cos(yaw)) / curvature,
                     next_yaw])


def body_intersects_wall(segments, pose, front=0.43, rear=0.12, half_width=0.155):
    """Clip each wall segment against the physical rectangular car footprint."""
    x, y, yaw = pose
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    local = (np.asarray(segments) - [x, y]) @ rotation
    starts, vectors = local[:, 0], local[:, 1] - local[:, 0]
    enter = np.zeros(len(local))
    leave = np.ones(len(local))
    possible = np.ones(len(local), dtype=bool)
    for axis, low, high in ((0, -rear, front), (1, -half_width, half_width)):
        parallel = np.abs(vectors[:, axis]) < 1e-12
        possible &= ~(parallel & ((starts[:, axis] < low) | (starts[:, axis] > high)))
        with np.errstate(divide='ignore', invalid='ignore'):
            first = (low - starts[:, axis]) / vectors[:, axis]
            last = (high - starts[:, axis]) / vectors[:, axis]
        enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(first, last)))
        leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(first, last)))
    return bool(np.any(possible & (enter <= leave)))


def compute(follower, ranges, reverse=False, dt=0.025):
    if reverse:
        return follower.compute(ranges[::-1], float(ANGLES[-1]), -ANGLE_STEP, 0.05, RANGE_MAX, dt=dt)
    return follower.compute(ranges, float(ANGLES[0]), ANGLE_STEP, 0.05, RANGE_MAX, dt=dt)


def rollout(direction=1, steps=350, dt=0.1, config=None, rectangular=False):
    """Return a diagnostic trajectory and outcome for one complete hairpin."""
    walls = rectangular_hairpin(direction) if rectangular else hairpin(direction)
    pose = np.array([-2.5, -direction * (1.05 if rectangular else 1.3), 0.0])
    follower = GapFollower(config or GapConfig())
    trajectory = [pose.copy()]
    commands = []
    for _ in range(steps):
        command = compute(follower, raycast(walls, pose), dt=dt)
        commands.append(command)
        if command.speed <= 1e-5:
            return 'stopped:' + command.reason, np.asarray(trajectory), commands
        # Check intermediate footprints as well as each control-cycle endpoint.
        for _ in range(5):
            pose = advance(pose, command, dt / 5)
            if body_intersects_wall(walls, pose):
                trajectory.append(pose.copy())
                return 'collision', np.asarray(trajectory), commands
        trajectory.append(pose.copy())
        if pose[0] < -0.5 and direction * pose[1] > 0.5:
            return 'finished', np.asarray(trajectory), commands
    return 'timeout', np.asarray(trajectory), commands


class GapFollowerTests(unittest.TestCase):
    def test_straight_corridor(self):
        command = compute(GapFollower(), raycast(corridor()))
        self.assertGreater(command.speed, 0)
        self.assertLessEqual(command.speed, GapConfig().max_speed)
        self.assertLess(abs(command.steering_angle), 0.03)

    def test_off_center_corridor_moves_away_from_near_wall(self):
        for offset in (-0.55, 0.55):
            with self.subTest(offset=offset):
                command = compute(GapFollower(), raycast(corridor(), (0.0, offset, 0.0)))
                self.assertGreater(command.speed, 0)
                self.assertLess(offset * command.steering_angle, 0)

    def test_blocked_corridor_stops(self):
        walls = np.concatenate((corridor(), [[[0.55, -1.0], [0.55, 1.0]]]))
        command = compute(GapFollower(), raycast(walls))
        self.assertEqual(command.speed, 0)

    def test_short_checked_path_allows_braking_bounded_crawl(self):
        # A nearby wall leaves 0.2 m of checked travel: less than the normal
        # planning threshold, but more than the stopping reserve. The previous
        # 0.3 m hard cutoff unnecessarily deadlocked this situation.
        config = GapConfig(min_speed=0.4, braking_decel=0.5, reaction_time=0.5)
        follower = GapFollower(config)
        for _ in range(5):
            previous = compute(follower, raycast(corridor()), dt=0.1)
        walls = np.concatenate((corridor(), [[[0.72, -1.0], [0.72, 1.0]]]))
        command = compute(follower, raycast(walls), dt=0.1)
        self.assertGreater(command.path_clearance, config.stop_distance)
        self.assertLess(command.path_clearance, config.min_path_distance)
        self.assertGreater(command.speed, 0)
        self.assertLess(command.speed, previous.speed)
        self.assertLessEqual(command.speed, config.crawl_speed)
        self.assertLess(command.speed, config.min_speed)
        stopping_distance = (command.speed * config.reaction_time
                             + command.speed ** 2 / (2 * config.braking_decel))
        self.assertLessEqual(stopping_distance,
                             command.path_clearance - config.stop_distance + 1e-9)

    def test_crawl_never_overrides_an_occupied_footprint(self):
        follower = GapFollower()
        moving = compute(follower, raycast(corridor()))
        self.assertGreater(moving.speed, 0)
        # This wall intersects the front body, not merely its future path.
        walls = np.concatenate((corridor(), [[[0.40, -1.0], [0.40, 1.0]]]))
        blocked = compute(follower, raycast(walls))
        self.assertEqual(blocked.speed, 0)
        self.assertEqual(blocked.path_clearance, 0)

    def test_geometric_stop_keeps_turn_command_but_invalid_scan_resets_it(self):
        follower = GapFollower()
        turning = compute(follower, raycast(corridor(), (0.0, -0.55, 0.0)))
        self.assertGreater(turning.speed, 0)
        self.assertGreater(turning.steering_angle, 0)
        walls = np.concatenate((corridor(), [[[0.55, -1.0], [0.55, 1.0]]]))
        blocked = compute(follower, raycast(walls))
        self.assertEqual(blocked.speed, 0)
        # A short visibility/clearance interruption should not unwind the
        # wheels and forget the corner while the car is stationary.
        self.assertEqual(blocked.steering_angle, turning.steering_angle)
        invalid = compute(follower, np.full(ANGLES.size, np.nan))
        self.assertEqual(invalid.speed, 0)
        self.assertEqual(invalid.steering_angle, 0)

    def test_continuation_and_crawl_configuration_rejects_invalid_limits(self):
        invalid_settings = (
            {'planning_horizon': 2.0},  # Cannot be shorter than the first arc.
            {'planning_horizon': 11.0},  # Cannot extend past the sensor cap.
            {'branch_distance': 0.2},
            {'branch_distance': 3.0},
            {'comfort_clearance': 0.0},
            {'crawl_speed': -0.1},
            {'crawl_speed': float('nan')},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                GapConfig(**settings)

    def test_invalid_ranges_stop(self):
        for invalid in (0.0, np.nan, np.inf, -np.inf, -1.0, RANGE_MAX + 1.0):
            with self.subTest(invalid=invalid):
                command = compute(GapFollower(), np.full(ANGLES.size, invalid))
                self.assertEqual(command.speed, 0)

    def test_invalid_metadata_stops(self):
        ranges = raycast(corridor())
        for angle_min, step, range_min, range_max in (
                (np.nan, ANGLE_STEP, 0.05, RANGE_MAX),
                (ANGLES[0], 0.0, 0.05, RANGE_MAX),
                (ANGLES[0], ANGLE_STEP, 1.0, 0.5),
                (ANGLES[0], ANGLE_STEP, -0.1, RANGE_MAX)):
            with self.subTest(metadata=(angle_min, step, range_min, range_max)):
                command = GapFollower().compute(ranges, angle_min, step, range_min, range_max)
                self.assertEqual(command.speed, 0)

    def test_missing_forward_sector_stops(self):
        ranges = raycast(corridor())
        ranges[np.abs(ANGLES) < math.radians(8)] = np.nan
        self.assertEqual(compute(GapFollower(), ranges).speed, 0)
        # Receiving a valid side-looking scan is not forward observation.
        self.assertEqual(GapFollower().compute(np.full(181, 10.0), math.pi / 2,
                                               ANGLE_STEP, 0.05, RANGE_MAX).speed, 0)

    def test_unknown_shoulders_cannot_be_driven_through(self):
        ranges = np.full(ANGLES.size, RANGE_MAX)
        unknown = ((np.abs(ANGLES) > math.radians(20))
                   & (np.abs(ANGLES) < math.radians(85)))
        ranges[unknown] = np.nan
        # The center rays remain valid, but the car's corners need unknown space.
        self.assertEqual(compute(GapFollower(), ranges).speed, 0)

    def test_single_unknown_front_ray_cannot_be_skipped_between_probes(self):
        for angle in (0.0, -0.5, 0.5, 2.0):
            with self.subTest(angle=angle):
                ranges = np.full(ANGLES.size, 4.0)
                ranges[np.argmin(np.abs(ANGLES - math.radians(angle)))] = np.nan
                self.assertEqual(compute(GapFollower(), ranges).speed, 0)

    def test_unknown_ray_behind_body_does_not_block_forward_departure(self):
        ranges = raycast(corridor())
        ranges[np.argmin(np.abs(ANGLES - math.radians(130)))] = np.nan
        command = compute(GapFollower(), ranges)
        self.assertGreater(command.speed, 0)
        self.assertLess(abs(command.steering_angle), 0.03)

    def test_reversed_scan_order_is_equivalent(self):
        ranges = raycast(hairpin(), (-0.5, -1.3, 0.15))
        forward = compute(GapFollower(), ranges)
        backward = compute(GapFollower(), ranges, reverse=True)
        self.assertAlmostEqual(forward.speed, backward.speed, places=7)
        self.assertAlmostEqual(forward.steering_angle, backward.steering_angle, places=7)

    def test_obstacle_at_body_side_stops(self):
        walls = np.concatenate((corridor(), [[[0.1, 0.1], [0.45, 0.1]]]))
        self.assertEqual(compute(GapFollower(), raycast(walls)).speed, 0)

    def test_rear_scan_edge_return_does_not_veto_forward_departure(self):
        # Reproduce the real scan's close peripheral return while the walls
        # are well outside the footprint. The full 270-degree scan stalls.
        ranges = raycast(corridor())
        ranges[-1] = 0.32
        full_scan = GapFollower(GapConfig(forward_fov_deg=270.0))
        self.assertEqual(compute(full_scan, ranges).speed, 0)
        command = compute(GapFollower(), ranges)
        self.assertGreater(command.speed, 0)
        self.assertLess(abs(command.steering_angle), 0.03)

    def test_close_obstacles_in_retained_scan_still_stop(self):
        # Front, both sides, and retained rear directions still constrain the
        # body. Cropping the extreme scan edge must not discard these returns.
        for angle in (0.0, -90.0, 90.0, -125.0, 125.0):
            with self.subTest(angle=angle):
                ranges = raycast(corridor())
                ranges[np.argmin(np.abs(ANGLES - math.radians(angle)))] = 0.20
                self.assertEqual(compute(GapFollower(), ranges).speed, 0)

    def test_infinity_policy_is_explicit(self):
        follower = GapFollower(GapConfig(inf_is_free=True))
        command = compute(follower, np.full(ANGLES.size, np.inf))
        self.assertGreater(command.speed, 0)
        # NaN and negative infinity never become open space under this setting.
        for invalid in (np.nan, -np.inf):
            self.assertEqual(compute(follower, np.full(ANGLES.size, invalid)).speed, 0)

    def test_obstacle_appearance_reduces_speed_within_braking_distance(self):
        config = GapConfig(max_speed=4.0)
        follower = GapFollower(config)
        for _ in range(40):
            previous = compute(follower, raycast(corridor()), dt=0.1)
        walls = np.concatenate((corridor(), [[[1.3, -1.0], [1.3, 1.0]]]))
        command = compute(follower, raycast(walls), dt=0.1)
        self.assertGreater(command.speed, 0)
        self.assertLess(command.speed, previous.speed)
        stopping_distance = (command.speed * config.reaction_time
                             + command.speed ** 2 / (2 * config.braking_decel))
        self.assertLessEqual(stopping_distance,
                             command.path_clearance - config.stop_distance + 1e-9)

    def test_left_and_right_hairpin_rollouts(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                outcome, poses, commands = rollout(direction)
                self.assertEqual(outcome, 'finished',
                                 f'{outcome}; last pose={poses[-1]}; command={commands[-1]}')
                self.assertTrue(all(0 < command.speed <= GapConfig().max_speed
                                    for command in commands))
                for command in commands:
                    lateral_accel = command.speed ** 2 * abs(math.tan(command.steering_angle)) / WHEELBASE
                    self.assertLessEqual(lateral_accel, GapConfig().max_lateral_accel + 1e-9)

    def test_left_and_right_rectangular_hairpin_rollouts(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                outcome, poses, commands = rollout(direction, rectangular=True)
                self.assertEqual(outcome, 'finished',
                                 f'{outcome}; last pose={poses[-1]}; command={commands[-1]}')


if __name__ == '__main__':
    unittest.main()
