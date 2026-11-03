"""Rounded turn entries with a narrowing apex and an initial heading error.

The dimensions are synthetic, not measurements from the user's distorted
photograph. A separate bounded-curvature reference proves these entries are
traversable; the controller receives only ray-cast scans. Ideal bicycle motion
cannot validate steering lag, traction, or physical-car performance.
"""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from test_gap_follow_standalone import (
    GapConfig, WHEELBASE, advance, body_intersects_wall, hairpin,
)
from test_gap_follow_traps import run_scenario


def rounded_entry(direction=1):
    return hairpin(direction, inner_radius=0.4, outer_radius=1.6)


def narrowing_rounded_turn(direction=1):
    """A 1.6 m-wide entry narrows as the outer wall curls toward the divider."""
    walls = []
    angles = np.linspace(-math.pi / 2, math.pi / 2, 81)
    for radius, center_x in ((0.5, 0.0), (2.1, -0.8)):
        points = np.stack((center_x + radius * np.cos(angles),
                           radius * np.sin(angles)), axis=1)
        walls.extend(zip(points[:-1], points[1:]))
        walls.extend((([-12.0, -radius], [center_x, -radius]),
                      ([center_x, radius], [-12.0, radius])))
    walls = np.asarray(walls)
    walls[:, :, 1] *= direction
    return walls


def early_turn_reference(direction=1):
    """A radius-1 m U turn starting 0.1 m before the inner arc's center.

    The steering limit allows this path. Its padded inner edge stays outside
    the divider and its outer corners fit inside the displaced outer arc.
    """
    poses = [np.array([x, -1.0, 0.0]) for x in np.linspace(-2.5, -0.1, 241)]
    poses.extend(np.array([-0.1 + math.sin(a), -math.cos(a), a])
                 for a in np.linspace(0.0, math.pi, 316)[1:])
    poses.extend(np.array([x, 1.0, math.pi])
                 for x in np.linspace(-0.1, -0.75, 66)[1:])
    poses = np.asarray(poses)
    poses[:, 1:] *= direction
    return poses


def feasible_reference(direction=1, heading_error=0.15):
    """Undo the entry heading, then take a constant-radius semicircle.

    This is an offline feasibility witness, not a route supplied to the planner.
    Every pose can be checked against the complete synthetic wall segments.
    """
    config = GapConfig()
    pose = np.array([-2.0, -1.0, heading_error])
    poses, steering_angles = [pose.copy()], []

    def drive(distance, steering):
        nonlocal pose
        count = max(1, math.ceil(distance / 0.01))
        command = SimpleNamespace(speed=1.0, steering_angle=steering)
        for _ in range(count):
            pose = advance(pose, command, distance / count)
            poses.append(pose.copy())
            steering_angles.append(steering)

    minimum_radius = WHEELBASE / math.tan(config.max_steering_angle)
    drive(heading_error * minimum_radius, -config.max_steering_angle)
    drive(-float(pose[0]), 0.0)
    radius = -float(pose[1])
    drive(math.pi * radius, math.atan(WHEELBASE / radius))
    drive(0.75, 0.0)
    poses = np.asarray(poses)
    poses[:, 1:] *= direction
    return poses, direction * np.asarray(steering_angles)


class CurvedEntryTests(unittest.TestCase):
    def test_narrowing_rounded_turn_has_a_feasible_early_turn(self):
        config = GapConfig()
        self.assertLess(math.atan(WHEELBASE / 1.0), config.max_steering_angle)
        for direction in (-1, 1):
            walls = narrowing_rounded_turn(direction)
            for pose in early_turn_reference(direction):
                self.assertFalse(body_intersects_wall(
                    walls, pose, front=config.vehicle_front,
                    rear=config.vehicle_rear, half_width=config.bubble_radius))

    def test_broad_entry_commits_before_narrowing_rounded_apex(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                outcome, poses, commands = run_scenario(
                    narrowing_rounded_turn(direction),
                    (-2.5, -float(direction), 0.0),
                    lambda p: p[0] < -0.5 and direction * p[1] > 0.5,
                    steps=180)
                self.assertEqual(outcome, 'finished',
                                 f'{outcome}; last pose={poses[-1]}; '
                                 f'command={commands[-1]}')

    def test_heading_error_entries_have_feasible_bounded_curvature_paths(self):
        config = GapConfig()
        for direction in (-1, 1):
            for heading_error in (0.15, 0.20):
                with self.subTest(direction=direction, heading_error=heading_error):
                    walls = rounded_entry(direction)
                    poses, steering = feasible_reference(direction, heading_error)
                    self.assertLessEqual(float(np.max(np.abs(steering))),
                                         config.max_steering_angle + 1e-12)
                    for pose in poses:
                        self.assertFalse(body_intersects_wall(
                            walls, pose, front=config.vehicle_front,
                            rear=config.vehicle_rear, half_width=config.bubble_radius))
                    self.assertLess(poses[-1, 0], -0.5)
                    self.assertGreater(direction * poses[-1, 1], 0.4)

    def test_heading_error_does_not_lure_car_into_inner_corner(self):
        for direction in (-1, 1):
            for heading_error in (0.15, 0.20):
                with self.subTest(direction=direction, heading_error=heading_error):
                    outcome, poses, commands = run_scenario(
                        rounded_entry(direction),
                        (-2.0, -float(direction), direction * heading_error),
                        lambda p: p[0] < -0.5 and direction * p[1] > 0.4,
                        steps=180)
                    self.assertEqual(outcome, 'finished',
                                     f'{outcome}; last pose={poses[-1]}; '
                                     f'command={commands[-1]}')


if __name__ == '__main__':
    unittest.main()
