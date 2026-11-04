"""Occupied-cell aggregation must enclose returns without inventing full cells.

Ground-truth comparisons use every raw point and independent rectangle geometry.
These tests concern spatial aggregation, not LiDAR measurement uncertainty.
"""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from test_gap_follow_standalone import ANGLES, ANGLE_STEP, GapConfig, GapFollower


def reference_follower():
    return GapFollower(GapConfig(
        slow_distance=1.5,
        max_lateral_accel=1.5, path_step=0.1, inner_wall_bias=0.0,
        wheelbase=0.33, vehicle_front=0.43, vehicle_rear=0.12,
        bubble_radius=0.25, lidar_offset=0.27, max_steering_angle=0.35))


def aggregate(follower, raw_points):
    """Supply exact polar returns for a synthetic rear-axle-frame point set."""
    raw_points = np.asarray(raw_points, dtype=float)
    relative = raw_points - [follower.config.lidar_offset, 0.0]
    distances = np.hypot(relative[:, 0], relative[:, 1])
    geometry = SimpleNamespace(cosine=relative[:, 0] / distances,
                               sine=relative[:, 1] / distances)
    return follower._obstacle_points(distances, np.ones(len(distances), dtype=bool), geometry)


def raw_clearance(config, raw_points, pose):
    """Signed point-to-body clearance evaluated without any aggregation."""
    x, y, yaw = pose
    dx, dy = np.asarray(raw_points).T - np.array([[x], [y]])
    local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return float(np.min(np.maximum.reduce((-config.vehicle_rear - local_x,
                                          local_x - config.vehicle_front,
                                          np.abs(local_y) - config.bubble_radius))))


def aggregate_clearance(follower, points, pose):
    x, y, yaw = pose
    px, py, padding = points
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx, dy = px - x, py - y
    local_x, local_y = dx * cosine + dy * sine, -dx * sine + dy * cosine
    return float(follower._obstacle_clearance(local_x, local_y, cosine, sine, padding))


class ObstacleCellTests(unittest.TestCase):
    def test_parallel_wall_stays_outside_body_across_cell_boundaries(self):
        follower = reference_follower()
        x = np.linspace(0.08, 0.42, 61)
        # The old fixed .03 m cell model changed from overlap to clear as the
        # same side wall crossed y=.27, despite every return being outside.
        for direction in (-1, 1):
            for gap in (0.0023, 0.009, 0.0199, 0.0201, 0.0269):
                with self.subTest(direction=direction, gap=gap):
                    raw = np.column_stack((x, np.full(x.size, direction * (0.25 + gap))))
                    actual = raw_clearance(follower.config, raw, (0.0, 0.0, 0.0))
                    grouped = aggregate_clearance(follower, aggregate(follower, raw),
                                                  (0.0, 0.0, 0.0))
                    self.assertGreater(actual, 0.0)
                    self.assertGreater(grouped, 0.0)
                    self.assertAlmostEqual(grouped, actual, places=11)
        self.assertLess(0.255 - 0.25 - 0.03 / math.sqrt(2), 0.0)

    def test_rotated_cell_bounds_never_overstate_raw_point_clearance(self):
        follower = reference_follower()
        rng = np.random.default_rng(7401)
        cells = rng.integers([-12, -25], [45, 26], size=(70, 2))
        raw = ((cells[:, None, :] + rng.uniform(0.03, 0.97, (70, 6, 2))) * 0.03).reshape(-1, 2)
        points = aggregate(follower, raw)
        px, py, half_extent = points
        self.assertEqual(half_extent.shape, (2, len(px)))
        self.assertLess(len(px), len(raw))
        enclosed = ((np.abs(raw[:, None, 0] - px) <= half_extent[0] + 1e-12)
                    & (np.abs(raw[:, None, 1] - py) <= half_extent[1] + 1e-12))
        self.assertTrue(np.all(np.any(enclosed, axis=1)))
        raw_bounds = []
        for translation in ((0.0, 0.0), (0.15, -0.08), (0.6, 0.1), (-0.1, 0.2),
                            (2.3, -1.7), (-2.0, 1.3)):
            for yaw in (0.0, 0.27, 0.95, -1.4, 2.9):
                with self.subTest(translation=translation, yaw=yaw):
                    pose = (*translation, yaw)
                    actual = raw_clearance(follower.config, raw, pose)
                    grouped = aggregate_clearance(follower, points, pose)
                    raw_bounds.append(actual)
                    self.assertLessEqual(grouped, actual + 1e-12)
        self.assertLess(min(raw_bounds), 0.0)
        self.assertGreater(max(raw_bounds), 0.0)

    def test_one_real_inside_return_cannot_be_averaged_out_of_its_cell(self):
        follower = reference_follower()
        raw = np.vstack((np.tile([0.205, 0.269], (100, 1)), [[0.205, 0.249]]))
        points = aggregate(follower, raw)
        self.assertLess(raw_clearance(follower.config, raw, (0.0, 0.0, 0.0)), 0.0)
        self.assertLess(aggregate_clearance(follower, points, (0.0, 0.0, 0.0)), 0.0)
        geometry = follower._scan_geometry(len(ANGLES), float(ANGLES[0]), ANGLE_STEP)
        count = geometry.angles.size
        checked, _ = follower._evaluate_paths(
            np.full(count, 9.0), np.ones(count, dtype=bool), np.zeros(count, dtype=bool),
            geometry, points=points)
        self.assertTrue(np.all(checked == 0.0))

    def test_thin_return_hit_between_clear_sampled_poses_still_blocks(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                follower = reference_follower()
                steering = direction * follower.config.max_steering_angle
                curvature = math.tan(steering) / follower.config.wheelbase
                travel = np.array([0.0, 0.1, 0.2, 0.3])
                yaw = curvature * travel
                x, y = np.sin(yaw) / curvature, (1 - np.cos(yaw)) / curvature
                follower._paths = follower._make_paths(
                    travel, x[None, :], y[None, :], yaw[None, :], np.array([curvature]))
                raw = np.array([[0.45, -direction * 0.235]])
                first = raw_clearance(follower.config, raw, (x[0], y[0], yaw[0]))
                last = raw_clearance(follower.config, raw, (x[1], y[1], yaw[1]))
                dense = np.linspace(0.0, 0.1, 201)
                intermediate = [raw_clearance(follower.config, raw,
                    (math.sin(curvature * s) / curvature,
                     (1 - math.cos(curvature * s)) / curvature, curvature * s)) for s in dense]
                self.assertGreater(first, 0.0)
                self.assertGreater(last, 0.0)
                self.assertLess(min(intermediate), -0.001)
                geometry = follower._scan_geometry(len(ANGLES), float(ANGLES[0]), ANGLE_STEP)
                count = geometry.angles.size
                checked, _ = follower._evaluate_paths(
                    np.full(count, 9.0), np.ones(count, dtype=bool), np.zeros(count, dtype=bool),
                    geometry, points=aggregate(follower, raw))
                self.assertEqual(float(checked[0]), 0.0)

    def test_scalar_padding_retains_legacy_clearance_semantics(self):
        follower = reference_follower()
        raw = np.array([[0.525, 0.285], [0.6, -0.31], [-0.18, 0.42]])
        padding = 0.03 / math.sqrt(2)
        points = (raw[:, 0], raw[:, 1], padding)
        for yaw in (0.0, 0.3, -1.1):
            pose = (0.1, -0.04, yaw)
            expected = raw_clearance(follower.config, raw, pose) - padding
            self.assertAlmostEqual(aggregate_clearance(follower, points, pose), expected, places=12)


if __name__ == '__main__':
    unittest.main()
