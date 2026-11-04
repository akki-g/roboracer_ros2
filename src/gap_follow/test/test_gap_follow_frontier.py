"""Checked arcs stop at sensor limits and rebuild lookups for changed scans."""

import math
import unittest

import numpy as np

from test_gap_follow_standalone import (
    ANGLES, ANGLE_STEP, RANGE_MAX, GapConfig, GapFollower, hairpin, raycast,
)


class VisibilityTests(unittest.TestCase):
    def test_sensor_range_limit_bounds_checked_travel(self):
        follower = GapFollower(GapConfig(max_lidar_range=3.0))
        geometry = follower._scan_geometry(len(ANGLES), ANGLES[0], ANGLE_STEP)
        size = geometry.angles.size
        path = follower._paths.select([len(follower._steering) // 2])
        travel, _ = follower._evaluate_paths(
            np.full(size, 3.0), np.ones(size, dtype=bool), np.zeros(size, dtype=bool),
            geometry, paths=path)
        # The leading body reaches beyond the sensor cap before the rear axle.
        self.assertLess(float(travel[0]), follower.config.path_horizon)

    def test_leaving_the_scan_field_of_view_bounds_checked_travel(self):
        follower = GapFollower(GapConfig(forward_fov_deg=90.0))
        geometry = follower._scan_geometry(len(ANGLES), ANGLES[0], ANGLE_STEP)
        size = geometry.angles.size
        path = follower._paths.select([len(follower._steering) - 1])
        travel, _ = follower._evaluate_paths(
            np.full(size, 9.0), np.ones(size, dtype=bool), np.ones(size, dtype=bool),
            geometry, paths=path)
        self.assertLess(float(travel[0]), follower.config.path_horizon)

    def test_changed_scan_layout_rebuilds_cached_beam_indices(self):
        follower = GapFollower()
        walls, pose = hairpin(), (-0.5, -1.3, 0.15)
        for angles in (ANGLES, np.linspace(-3 * math.pi / 4, 3 * math.pi / 4, 541),
                       np.linspace(3 * math.pi / 4, -3 * math.pi / 4, 721)):
            with self.subTest(beams=len(angles), descending=angles[0] > angles[-1]):
                scan = raycast(walls, pose, angles=angles)
                follower.stop('reset_control_history')
                args = (scan, float(angles[0]), float(angles[1] - angles[0]), 0.05, RANGE_MAX)
                actual = follower.compute(*args)
                expected = GapFollower().compute(*args)
                self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
