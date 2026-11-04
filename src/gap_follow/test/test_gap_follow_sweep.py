"""Near-term swept-body regressions, independent of recorded scans or ROS.

The supplied point cells isolate collision checking from raycasting. Visibility
uses a fully observed scan except where a test explicitly removes a beam.
Dense fixture checks below use direct bicycle poses and point/rectangle distance,
not the planner's sweep padding or refinement implementation.
"""

import math
import unittest

import numpy as np

from test_gap_follow_standalone import ANGLES, ANGLE_STEP, GapConfig, GapFollower


CELL_PAD = 0.03 / math.sqrt(2)


def point_clearance(config, steering, point, travel):
    """Independent signed box clearance, reduced by occupied-cell uncertainty."""
    travel = np.asarray(travel)
    curvature = math.tan(steering) / config.wheelbase
    yaw = curvature * travel
    x = np.sin(yaw) / curvature
    y = (1 - np.cos(yaw)) / curvature
    dx, dy = point[0] - x, point[1] - y
    local_x = dx * np.cos(yaw) + dy * np.sin(yaw)
    local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
    return np.maximum.reduce((-config.vehicle_rear - local_x,
                              local_x - config.vehicle_front,
                              np.abs(local_y) - config.bubble_radius)) - CELL_PAD


class SweepTests(unittest.TestCase):
    def setup_arc(self, direction=-1, step=0.1):
        follower = GapFollower(GapConfig(
            slow_distance=1.5, path_step=step,
            wheelbase=0.33, vehicle_front=0.43, vehicle_rear=0.12,
            bubble_radius=0.25, lidar_offset=0.27, max_steering_angle=0.35))
        geometry = follower._scan_geometry(len(ANGLES), float(ANGLES[0]), ANGLE_STEP)
        steering = direction * follower.config.max_steering_angle
        curvature = math.tan(steering) / follower.config.wheelbase
        travel = np.linspace(0.0, 0.3, int(round(0.3 / step)) + 1)
        yaw = curvature * travel[None, :]
        x, y = np.sin(yaw) / curvature, (1 - np.cos(yaw)) / curvature
        # Install one primary arc so adaptive near-term checks exercise the
        # normal path-evaluation entry point.
        follower._paths = follower._make_paths(travel, x, y, yaw,
                                               np.array([curvature]))
        count = geometry.angles.size
        return (follower, geometry, steering, np.full(count, 9.0),
                np.ones(count, dtype=bool), np.zeros(count, dtype=bool))

    @staticmethod
    def evaluate(follower, geometry, distances, valid, hits, point):
        points = (np.array([point[0]]), np.array([point[1]]), CELL_PAD)
        travel, _ = follower._evaluate_paths(distances, valid, hits, geometry,
                                                points=points)
        return float(travel[0])

    def test_mirrored_departures_clear_nearby_front_corner(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                follower, geometry, steering, distances, valid, hits = self.setup_arc(direction)
                point = (0.525, -direction * 0.285)
                clearance = point_clearance(follower.config, steering, point,
                                            np.linspace(0, 0.3, 601))
                self.assertGreater(float(np.min(clearance)), 0.03)
                # This is a false veto of the original .1 m sampling pad,
                # despite positive clearance throughout the full interval.
                corner = math.hypot(follower.config.vehicle_front,
                                    follower.config.bubble_radius)
                old_pad = 0.05 * (1 + corner * abs(math.tan(steering) / follower.config.wheelbase))
                self.assertLess(float(point_clearance(follower.config, steering, point, 0.1)), old_pad)
                checked = self.evaluate(follower, geometry, distances, valid, hits, point)
                self.assertGreaterEqual(checked, 0.3 - 1e-9)

    def test_collision_between_clear_coarse_poses_still_blocks(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                follower, geometry, steering, distances, valid, hits = self.setup_arc(direction)
                # A stationary occupied cell swept by the outer front corner
                # after .025 m. Both coarse endpoint bodies clear the cell.
                point = (0.4600, -direction * 0.2367)
                endpoints = point_clearance(follower.config, steering, point, [0.0, 0.1])
                interior = point_clearance(follower.config, steering, point,
                                           np.linspace(0, 0.1, 201))
                self.assertTrue(np.all(endpoints > 0.0))
                self.assertLess(float(np.min(interior)), -0.01)
                checked = self.evaluate(follower, geometry, distances, valid, hits, point)
                self.assertEqual(checked, 0.0)

    def test_occupied_current_footprint_cannot_be_refined_away(self):
        follower, geometry, steering, distances, valid, hits = self.setup_arc()
        point = (0.435, 0.255)
        self.assertLess(float(point_clearance(follower.config, steering, point, 0.0)), 0.0)
        self.assertEqual(self.evaluate(follower, geometry, distances, valid, hits, point), 0.0)

    def test_obstacles_outside_path_reach_do_not_block_or_crash(self):
        follower, geometry, steering, distances, valid, hits = self.setup_arc()
        # Filtering all supplied points leaves an empty collision set; the
        # observed, clear arc must still be available without a reduction error.
        self.assertGreaterEqual(self.evaluate(follower, geometry, distances, valid, hits,
                                              (20.0, 20.0)), 0.3 - 1e-9)

    def test_departure_refinement_keeps_missing_forward_beam_blocked(self):
        follower, geometry, steering, distances, valid, hits = self.setup_arc()
        point = (0.525, 0.285)
        self.assertGreaterEqual(self.evaluate(follower, geometry, distances, valid, hits, point),
                                0.3 - 1e-9)
        forward = int(np.argmin(np.abs(geometry.angles)))
        valid[forward], distances[forward] = False, 0.0
        self.assertEqual(self.evaluate(follower, geometry, distances, valid, hits, point), 0.0)

    def test_inserted_midpoint_cannot_cross_an_unknown_ray(self):
        follower, geometry, steering, distances, valid, hits = self.setup_arc(step=0.3)
        point = (0.525, 0.285)
        self.assertGreaterEqual(self.evaluate(follower, geometry, distances, valid, hits, point),
                                0.3 - 1e-9)
        # This side ray intersects the new leading-body area at the inserted
        # .15 m pose, although the original 0 and .3 m poses do not enter it.
        beam = int(np.argmin(np.abs(geometry.angles + math.radians(71))))
        valid[beam], distances[beam] = False, 0.0
        self.assertFalse(np.any(follower._unknown_path_blocks(valid, geometry, follower._paths)))
        self.assertEqual(self.evaluate(follower, geometry, distances, valid, hits, point), 0.0)


if __name__ == '__main__':
    unittest.main()
