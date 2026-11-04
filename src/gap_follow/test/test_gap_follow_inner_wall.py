"""Inside-line preference must preserve observed-space and stopping constraints.

These static LiDAR fixtures use ideal bicycle geometry; they do not establish
physical-car grip, steering response, or lap-time improvement.
"""

import math
import unittest

import numpy as np

from test_gap_follow_standalone import (
    ANGLES, GapConfig, GapFollower, advance, body_intersects_wall, compute,
    corridor, hairpin, raycast,
)


def reference_config(**overrides):
    settings = dict(slow_distance=1.5,
                    max_lateral_accel=1.5, path_step=0.1, inner_wall_bias=0.0,
                    inner_wall_clearance=0.10)
    settings.update(overrides)
    return GapConfig(**settings)


def turning_pose(direction=1, radius=1.5, angle=0.7):
    return (radius * math.sin(angle), -direction * radius * math.cos(angle),
            direction * angle)


class InnerWallTests(unittest.TestCase):
    def compare(self, scan, **overrides):
        disabled = GapFollower(reference_config(inner_wall_bias=0.0, **overrides))
        enabled = GapFollower(reference_config(inner_wall_bias=1.0, **overrides))
        return compute(disabled, scan), compute(enabled, scan)

    def test_zero_bias_ignores_clearance_preference(self):
        scan = raycast(hairpin(), turning_pose())
        first = compute(GapFollower(reference_config(inner_wall_clearance=0.0)), scan)
        second = compute(GapFollower(reference_config(inner_wall_clearance=0.8)), scan)
        self.assertEqual(first, second)

    def test_straight_and_off_center_correction_are_unchanged(self):
        for offset in (-0.45, 0.0, 0.45):
            with self.subTest(offset=offset):
                before, after = self.compare(raycast(corridor(), (0.0, offset, 0.0)))
                self.assertEqual(before, after)

    def test_turn_without_an_inner_divider_is_unchanged(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                walls = hairpin(direction)
                # The helper emits the inner arc and its straights first.
                outer_only = walls[len(walls) // 2:]
                before, after = self.compare(raycast(outer_only, turning_pose(direction)))
                self.assertGreater(direction * before.steering_angle, 0.1)
                self.assertEqual(before, after)

    def test_short_isolated_obstacle_does_not_establish_an_inner_wall(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                walls = hairpin(direction)
                pose = turning_pose(direction)
                tangent = np.array([math.cos(pose[2]), math.sin(pose[2])])
                inside = direction * np.array([-math.sin(pose[2]), math.cos(pose[2])])
                center = np.asarray(pose[:2]) + 0.65 * inside
                post = np.array([[center - 0.025 * tangent, center + 0.025 * tangent]])
                outer_and_post = np.concatenate((walls[len(walls) // 2:], post))
                before, after = self.compare(raycast(outer_and_post, pose))
                self.assertGreater(direction * before.steering_angle, 0.1)
                self.assertEqual(before, after)

    def test_mirrored_bias_moves_the_replanned_line_toward_the_inner_wall(self):
        # A broad annular bend with a slightly outward entry heading offers
        # comparable checked arcs. Compare the driven lines at matched travel
        # distances so speed changes cannot masquerade as a different line.
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                walls = hairpin(direction, inner_radius=1.0, outer_radius=2.7)
                radius, angle = 1.9, 0.7
                start = np.array([radius * math.sin(angle),
                                  -direction * radius * math.cos(angle),
                                  direction * (angle - 0.15)])
                profiles, first_commands = [], []
                for bias in (0.0, 1.0):
                    config = reference_config(inner_wall_bias=bias)
                    follower = GapFollower(config)
                    pose = start.copy()
                    travelled, radii = [0.0], [radius]
                    for step in range(20):
                        command = compute(follower, raycast(walls, pose), dt=0.1)
                        if step == 0:
                            first_commands.append(command)
                        self.assertGreater(command.speed, 0.0)
                        for _ in range(5):
                            pose = advance(pose, command, 0.02)
                            self.assertFalse(body_intersects_wall(
                                walls, pose, front=config.vehicle_front,
                                rear=config.vehicle_rear, half_width=config.bubble_radius))
                        travelled.append(travelled[-1] + command.speed * 0.1)
                        radii.append(float(np.hypot(pose[0], pose[1])))
                    profiles.append((np.asarray(travelled), np.asarray(radii)))
                before, after = first_commands
                # Replanning can start with the same steering and diverge
                # later. The line comparison below measures the behavior.
                self.assertGreater(direction * before.steering_angle, 0.0)
                self.assertGreater(direction * after.steering_angle, 0.0)
                # Remove acceleration-ramp suppression when establishing the
                # original route's speed ceiling and required braking travel.
                reference = GapFollower(reference_config())
                reference._previous_speed = reference.config.max_speed
                ceiling = compute(reference, raycast(walls, start), dt=0.1).speed
                needed = (ceiling * reference.config.reaction_time
                          + ceiling ** 2 / (2 * reference.config.braking_decel))
                self.assertGreaterEqual(after.path_clearance, needed - 1e-9)
                self.assertGreaterEqual(after.path_clearance,
                                        before.path_clearance - 0.10 - 1e-9)
                # Compare at matched travelled distances, so a speed change
                # cannot masquerade as a different line through the bend.
                common_end = min(profile[0][-1] for profile in profiles)
                samples = np.linspace(0.1, common_end, 80)
                radial = [np.interp(samples, travelled, radii)
                          for travelled, radii in profiles]
                self.assertLess(float(np.mean(radial[1])),
                                float(np.mean(radial[0])) - 0.01)

    def test_preference_cannot_override_occupied_body_or_unknown_forward_ray(self):
        walls = np.concatenate((corridor(), [[[0.40, -1.0], [0.40, 1.0]]]))
        occupied = raycast(walls)
        missing = raycast(corridor())
        missing[int(np.argmin(np.abs(ANGLES)))] = np.nan
        for scan in (occupied, missing):
            with self.subTest(missing=bool(np.any(np.isnan(scan)))):
                command = compute(GapFollower(reference_config(inner_wall_bias=1.0)), scan)
                self.assertEqual(command.speed, 0.0)
                self.assertEqual(command.path_clearance, 0.0)

    def test_enabled_preference_keeps_braking_and_curvature_speed_bounds(self):
        config = reference_config(inner_wall_bias=1.0, max_speed=4.0)
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                follower = GapFollower(config)
                follower._previous_speed = 4.0
                command = compute(follower, raycast(hairpin(direction), turning_pose(direction)))
                self.assertGreater(command.speed, 0.0)
                curvature = abs(math.tan(command.steering_angle) / config.wheelbase)
                self.assertLessEqual(command.speed ** 2 * curvature,
                                     config.max_lateral_accel + 1e-9)
                needed = (command.speed * config.reaction_time
                          + command.speed ** 2 / (2 * config.braking_decel))
                self.assertLessEqual(needed, command.path_clearance + 1e-9)

    def test_inner_wall_configuration_rejects_invalid_values(self):
        invalid = ({'inner_wall_bias': -0.01}, {'inner_wall_bias': 1.01},
                   {'inner_wall_bias': float('nan')}, {'inner_wall_bias': True},
                   {'inner_wall_clearance': -0.01}, {'inner_wall_clearance': 1.01})
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    reference_config(**override)
        reference_config(inner_wall_bias=0.0, inner_wall_clearance=0.0)
        reference_config(inner_wall_bias=1.0, inner_wall_clearance=0.1)


if __name__ == '__main__':
    unittest.main()
