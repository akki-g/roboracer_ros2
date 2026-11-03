"""Closed-loop regressions for turn exits, changing widths and misleading pockets.

These use static segment scans and ideal bicycle motion. They do not model
actuator lag, slipping tires, or prove physical-car lap completion.
"""

import unittest

import numpy as np

from test_gap_follow_standalone import (
    GapConfig, GapFollower, advance, body_intersects_wall, compute, raycast,
)


def dogleg(direction=1, half_width=0.7, offset=2.0):
    """East/north/east passage with two opposing squared turns."""
    w = half_width
    walls = np.array([
        [[-8.0, -w], [2.0 + w, -w]],
        [[2.0 + w, -w], [2.0 + w, offset - w]],
        [[2.0 + w, offset - w], [12.0, offset - w]],
        [[-8.0, w], [2.0 - w, w]],
        [[2.0 - w, w], [2.0 - w, offset + w]],
        [[2.0 - w, offset + w], [12.0, offset + w]],
    ])
    walls[:, :, 1] *= direction
    return walls


def forward_pocket(direction=1):
    """A visible dead end straight ahead, with the through route to the side."""
    walls = np.array([
        [[-8.0, -0.7], [4.0, -0.7]],
        [[4.0, -0.7], [4.0, 0.7]],
        [[4.0, 0.7], [2.0, 0.7]],
        [[2.0, 0.7], [2.0, 10.0]],
        [[0.0, 10.0], [0.0, 0.7]],
        [[0.0, 0.7], [-8.0, 0.7]],
    ])
    walls[:, :, 1] *= direction
    return walls


def side_pocket(direction=1):
    """A six-metre-wide recess alongside an otherwise straight passage."""
    walls = np.array([
        [[-8.0, -0.7], [12.0, -0.7]],
        [[-8.0, 0.7], [0.0, 0.7]],
        [[0.0, 0.7], [0.0, 6.7]],
        [[0.0, 6.7], [6.0, 6.7]],
        [[6.0, 6.7], [6.0, 0.7]],
        [[6.0, 0.7], [12.0, 0.7]],
    ])
    walls[:, :, 1] *= direction
    return walls


def changing_width_dogleg(direction=1, half_width=0.6, offset=2.0):
    """A wide entry narrows before an S bend and widens again after it."""
    w, wide = half_width, 1.2
    lower = [[-8.0, -wide], [-2.0, -wide], [-1.0, -w], [2.0 + w, -w],
             [2.0 + w, offset - w], [4.0, offset - w], [5.0, offset - wide],
             [12.0, offset - wide]]
    upper = [[-8.0, wide], [-2.0, wide], [-1.0, w], [2.0 - w, w],
             [2.0 - w, offset + w], [4.0, offset + w], [5.0, offset + wide],
             [12.0, offset + wide]]
    walls = np.array(list(zip(lower[:-1], lower[1:]))
                     + list(zip(upper[:-1], upper[1:])))
    walls[:, :, 1] *= direction
    return walls


def run_scenario(walls, start, reached_goal, config=None, steps=250, dt=0.1):
    """Evaluate scans and check the physical body between each command update."""
    follower = GapFollower(config or GapConfig())
    pose = np.asarray(start, dtype=float)
    trajectory, commands = [pose.copy()], []
    for _ in range(steps):
        command = compute(follower, raycast(walls, pose), dt=dt)
        commands.append(command)
        if command.speed <= 1e-5:
            return 'stopped:' + command.reason, np.asarray(trajectory), commands
        for _ in range(5):
            pose = advance(pose, command, dt / 5)
            if body_intersects_wall(walls, pose):
                trajectory.append(pose.copy())
                return 'collision', np.asarray(trajectory), commands
        trajectory.append(pose.copy())
        if reached_goal(pose):
            return 'finished', np.asarray(trajectory), commands
    return 'timeout', np.asarray(trajectory), commands


class GapTrapTests(unittest.TestCase):
    def assert_finished(self, result):
        outcome, poses, commands = result
        self.assertEqual(outcome, 'finished',
                         f'{outcome}; last pose={poses[-1]}; command={commands[-1]}')
        return poses, commands

    def test_opposing_turn_exit_does_not_deadlock(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                self.assert_finished(run_scenario(
                    dogleg(direction), (-3.0, 0.0, 0.0), lambda p: p[0] > 6.0))

    def test_visible_forward_dead_end_chooses_side_exit(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                self.assert_finished(run_scenario(
                    forward_pocket(direction), (-3.0, 0.0, 0.0),
                    lambda p: direction * p[1] > 4.0))

    def test_broad_side_recess_does_not_pull_car_off_through_route(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                poses, _ = self.assert_finished(run_scenario(
                    side_pocket(direction), (-3.0, 0.0, 0.0), lambda p: p[0] > 8.0))
                # The through route remains clear, so taking the recess is
                # unnecessary even though it offers much greater lateral space.
                self.assertLess(float(np.max(np.abs(poses[:, 1]))), 0.7)

    def test_wide_narrow_wide_transition_around_opposing_turns(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                self.assert_finished(run_scenario(
                    changing_width_dogleg(direction), (-4.0, 0.0, 0.0),
                    lambda p: p[0] > 7.0))


if __name__ == '__main__':
    unittest.main()
