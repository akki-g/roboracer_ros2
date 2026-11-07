"""The sim opponent's control logic (no ROS needed)."""

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rl_policy.opponent_pursuit import SpeedSchedule, Track, command, corner_speed, parse_speed  # noqa: E402

MAPS = pathlib.Path(__file__).resolve().parents[3] / "sim" / "maps"


def circle(r=3.0, n=600):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Track(np.c_[r * np.cos(a), r * np.sin(a)])


def test_parse_speed():
    assert parse_speed("2.0") == (2.0, 2.0)
    assert parse_speed("1.0-4.0") == (1.0, 4.0)
    assert parse_speed("4:1") == (1.0, 4.0)
    with pytest.raises(ValueError):
        parse_speed("1-2-3")


def test_schedule_fixed_and_varying():
    fixed = SpeedSchedule(1.5, 1.5)
    assert all(fixed(t) == 1.5 for t in range(0, 60, 5))
    vary = SpeedSchedule(1.0, 4.0, period=6.0, seed=3)
    seen = {round(vary(t), 3) for t in np.arange(0, 60, 0.5)}
    assert len(seen) >= 5 and all(1.0 <= v <= 4.0 for v in seen)


def test_on_a_circle_steers_left_and_slows_for_the_bend():
    track = circle(3.0)
    # on the circle at angle 0 heading counter-clockwise: pure pursuit turns left, ~atan(L / r)
    speed, steer, target = command(track, SpeedSchedule(10.0, 10.0), 0.0, 3.0, 0.0, math.pi / 2, 2.0, a_lat=4.0)
    assert 0.05 < steer < 0.2
    assert target == 10.0 and speed == pytest.approx(math.sqrt(4.0 * 3.0), rel=0.05)
    assert corner_speed(track, 0, 4.0) == pytest.approx(math.sqrt(12.0), rel=0.05)


@pytest.mark.skipif(not (MAPS / "LastRace2_centerline.csv").exists(), reason="no LastRace2 sim map")
def test_last_race_start_poses():
    track = Track.from_csv(MAPS / "LastRace2_centerline.csv")
    x0, y0, _ = track.pose_ahead(0.0)
    x, y, _ = track.pose_ahead(4.0)
    assert 3.0 < math.hypot(x - x0, y - y0) <= 4.05  # 4 m along the track, so at most 4 m apart
