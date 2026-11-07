"""Safety layer (spec 7.3) - plain pytest, no ROS needed:
    python3 -m pytest src/rl_policy/test
"""

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rl_policy import obs as ob  # noqa: E402
from rl_policy.safety import (  # noqa: E402
    DEFAULT_MAX_SPEED,
    STEER_LIMIT,
    SafetyConfig,
    apply_safety,
    min_ttc,
)

FRESH = dict(scan_age=0.02, odom_age=0.02, ttc=math.inf)
ANGLES = -ob.FOV / 2 + np.arange(1081) * (ob.FOV / 1080)


def test_default_max_speed_is_1_5():
    assert DEFAULT_MAX_SPEED == 1.5 and SafetyConfig().max_speed == 1.5


def test_speed_cap():
    r = apply_safety(6.0, 0.1, SafetyConfig(), **FRESH)
    assert r.speed == 1.5 and "speed_cap" in r.overrides
    r = apply_safety(1.0, 0.1, SafetyConfig(), **FRESH)
    assert r.speed == 1.0 and r.overrides == []
    r = apply_safety(-2.0, 0.0, SafetyConfig(), **FRESH)  # never reverse
    assert r.speed == 0.0


def test_steering_clamp():
    r = apply_safety(1.0, 0.9, SafetyConfig(), **FRESH)
    assert r.steer == pytest.approx(STEER_LIMIT) and "steer_clamp" in r.overrides
    r = apply_safety(1.0, -0.9, SafetyConfig(), **FRESH)
    assert r.steer == pytest.approx(-STEER_LIMIT)
    with pytest.raises(ValueError):
        SafetyConfig(steer_limit=0.6).validate()


def test_emergency_brake_ttc():
    ranges = np.full(1081, 10.0)
    ranges[540] = 0.6  # straight ahead
    assert min_ttc(ranges, ANGLES, 1.5) == pytest.approx(0.4)
    r = apply_safety(1.5, 0.0, SafetyConfig(), scan_age=0.02, odom_age=0.02, ttc=min_ttc(ranges, ANGLES, 1.5))
    assert r.speed == 0.0 and "emergency_brake" in r.overrides
    # same obstacle at 1.0 m/s: TTC 0.6 s, no brake
    assert min_ttc(ranges, ANGLES, 1.0) == pytest.approx(0.6)
    # obstacle at 60 deg is outside the +-30 deg cone
    ranges = np.full(1081, 10.0)
    ranges[np.argmin(np.abs(ANGLES - math.radians(60)))] = 0.2
    assert min_ttc(ranges, ANGLES, 2.0) > 1.0
    # at 25 deg the closing speed is v cos(theta)
    ranges = np.full(1081, 10.0)
    i = np.argmin(np.abs(ANGLES - math.radians(25)))
    ranges[i] = 0.5
    assert min_ttc(ranges, ANGLES, 1.0) == pytest.approx(0.5 / math.cos(ANGLES[i]))


def test_ttc_brake_switch():
    """ttc_brake=False never brakes on TTC; the other checks still apply."""
    off = SafetyConfig(ttc_brake=False)
    off.validate()
    r = apply_safety(1.0, 0.0, off, scan_age=0.0, odom_age=0.0, ttc=0.05)
    assert r.speed == 1.0 and "emergency_brake" not in r.overrides
    r = apply_safety(1.0, 0.0, off, scan_age=1.0, odom_age=0.0, ttc=0.05)
    assert r.speed == 0.0 and "watchdog_scan" in r.overrides
    on = apply_safety(1.0, 0.0, SafetyConfig(), scan_age=0.0, odom_age=0.0, ttc=0.05)
    assert on.speed == 0.0 and "emergency_brake" in on.overrides


def test_ttc_ignores_invalid_returns_and_handles_standstill():
    ranges = np.full(1081, np.inf)
    ranges[530:550] = np.nan
    assert math.isinf(min_ttc(ranges, ANGLES, 2.0))
    ranges = np.full(1081, 10.0)
    ranges[540] = 0.001  # below range_min: a bad return, not an obstacle
    assert min_ttc(ranges, ANGLES, 2.0, range_min=0.02) == pytest.approx(5.0, rel=1e-3)
    ranges[540] = 0.003  # stopped car: closing speed floors at 0.01 m/s
    assert min_ttc(ranges, ANGLES, 0.0) == pytest.approx(0.3)


def test_watchdog():
    cfg = SafetyConfig()
    r = apply_safety(1.0, 0.1, cfg, scan_age=0.15, odom_age=0.02, ttc=math.inf)
    assert r.speed == 0.0 and "watchdog_scan" in r.overrides
    r = apply_safety(1.0, 0.1, cfg, scan_age=0.05, odom_age=0.25, ttc=math.inf)
    assert r.speed == 0.0 and "watchdog_odom" in r.overrides
    r = apply_safety(1.0, 0.1, cfg, scan_age=math.inf, odom_age=math.inf, ttc=math.inf)
    assert r.speed == 0.0
    r = apply_safety(1.0, 0.1, cfg, scan_age=0.099, odom_age=0.199, ttc=math.inf)
    assert r.speed == 1.0


def test_invalid_input_and_non_finite():
    r = apply_safety(1.0, 0.1, SafetyConfig(), valid=False, **FRESH)
    assert r.speed == 0.0 and "invalid_input" in r.overrides
    r = apply_safety(float("nan"), 0.1, SafetyConfig(), **FRESH)
    assert r.speed == 0.0 and r.steer == 0.0


def test_order_brake_wins_over_cap():
    r = apply_safety(8.0, 1.0, SafetyConfig(), scan_age=0.02, odom_age=0.02, ttc=0.1)
    assert r.speed == 0.0 and r.overrides == ["speed_cap", "steer_clamp", "emergency_brake"]


def test_executed_action_round_trip():
    """The node feeds the published command back as the previous action."""
    a = ob.command_to_action(1.5, 0.2)
    assert a[0] == pytest.approx(2 * 1.5 / 8.0 - 1) and a[1] == pytest.approx(0.2 / ob.S_MAX)
    np.testing.assert_allclose(ob.command_to_action(0.0, 0.0), ob.RESET_ACTION)


def test_map_speed_knee():
    from rl_policy.safety import map_speed

    cfg = SafetyConfig(max_speed=3.0, scale_speed=True, scale_knee=0.7)  # knee 2.1 m/s
    for v, exp in ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (2.1, 2.1), (8.0, 3.0)):
        assert map_speed(v, cfg, 8.0) == pytest.approx(exp)
    assert map_speed(5.0, cfg, 8.0) == pytest.approx(2.1 + 2.9 * 0.9 / 5.9)
    vs = [map_speed(v, cfg, 8.0) for v in np.linspace(0, 8, 81)]
    assert np.all(np.diff(vs) >= 0) and max(vs) == pytest.approx(3.0)  # monotone, reaches the cap
    for v in np.linspace(0, 8, 17):  # never over the cap: the clamp never fires
        r = apply_safety(map_speed(v, cfg, 8.0), 0.0, cfg, **FRESH)
        assert "speed_cap" not in r.overrides
    assert map_speed(5.0, SafetyConfig(max_speed=3.0), 8.0) == 5.0  # off: unchanged (then clamped)
    assert map_speed(5.0, SafetyConfig(max_speed=10.0, scale_speed=True), 8.0) == 5.0  # cap above the range
    assert map_speed(4.0, SafetyConfig(max_speed=3.0, scale_speed=True, scale_knee=0.0), 8.0) == pytest.approx(1.5)  # linear


def test_speed_gain_and_min_speed():
    from rl_policy.safety import MIN_SPEED_FROM, shape_speed

    off = SafetyConfig(max_speed=10.0)
    assert shape_speed(3.0, off, 8.0) == 3.0 and shape_speed(0.2, off, 8.0) == 0.2  # both off by default
    gain = SafetyConfig(max_speed=10.0, speed_gain=1.2)
    assert shape_speed(3.0, gain, 8.0) == pytest.approx(3.6)
    floor = SafetyConfig(max_speed=10.0, min_speed=0.5)
    assert shape_speed(0.2, floor, 8.0) == 0.5  # a crawl is lifted to the floor
    assert shape_speed(0.0, floor, 8.0) == 0.0 and shape_speed(MIN_SPEED_FROM, floor, 8.0) == MIN_SPEED_FROM  # stop stays
    assert shape_speed(2.0, floor, 8.0) == 2.0  # faster requests untouched
    both = SafetyConfig(max_speed=4.0, speed_gain=1.5, min_speed=0.5)
    r = apply_safety(shape_speed(3.0, both, 8.0), 0.0, both, scan_age=0.0, odom_age=0.0, ttc=math.inf)
    assert r.speed == 4.0 and "speed_cap" in r.overrides  # the cap still wins after the gain


def test_min_speed_never_beats_the_brakes():
    from rl_policy.safety import shape_speed

    cfg = SafetyConfig(max_speed=10.0, min_speed=0.6, ttc_brake=True)
    s = shape_speed(0.2, cfg, 8.0)
    assert apply_safety(s, 0.0, cfg, scan_age=0.0, odom_age=0.0, ttc=0.1).speed == 0.0  # emergency brake
    assert apply_safety(s, 0.0, cfg, scan_age=1.0, odom_age=0.0, ttc=math.inf).speed == 0.0  # watchdog


def test_gain_and_floor_validation():
    for bad in (dict(speed_gain=0.2), dict(speed_gain=3.0), dict(min_speed=-0.1), dict(min_speed=5.0)):
        with pytest.raises(ValueError):
            SafetyConfig(**bad).validate()
    SafetyConfig(speed_gain=1.3, min_speed=0.5).validate()


def test_deadman_held():
    from rl_policy.safety import deadman_held

    assert deadman_held(-1, [], math.inf, 0.5)  # off
    rb = [0, 0, 0, 0, 0, 1, 0]
    assert deadman_held(5, rb, 0.05, 0.5)
    assert not deadman_held(5, [0] * 7, 0.05, 0.5)  # released
    assert not deadman_held(5, rb, 0.8, 0.5)  # stale /joy counts as released
    assert not deadman_held(5, [], math.inf, 0.5)  # no joystick yet
    assert not deadman_held(9, rb, 0.05, 0.5)  # button index past the controller's buttons


def test_speed_smoothing():
    from rl_policy.safety import request_speed, shape_speed

    cfg = SafetyConfig(max_speed=10.0, speed_smoothing=0.5)
    assert request_speed(4.0, cfg, 8.0) == 4.0  # no previous request: unchanged
    assert request_speed(0.0, cfg, 8.0, prev=4.0) == 2.0  # a brake pulse is halved
    assert request_speed(4.0, cfg, 8.0, prev=2.0) == 3.0
    off = SafetyConfig(max_speed=10.0)
    assert request_speed(0.0, off, 8.0, prev=4.0) == 0.0  # off by default
    floor = SafetyConfig(max_speed=10.0, speed_smoothing=0.5, min_speed=0.5)
    assert shape_speed(0.0, floor, 8.0, prev=0.4) == pytest.approx(0.2)  # the policy stops: no floor, the average slows
    assert shape_speed(0.4, floor, 8.0, prev=0.2) == 0.5
    with pytest.raises(ValueError):
        SafetyConfig(speed_smoothing=0.0).validate()
