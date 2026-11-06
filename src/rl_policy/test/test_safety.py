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
    assert a[0] == pytest.approx(2 * 1.5 / 8.0 - 1) and a[1] == pytest.approx(0.2 / 0.4189)
    np.testing.assert_allclose(ob.command_to_action(0.0, 0.0), ob.RESET_ACTION)


def test_speed_scale():
    from rl_policy.safety import speed_scale

    cfg = SafetyConfig(max_speed=3.0, scale_speed=True)
    s = speed_scale(cfg, 8.0)
    assert s == pytest.approx(3.0 / 8.0)
    # the policy's whole range lands inside the cap, in proportion: the clamp never fires
    for v in (0.0, 2.0, 5.0, 8.0):
        r = apply_safety(v * s, 0.0, cfg, **FRESH)
        assert "speed_cap" not in r.overrides and r.speed == pytest.approx(v * 3.0 / 8.0)
    # previous action, as the node computes it: the policy's own request
    r = apply_safety(6.0 * s, 0.1, cfg, **FRESH)
    np.testing.assert_allclose(ob.command_to_action(r.speed / s, r.steer, 8.0), ob.command_to_action(6.0, 0.1, 8.0))
    # a safety stop still shows as a stop
    r = apply_safety(6.0 * s, 0.1, cfg, scan_age=0.5, odom_age=0.02, ttc=math.inf)
    assert r.speed == 0.0 and ob.command_to_action(r.speed / s, r.steer, 8.0)[0] == -1.0
    assert speed_scale(SafetyConfig(max_speed=3.0), 8.0) == 1.0  # off: clamp
    assert speed_scale(SafetyConfig(max_speed=10.0, scale_speed=True), 8.0) == 1.0  # cap above the range
    assert speed_scale(SafetyConfig(max_speed=0.0, scale_speed=True), 8.0) == 1.0  # zero cap: clamp to 0
