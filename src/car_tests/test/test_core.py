import json
import math
from dataclasses import replace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from car_tests.core import Engine, Params, lateral, scan_distances, steer_schedule, stop_distance


def fresh(e, t, speed=0, distance=30):
    e.scan([distance] * 181, -math.pi / 2, math.pi / 180, 0.02, 50, t)
    e.sensor(speed * e.p.erpm_gain, 0, t)
    e.odom(speed, t)
    e.joy_t = t


def armed(test="straight", p=Params()):
    e = Engine(test, p)
    fresh(e, 1)
    e.joy([0] * 6, 1)
    e.joy([0, 0, 0, 0, 0, 1], 1.01)
    return e


def test_stop_math_and_offset_not_free_space():
    p = Params()
    assert stop_distance(3, p) == 2.75
    assert stop_distance(-3, p) == 2.75
    assert stop_distance(3, replace(p, lidar_offset=1)) == 2.75


def test_scan_uses_angles_and_clustered_clearance():
    p = Params()
    ranges = [10.] * 101
    ranges[19:22] = [0.2, 0.21, 0.2]  # rear/side obstacle spanning 3 beams
    ranges[79:82] = [2.0, 2.0, 2.0]  # forward at zero, not array midpoint
    forward, near = scan_distances(ranges, -0.8, 0.01, .05, 20, p)
    assert forward == 2 and near == .21
    assert scan_distances([float("nan")] * 10, -.5, .1, .05, 20, p) == (None, None)


def test_scan_ignores_isolated_and_below_minimum_readings():
    # The UST-10LX gives isolated spurious short readings (about 3 per scan on the car):
    # one beam, or readings under short_reject, are not an obstacle.
    p = Params()
    ranges = [10.] * 101
    ranges[20] = 0.1  # single-beam spike
    ranges[40:43] = [0.03, 0.02, 0.04]  # below short_reject
    ranges[80] = 0.3  # single forward spike
    forward, near = scan_distances(ranges, -0.8, 0.01, .02, 20, p)
    assert forward == 10 and near == 10


def test_policy_sweep_skips_wall_distance_stops():
    e = Engine("policy-sweep")
    fresh(e, 1, distance=0.3)  # ducts closer than min_clearance and the stopping distance
    e.joy([0] * 6, 1)
    e.joy([0, 0, 0, 0, 0, 1], 1.01)
    e.policy(1.5, 0.0, 1.01)
    assert e.tick(1.01) == (1.5, 0.0) and e.phase == "policy"


def test_requires_release_then_rb_and_latches_release():
    e = Engine("grip")
    fresh(e, 0)
    e.joy([0, 0, 0, 0, 0, 1], 0)
    assert e.tick(0) == (0, 0)
    e.joy([0] * 6, 0)
    e.joy([0, 0, 0, 0, 0, 1], .01)
    assert e.tick(.01)[0] == pytest.approx(1)
    e.joy([1, 0, 0, 0, 0, 0], .02)  # other button cannot bypass
    assert e.tick(.02) == (0, 0) and e.reason == "RB released"
    e.joy([0, 0, 0, 0, 0, 1], .03)
    assert e.tick(.03) == (0, 0)


def test_lb_and_rb_together_cannot_start():
    e = Engine("grip")
    fresh(e, 1)
    e.joy([0]*6, 1)
    e.joy([0, 0, 0, 0, 1, 1], 1.01)
    assert e.tick(1.01) == (0, 0) and e.reason == 'LB override'


@pytest.mark.parametrize("topic", ["scan_t", "sensor_t", "joy_t", "odom_t"])
def test_stale_stops(topic):
    e = armed()
    fresh(e, 1.3)
    setattr(e, topic, 1)
    assert e.tick(1.3) == (0, 0)
    assert e.reason.startswith("stale")


def test_forward_stop_uses_requested_and_actual_speed():
    e = armed(p=replace(Params(), targets=(3.,)))
    fresh(e, 1.02, distance=2.7)
    assert e.tick(1.02) == (0, 0)
    assert e.reason == "forward stopping distance"
    e = armed()
    fresh(e, 1.02, speed=3, distance=2.8)
    assert e.tick(1.02) == (0, 0)  # conservative ERPM offset allowance


def test_unreachable_target_refused_before_motion():
    e = Engine("straight", replace(Params(), targets=(3.,)))
    fresh(e, 1, distance=5)
    e.joy([0] * 6, 1)
    e.joy([0, 0, 0, 0, 0, 1], 1.01)
    assert e.tick(1.01) == (0, 0)
    assert "unreachable" in e.reason


def test_skip_lateral_combos_both_directions():
    p = Params()
    steps, skipped = steer_schedule(2, p)
    assert skipped == [.42]
    assert {d for _, _, d in steps} == {0, .1, -.1, .2, -.2, .3, -.3}
    assert all(lateral(2, d) <= 5 for _, _, d in steps)


@pytest.mark.parametrize("change", [dict(rate_hz=0), dict(max_speed=float("nan")), dict(steer_limit=.43),
                                   dict(stale_s=.3), dict(a_brake=-1), dict(margin=0), dict(cone_deg=91),
                                   dict(targets=()), dict(targets=(4.,)), dict(targets=(float("inf"),)),
                                   dict(hold_s=-1)])
def test_invalid_params(change):
    with pytest.raises(ValueError):
        Engine("straight", replace(Params(), **change))


def test_grip_cap_and_right_sign():
    e = armed("grip", replace(Params(), direction="right", ramp=1.))
    assert e.tick(1.02)[1] == -.3
    fresh(e, 3)
    assert e.tick(3) == (0, 0)
    assert e.phase == "brake" and e.reason == "lateral acceleration cap"


def test_zero_on_exit_and_terminal_tick():
    e = armed()
    assert e.tick(1.02)[0] > 0
    commands = e.exit_commands("exception", 1.03)
    assert len(commands) / e.p.rate_hz >= .5
    assert set(commands) == {(0, 0)}
    assert e.tick(1.04) == (0, 0)
    json.dumps(e.events, allow_nan=False)


def test_one_target_per_press_normal_completion():
    e = armed(p=replace(Params(), hold_s=.1, targets=(1., 1.5)))
    fresh(e, 1.02, speed=1)
    assert e.tick(1.02)[0] == 1
    fresh(e, 1.2, speed=1)
    assert e.tick(1.2) == (0, 0)
    for t in (1.3, 1.5, 1.7, 1.9):
        fresh(e, t)
        e.tick(t)
    assert e.phase == "between"
    fresh(e, 2)
    e.joy([0, 0, 0, 0, 0, 1], 2)
    assert e.phase == "between"
    e.joy([0]*6, 2)
    e.joy([0, 0, 0, 0, 0, 1], 2.01)
    assert e.run == 1 and e.tick(2.01)[0] == 1.5


def test_policy_guard_caps_and_stale():
    e = armed("policy-sweep", replace(Params(), max_speed=2.))
    e.policy(5, .8, 1.02)
    v, d = e.tick(1.02)
    assert v == 2 and d <= .42 and lateral(v, d) <= 5 + 1e-9
    fresh(e, 1.3)
    assert e.tick(1.3) == (0, 0) and e.reason == "stale policy command"


@pytest.mark.parametrize("cause", ["near", "fault", "nan", "duration", "LB"])
def test_other_faults(cause):
    e = armed()
    if cause == "near": e.near = .1
    if cause == "fault": e.fault = 4
    if cause == "nan": e.odom_speed = float("nan")
    if cause == "duration": e.started = -200
    if cause == "LB": e.joy([0, 0, 0, 0, 1, 1], 1.02)
    assert e.tick(1.02) == (0, 0) and e.phase == "aborted"
