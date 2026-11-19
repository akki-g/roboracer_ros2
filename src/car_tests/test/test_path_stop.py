import math
from dataclasses import replace

import pytest

from car_tests.core import Engine, Params, arc_point, distance_to_arc, path_blocked, stop_distance
from test_core import armed, fresh


def scene(points=(), missing=(), p=Params()):
    amin, inc = -3*math.pi/4, math.pi/720
    ranges = [10.] * 1081
    for x, y, beams in points:
        x -= p.lidar_offset
        r, angle = math.hypot(x, y), math.atan2(y, x)
        index = round((angle-amin)/inc)
        for j in range(index-beams//2, index-beams//2+beams):
            ranges[j] = r
    for j in missing:
        ranges[j] = float('nan')
    return ranges, amin, inc, .02, 10.


@pytest.mark.parametrize('sign', [-1, 1])
def test_obstacle_on_arc_and_outside_straight_cone(sign):
    p = Params()
    k = math.tan(sign*.3)/p.wheelbase
    x, y = arc_point(1.3, k)
    assert abs(math.atan2(y, x-p.lidar_offset)) > math.radians(p.cone_deg)
    assert path_blocked(scene([(x, y, 5)]), 2., sign*.3, p)
    assert not path_blocked(scene([(x, y, 5)]), 2., -sign*.3, p)


@pytest.mark.parametrize('sign', [-1, 1])
def test_forward_object_outside_turn_does_not_stop(sign):
    p = Params()
    assert path_blocked(scene([(1.6, 0., 5)]), 2., 0., p)
    assert path_blocked(scene([(1.6, 0., 5)]), 2., sign*.3, p) is False


def test_noise_rules_and_dropout_fail_closed():
    p = Params()
    x, y = arc_point(1.1, math.tan(.3)/p.wheelbase)
    assert path_blocked(scene([(x, y, 1)]), 2., .3, p) is False
    assert path_blocked(scene([(x, y, 3)]), 2., .3, p) is True
    ranges, *geometry = scene()
    ranges[539:542] = [.03]*3
    assert path_blocked((ranges, *geometry), 2., .3, p) is False
    assert path_blocked(scene(missing=range(700, 850)), 2., .3, p) is None
    assert path_blocked(scene(missing=range(1081)), 2., .3, p) is None


def test_path_visibility_ignores_short_invalid_sentinel_streak():
    ranges, *geometry = scene()
    ranges[745:748] = [65.533]*3  # real Hokuyo out-of-range sentinel, not an obstacle
    assert path_blocked((ranges, *geometry), 2., .3, Params()) is False


def test_finite_arc_endpoints_width_and_offset():
    p = Params()
    length = stop_distance(2, p)
    assert distance_to_arc(2, .4, 0, 1) == pytest.approx(math.hypot(1, .4))
    assert path_blocked(scene([(length+.6, 0, 5)]), 2., 0., p) is False
    assert path_blocked(scene([(1., .29, 5)]), 2., 0., p) is True
    assert path_blocked(scene([(1., .35, 5)]), 2., 0., p) is False
    assert path_blocked(scene([(1., .35, 5)]), 2., 0., replace(p, side_margin=.25)) is True
    # Same raw scan, shifted rear-axle geometry changes finite-arc endpoint.
    s = scene([(length+.2, 0, 5)])
    assert path_blocked(s, 2., 0., p) is True
    assert path_blocked(s, 2., 0., replace(p, lidar_offset=.7)) is False


def test_engine_grip_path_and_clearance_and_transition():
    e = armed('grip')
    e.tick(1.02)
    fresh(e, 1.04, speed=2.)
    e.scan(*scene([(1.6, 0., 5)]), 1.04)
    assert e._health(1.04, 2., .3) == ''
    x, y = arc_point(1.3, math.tan(.3)/e.p.wheelbase)
    e.scan(*scene([(x, y, 5)]), 1.04)
    assert e._health(1.04, 2., .3) == 'path stopping distance'
    e.scan(*scene([(1.6, 0., 5)]), 1.04)
    assert e._health(1.04, 2., 0.) == 'path stopping distance'
    e.near = .2
    assert e._health(1.04, 2., .3) == 'minimum clearance'


def test_brake_transition_keeps_previous_arc_for_health_check():
    e = armed('grip')
    e.tick(1.02)
    fresh(e, 1.04, speed=2.)
    x, y = arc_point(1.3, math.tan(.3)/e.p.wheelbase)
    e.scan(*scene([(x, y, 5)]), 1.04)
    e._brake(1.04, 'speed cap')
    assert e.last_output == (0., 0.)
    assert e._health(1.04, 0., 0.) == 'path stopping distance'


@pytest.mark.parametrize('change', [dict(car_half_width=0), dict(side_margin=-.1), dict(side_margin=float('nan'))])
def test_path_parameters_validated(change):
    with pytest.raises(ValueError):
        Engine('grip', replace(Params(), **change))


def _reference(scan, speed, steering, p):
    """The original pure-Python path_blocked (per-beam loop and per-point visibility), for equivalence."""
    from car_tests.core import _cluster_min
    ranges, amin, inc, rmin, rmax = scan
    valid = [math.isfinite(r) and max(rmin, p.short_reject) <= r <= rmax for r in ranges]
    if not any(valid):
        return None
    curvature = math.tan(steering)/p.wheelbase
    length, width = stop_distance(speed, p), p.car_half_width+p.side_margin
    hits = []
    for i, r in enumerate(ranges):
        a = amin+i*inc
        d = distance_to_arc(r*math.cos(a)+p.lidar_offset, r*math.sin(a), curvature, length) if valid[i] else None
        hits.append(d if d is not None and d <= width else None)
    if _cluster_min(hits, p.cluster_beams) is not None:
        return True
    count = max(2, math.ceil(length/.10))
    for j in range(1, count+1):
        s = length*j/count
        if s < p.lidar_offset+p.min_clearance:
            continue
        x, y = arc_point(s, curvature)
        for lateral_offset in (-width, 0., width):
            xx = x-lateral_offset*math.sin(curvature*s)-p.lidar_offset
            yy = y+lateral_offset*math.cos(curvature*s)
            index = round((math.atan2(yy, xx)-amin)/inc)
            half = p.cluster_beams//2
            lo, hi = index-2*p.cluster_beams, index+2*p.cluster_beams+1
            if index-half < 0 or index+half >= len(valid):
                return None
            nearby = [1. if value else None for value in valid[max(0, lo):min(len(valid), hi)]]
            if _cluster_min(nearby, p.cluster_beams) is None:
                return None
    return False


@pytest.mark.parametrize('seed', range(12))
def test_vectorised_path_blocked_matches_reference(seed):
    # Random scans (open, cluttered, with dropout sectors and sentinels), both turn directions,
    # straight, a range of speeds: the NumPy version must return exactly what the original did.
    import random
    rng = random.Random(seed)
    p = Params()
    outcomes = set()
    for _ in range(25):
        far = rng.random() < .5
        ranges = [rng.uniform(2.5, 9.) if far else rng.uniform(.3, 4.) for _ in range(1081)]
        for j in rng.sample(range(1081), 25):
            ranges[j] = rng.choice([.02, 65.533, float('inf'), float('nan')])
        if rng.random() < .4:
            start = rng.randrange(0, 1000)
            for j in range(start, min(1081, start+rng.randrange(5, 120))):
                ranges[j] = float('nan')
        scan = (ranges, -3*math.pi/4, math.pi/720, .02, 30.)
        speed, steering = rng.uniform(.3, 3.5), rng.choice([0., .3, -.3, rng.uniform(-.42, .42)])
        want = _reference(scan, speed, steering, p)
        assert path_blocked(scan, speed, steering, p) == want
        outcomes.add(want)
    assert len(outcomes) >= 2  # the scans exercise more than one outcome
