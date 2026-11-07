"""The sim-to-car drive relay's packet format and command gate (no ROS needed)."""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rl_policy.drive_relay import decode, encode, relay_command  # noqa: E402


def test_round_trip_and_bad_packets():
    assert decode(encode(7, 3.25, -0.2)) == (7, 3.25, -0.2)
    assert decode(b"short") is None
    assert decode(encode(1, math.nan, 0.0)) is None


def test_gate():
    cmd = (4.0, 0.3)
    assert relay_command(cmd, 0.05, 0.3, True, 10.0, 0.4422) == (4.0, 0.3)
    assert relay_command(cmd, 0.5, 0.3, True, 10.0, 0.4422) == (0.0, 0.0)  # stale: stop
    assert relay_command(cmd, 0.05, 0.3, False, 10.0, 0.4422) == (0.0, 0.0)  # deadman released: stop
    assert relay_command(None, 0.0, 0.3, True, 10.0, 0.4422) == (0.0, 0.0)  # nothing received yet
    assert relay_command((12.0, 0.9), 0.05, 0.3, True, 10.0, 0.4422) == (10.0, 0.4422)  # clamped
    assert relay_command((-1.0, -0.9), 0.05, 0.3, True, 10.0, 0.4422) == (0.0, -0.4422)
