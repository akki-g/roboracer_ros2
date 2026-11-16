"""The board test's replay check and timing on this machine (no ROS)."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rl_policy import board_test as bt  # noqa: E402
from rl_policy.numpy_policy import NumpyPolicy  # noqa: E402
from rl_policy.safety import SafetyConfig  # noqa: E402


def test_replay_file():
    rep = bt.load_replay()
    t = len(rep["speed"])
    assert rep["scans"].shape == (t, 1081) and rep["scans"].dtype == np.float32
    assert rep["jetson_action"].shape == (t, 2) and rep["prev_action"].shape == (t, 2)
    assert rep["reset"][0]  # the recording starts at an episode start, so the GRU state is known
    assert np.all(np.abs(rep["jetson_action"]) < 1) and rep["speed"].max() > 2.0  # a real drive


def test_same_actions_as_the_jetson():
    rep = bt.load_replay()
    acts = bt.replay_actions(NumpyPolicy(bt.find_share("models", rep["policy"])), rep)
    assert np.abs(acts - rep["jetson_action"]).max() < bt.ACTION_TOL
    other = bt.replay_actions(NumpyPolicy(bt.find_share("models", "gru1025_obs_u0750.npz")), rep)
    assert np.abs(other - rep["jetson_action"]).max() > 0.1  # the check tells policies apart


def test_timing_and_main(capsys):
    rep = bt.load_replay()
    pol = NumpyPolicy(bt.find_share("models", rep["policy"]))
    t_scan, t_fwd, t_tick = bt.time_ticks(pol, rep, 600, SafetyConfig(max_speed=10.0, ttc_half_angle=0.0))
    assert len(t_tick) == 600 and np.all(t_tick >= t_fwd) and np.all(t_scan > 0)
    assert bt.main(["--calls", "300"]) == 0
    out = capsys.readouterr().out
    assert "same actions as the Jetson" in out and "board test: PASS" in out
