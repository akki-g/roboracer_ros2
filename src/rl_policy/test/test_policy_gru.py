"""Car-side checks of the recurrent NumpyPolicy (no ROS, no JAX): a random
cnn_gru file in the export format."""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rl_policy import obs as ob  # noqa: E402
from rl_policy.numpy_policy import NumpyPolicy  # noqa: E402

CONV = [[16, 5, 2], [32, 5, 2], [32, 3, 2], [32, 3, 2]]


@pytest.fixture
def gru_npz(tmp_path):
    rng = np.random.default_rng(0)
    a = {}
    c_in = 1
    for i, (f, k, _) in enumerate(CONV):
        a[f"conv{i}_kernel"] = rng.normal(0, 0.3, (k, c_in, f)).astype(np.float32)
        a[f"conv{i}_bias"] = rng.normal(0, 0.1, f).astype(np.float32)
        c_in = f
    a["proj_kernel"] = rng.normal(0, 0.05, (17 * 32, 128)).astype(np.float32)
    a["proj_bias"] = np.zeros(128, np.float32)
    for g in ("ir", "iz", "in"):
        a[f"gru_{g}_kernel"] = rng.normal(0, 0.1, (131, 128)).astype(np.float32)
        a[f"gru_{g}_bias"] = np.zeros(128, np.float32)
    for g in ("hr", "hz", "hn"):
        a[f"gru_{g}_kernel"] = rng.normal(0, 0.1, (128, 128)).astype(np.float32)
    a["gru_hn_bias"] = np.zeros(128, np.float32)
    a["actor_hidden_kernel"] = rng.normal(0, 0.2, (128, 64)).astype(np.float32)
    a["actor_hidden_bias"] = np.zeros(64, np.float32)
    a["actor_out_kernel"] = rng.normal(0, 0.5, (64, 2)).astype(np.float32)
    a["actor_out_bias"] = np.zeros(2, np.float32)
    a["log_std"] = np.full(2, -0.5, np.float32)
    a["obs_mean"] = np.zeros(ob.OBS_DIM, np.float32)
    a["obs_std"] = np.ones(ob.OBS_DIM, np.float32)
    cfg = {"arch": "cnn_gru", "obs_dim": ob.OBS_DIM, "num_beams": ob.NUM_BEAMS, "frame_stack": 1,
           "beam_indices": ob.CAR_BEAM_INDICES.tolist(), "gru_dim": 128, "conv_layers": CONV, "norm_clip": 10.0,
           "v_cmd_max": ob.V_CMD_MAX, "steer_max": ob.S_MAX, "policy_rate_hz": 25.0}
    path = tmp_path / "gru.npz"
    np.savez(path, config=np.array(json.dumps(cfg)), **a)
    return path


def _obs(rng):
    return ob.build_obs(ob.init_stack(ob.normalize_scan(rng.uniform(0.3, 9, ob.NUM_BEAMS)), 1),
                        rng.uniform(0, 6), rng.uniform(-1, 1, 2))


def test_reset_and_step(gru_npz):
    pol = NumpyPolicy(gru_npz)
    rng = np.random.default_rng(1)
    obs = [_obs(rng) for _ in range(20)]
    first = [pol(o) for o in obs]
    assert np.linalg.norm(pol.h) > 0
    pol.reset()
    assert not pol.h.any()
    again = [pol(o) for o in obs]  # same inputs after a reset: same actions
    np.testing.assert_array_equal(np.array(first), np.array(again))
    pol.reset()
    pol(obs[5])
    assert not np.allclose(pol(obs[0]), first[0])  # history changes the action
    assert all(np.all(np.abs(a) < 1) for a in first)


def test_batch_matches_single_and_mean_is_stateless(gru_npz):
    pol = NumpyPolicy(gru_npz)
    rng = np.random.default_rng(2)
    o = np.stack([_obs(rng) for _ in range(4)])
    h = rng.normal(0, 0.5, (4, 128))
    mean_b, h_b = pol.forward(o, h)
    for i in range(4):
        mean_i, h_i = pol.forward(o[i], h[i])
        np.testing.assert_allclose(mean_b[i], mean_i, atol=1e-12)
        np.testing.assert_allclose(h_b[i], h_i, atol=1e-12)
    pol.reset()
    pol.mean(o[0])
    assert not pol.h.any()  # mean() does not advance the state


def test_bench_runs(gru_npz, capsys):
    from rl_policy.bench_policy import main

    main([str(gru_npz), "--calls", "300"])
    assert "p99" in capsys.readouterr().out
