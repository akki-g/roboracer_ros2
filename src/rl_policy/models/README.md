# Exported policies

Actor weights, observation normalizer and config for `numpy_policy.py`
(written by `python -m rl.export_policy` in the f1tenth_gym_jax fork). Each
file's embedded `config` records the source checkpoint's update and SHA-256.

| File | Source | Held-out eval (24 maps: 6 held-out + 18 never used) |
|---|---|---|
| `policy.npz` (launch default) | Newton 12 h run `20261105_71ba43c6fd`, update 325 (sha256 `5e965378720d…`) | nominal 1.00 / 1.00; randomized 0.92 / 0.92 (held-out / never used) |
| `policy_u0400.npz` | same run, update 400 (sha256 `58c3dec2b1d2…`) | not evaluated yet |
| `gru_u02050.npz` | **Phase-1c CNN+GRU, final** (run `20261106_f9b314198a`, update 2050 of 2077, sha256 `d81fd791de26…`) | nominal held-out completion 1.00; laps Spielberg 42.7 s, Duct1077 9.1 s, Duct1112 7.9 s |
| `mlp3_u02050.npz` | Phase-1c MLP-3 baseline, final (run `20261106_4ca8863226`, update 2050, sha256 `b33ad87a0e1f…`; 271 beams, 3-frame stack) | nominal held-out completion 1.00; laps Spielberg 48.6 s, Duct1077 10.9 s, Duct1112 9.9 s |
| `gru_u00150.npz` | Phase-1c CNN+GRU run `20261106_f9b314198a`, update 150 (sha256 `4cb839d680f5…`) | nominal held-out completion 0.93 (narrow ducts 0.7-0.9, racetracks 1.0); ROS sim Duct1077: 8.4 laps / 90 s, 0 crashes, 6.5 m/s mean |
| `gru_u00100.npz` | same run, update 100 (sha256 `d7a8995fc13b…`) | nominal held-out completion 1.00; ROS sim Duct1077: 7.8 laps / 90 s, 0 crashes, 6.0 m/s mean |
| `policy_cpu5min.npz` | 5-minute CPU smoke run, for plumbing tests only | — |

The 12 h run was still training when these were exported (2,250 updates
planned); a later checkpoint will replace `policy.npz` after the final
evaluation. Pick a file with `policy_path:=` on `policy.launch.py`.
