# Exported policies

Actor weights, observation normalizer and config for `numpy_policy.py`
(written by `python -m rl.export_policy` in the f1tenth_gym_jax fork). Each
file's embedded `config` records the source checkpoint's update and SHA-256.

## Lab fine-tunes (Nov 7)

CNN+GRU, fine-tuned for 10.5 h from the phase-1c run `20261106_f9b314198a` at update 1025, on lab-size
tracks (the last race's layout, race-like and hand-drawn tracks, speed traps; 0.8-2.6 m wide) with the
car's physics: rollover (~45-50 deg tip), carpet friction 1.1-1.5, 0-240 ms delay, the servo's steering
stops (0.4422 rad) and left/right asymmetry, 5.5-6.5 m/s^2 acceleration, random direction per episode.

| File | Training | Held-out (8 lab tracks, nominal, 1 lap x 10) | Car code in the sim (numpy policy, randomized lab physics, 80 laps) |
|---|---|---|---|
| `gru1025_solo_u0950.npz` (**launch default**) | solo (run `20261107_3be40d477b`, update 950, sha256 `bcb2dbfecc11…`) | 1.00 | 0.89 complete, 3% crash, 9% rollover, 3.28 m/s |
| `gru1025_obs_u0750.npz` | parked-car obstacles (run `20261107_4164028b24`, update 750, sha256 `e178110a8b51…`) | 0.99 | **0.97** complete, 1% crash, 1% rollover, 3.19 m/s |
| `gru1025_opp_u0750.npz` | opponents: past selves, follow-the-gap, racing-line cars, some slow (run `20261107_b00b11b533`, update 750, sha256 `6a5559197d77…`) | 0.96 | 0.76 complete, 9% crash, 11% rollover, 3.16 m/s |
| `gru1025_mixed_u0650.npz` | opponents on half the tracks, obstacles on the rest (run `20261107_0c3afa3f29`, update 650, sha256 `7e0a354efaf9…`) | 0.99 | 0.71 complete, 7% crash, 16% rollover, 2.99 m/s |

The race-trained files take the same inputs as the solo one (their critic-only training inputs are not
exported); all four match the JAX policy within 3e-6 over 1,000 steps. Most failures in the sim are
rollovers at speed: keep the max_speed cap.

## Earlier policies

| File | Source | Held-out eval (24 maps: 6 held-out + 18 never used) |
|---|---|---|
| `policy.npz` | Newton 12 h run `20261105_71ba43c6fd`, update 325 (sha256 `5e965378720d…`) | nominal 1.00 / 1.00; randomized 0.92 / 0.92 (held-out / never used) |
| `policy_u0400.npz` | same run, update 400 (sha256 `58c3dec2b1d2…`) | not evaluated yet |
| `gru_u02050.npz` | **Phase-1c CNN+GRU, final** (run `20261106_f9b314198a`, update 2050 of 2077, sha256 `d81fd791de26…`) | nominal held-out completion 1.00; laps Spielberg 42.7 s, Duct1077 9.1 s, Duct1112 7.9 s |
| `mlp3_u02050.npz` | Phase-1c MLP-3 baseline, final (run `20261106_4ca8863226`, update 2050, sha256 `b33ad87a0e1f…`; 271 beams, 3-frame stack) | nominal held-out completion 1.00; laps Spielberg 48.6 s, Duct1077 10.9 s, Duct1112 9.9 s |
| `gru_u00150.npz` | Phase-1c CNN+GRU run `20261106_f9b314198a`, update 150 (sha256 `4cb839d680f5…`) | nominal held-out completion 0.93 (narrow ducts 0.7-0.9, racetracks 1.0); ROS sim Duct1077: 8.4 laps / 90 s, 0 crashes, 6.5 m/s mean |
| `gru_u00100.npz` | same run, update 100 (sha256 `d7a8995fc13b…`) | nominal held-out completion 1.00; ROS sim Duct1077: 7.8 laps / 90 s, 0 crashes, 6.0 m/s mean |
| `policy_cpu5min.npz` | 5-minute CPU smoke run, for plumbing tests only | — |

The 12 h run was still training when these were exported (2,250 updates
planned); a later checkpoint will replace `policy.npz` after the final
evaluation. Pick a file with `policy_path:=` on `policy.launch.py`.
