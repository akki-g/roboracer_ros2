# Exported policies

Actor weights, observation normalizer and config for `numpy_policy.py`
(written by `python -m rl.export_policy` in the f1tenth_gym_jax fork). Each
file's embedded `config` records the source checkpoint's update and SHA-256.

| File | Source | Held-out eval (24 maps: 6 held-out + 18 never used) |
|---|---|---|
| `policy.npz` (launch default) | Newton 12 h run `20261105_71ba43c6fd`, update 325 (sha256 `5e965378720d…`) | nominal 1.00 / 1.00; randomized 0.92 / 0.92 (held-out / never used) |
| `policy_u0400.npz` | same run, update 400 (sha256 `58c3dec2b1d2…`) | not evaluated yet |
| `policy_cpu5min.npz` | 5-minute CPU smoke run, for plumbing tests only | — |

The 12 h run was still training when these were exported (2,250 updates
planned); a later checkpoint will replace `policy.npz` after the final
evaluation. Pick a file with `policy_path:=` on `policy.launch.py`.
