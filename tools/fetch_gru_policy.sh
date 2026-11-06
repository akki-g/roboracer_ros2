#!/bin/bash
# Fetch the best recent phase-1c GRU checkpoint from Newton and export it for
# the car. Run on the Mac (needs `ssh -fN newton` and ../f1tenth_gym_jax):
#   tools/fetch_gru_policy.sh [run name] [update]
# Picks, among the run's held-out evals (every 50 updates, nominal settings),
# those with completion 1.00, the one with the lowest mean lap time; or the
# given update. Writes src/rl_policy/models/gru_u<update>.npz and times it.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-20261106_f9b314198a}"
REMOTE="carvel/f1tenth_gym_jax/runs/phase1c/$RUN"
FORK=../f1tenth_gym_jax
ssh -O check newton 2>/dev/null || { echo "no shared Newton connection: run  ssh -fN newton  first" >&2; exit 1; }
if [ -n "${2:-}" ]; then
  U=$(printf "%05d" "$2")
else
  U=$(ssh -o BatchMode=yes newton "cd $REMOTE && python3 -" <<'PY'
import glob, json
best = None
for f in sorted(glob.glob("eval/update_*.json")):
    j = json.load(open(f))
    times = [v["lap_time_mean"] for v in j["per_map"].values()]
    if j["completion_rate"] < 1.0 or None in times:
        continue
    key = sum(times) / len(times)
    print(f"update {j['update']:5d}: completion 1.00, mean lap {key:.2f} s", file=__import__("sys").stderr)
    if best is None or key < best[0]:
        best = (key, j["update"])
if best is None:
    raise SystemExit("no eval with completion 1.00 yet")
print(f"{best[1]:05d}")
PY
  )
fi
echo "using update $U"
DEST="$FORK/runs/phase1c_newton/$RUN"
mkdir -p "$DEST"
scp -q -o BatchMode=yes "newton:$REMOTE/ckpt_$U.safetensors" "newton:$REMOTE/config.json" "$DEST/"
(cd "$FORK" && JAX_PLATFORMS=cpu uv run --frozen python -m rl.export_policy "runs/phase1c_newton/$RUN/ckpt_$U.safetensors" \
  --out "../roboracer_ros2/src/rl_policy/models/gru_u$U.npz")
"$FORK/.venv/bin/python" -c "import sys; sys.path.insert(0, 'src/rl_policy'); from rl_policy.bench_policy import main; main(['src/rl_policy/models/gru_u$U.npz', '--calls', '3000'])"
echo "ready: src/rl_policy/models/gru_u$U.npz   (git add it, push, pull on the car, colcon build rl_policy)"
