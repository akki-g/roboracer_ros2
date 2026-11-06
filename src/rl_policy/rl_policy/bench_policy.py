"""Time the NumPy policy forward pass (spec: under 5 ms at the 99th percentile
on the Jetson CPU, against a 40 ms tick). No ROS needed:

    ros2 run rl_policy bench_policy <policy.npz> [--calls 10000]
    python3 -m rl_policy.bench_policy <policy.npz>
"""

import argparse
import os
import sys
import time

import numpy as np

from . import obs as ob
from .numpy_policy import NumpyPolicy

BUDGET_P99_MS = 5.0
TICK_MS = 40.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy")
    ap.add_argument("--calls", type=int, default=10000)
    a = ap.parse_args(argv)
    pol = NumpyPolicy(a.policy)
    cfg = pol.config
    n, fs = len(cfg["beam_indices"]), int(cfg["frame_stack"])
    rng = np.random.default_rng(0)
    stack = ob.init_stack(ob.normalize_scan(rng.uniform(0.3, 9.0, n)), fs)
    prev = ob.reset_action()
    pol.reset()
    times = np.empty(a.calls)
    for i in range(a.calls):
        stack = ob.push_scan(stack, ob.normalize_scan(rng.uniform(0.3, 9.0, n)))
        o = ob.build_obs(stack, rng.uniform(0, 7), prev)
        t0 = time.perf_counter()
        prev = pol(o)
        times[i] = (time.perf_counter() - t0) * 1e3
        if i % 2500 == 2499:
            pol.reset()  # as on re-activation
    p50, p99, mx = np.percentile(times, 50), np.percentile(times, 99), times.max()
    print(f"{cfg.get('arch', 'mlp')} policy, {a.calls} calls: p50 {p50:.3f} ms  p99 {p99:.3f} ms  max {mx:.3f} ms "
          f"(budget p99 < {BUDGET_P99_MS} ms; tick {TICK_MS:.0f} ms)")
    print(f"calls longer than a tick: {int((times > TICK_MS).sum())}   "
          f"BLAS threads: OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    ok = p99 < BUDGET_P99_MS
    print("PASS" if ok else "FAIL")
    if not ok:
        print("Spikes, not the median, usually fail this. On the Jetson: sudo nvpmodel -m 0 && sudo jetson_clocks "
              "(full clocks), stop anything else using the CPU, and run it again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
