"""Policy test for a new car computer (written for the Kria KV260). No ROS, no car hardware:

    python3 -m rl_policy.board_test                 # from src/rl_policy
    ros2 run rl_policy board_test                   # from a built workspace
    tools/kv260_test.sh                             # this, then the ROS loop test

1. Same actions as the Jetson. data/car_replay.npz holds 20 s of a drive on the car
   (Nov 7, gru1025_solo_u0950.npz on the Jetson Orin): the 1081-beam scan each policy
   tick used, the measured speed and previous action it was fed, and the action the
   Jetson computed. The same inputs go through the policy here; the actions must agree
   within ACTION_TOL.
2. Timing. The work of one policy tick as policy_node does it (scan message to arrays,
   emergency-brake TTC, observation, forward pass, safety layer), on the recorded
   scans. The 99th percentile must stay under half a tick and no tick may take longer
   than a whole one (40 ms at the policy's 25 Hz).
"""

import argparse
import array
import os
import pathlib
import platform
import sys
import time

import numpy as np

from . import obs as ob
from .numpy_policy import NumpyPolicy
from .safety import SafetyConfig, apply_safety, min_ttc, request_speed

ACTION_TOL = 1e-6  # on tanh(mean), in [-1, 1]; the Mac agrees with the Jetson to 3e-15
JETSON_FORWARD_P99_MS = 5.0  # bench_policy's budget on the Jetson, for comparison

PKG = pathlib.Path(__file__).resolve().parents[1]


def find_share(*parts):
    """A file of this package: in the source tree, else in the installed share directory."""
    p = PKG.joinpath(*parts)
    if p.exists():
        return p
    try:
        from ament_index_python.packages import get_package_share_directory
        q = pathlib.Path(get_package_share_directory("rl_policy")).joinpath(*parts)
        if q.exists():
            return q
    except Exception:  # no ROS environment
        pass
    raise FileNotFoundError(f"{'/'.join(parts)} not found under {PKG} or the installed rl_policy share directory")


def load_replay(path=None):
    """The recorded drive: dict with scans [T, 1081] float32 (m), speed, prev_action, reset,
    jetson_action, the scan layout and the policy file name."""
    path = pathlib.Path(path) if path else find_share("data", "car_replay.npz")
    with np.load(path, allow_pickle=False) as d:
        r = {k: d[k] for k in d.files}
    # ranges are stored in mm (the UST-10LX's resolution): /1000 gives back the driver's float32 exactly
    r["scans"] = (r.pop("scans_mm").astype(np.float64) / 1000.0).astype(np.float32)
    for k in ("policy", "source"):
        r[k] = str(r[k])
    for k in ("angle_min", "angle_increment", "range_min", "range_max"):
        r[k] = float(r[k])
    return r


def _read(path):
    try:
        return pathlib.Path(path).read_text().strip().strip("\0")
    except OSError:
        return None


def board_info():
    lines = []
    model = _read("/proc/device-tree/model") or _read("/sys/devices/virtual/dmi/id/product_name") or "unknown"
    lines.append(f"board      {model} ({platform.machine()}, {os.cpu_count()} cores)")
    lines.append(f"kernel     {platform.release()}")
    gov = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    cur = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    top = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    if gov:
        lines.append(f"cpu clock  governor {gov}, now {int(cur) / 1e3:.0f} MHz, max {int(top) / 1e3:.0f} MHz"
                     if cur and top else f"cpu clock  governor {gov}")
    lines.append(f"python     {platform.python_version()}, numpy {np.__version__}, "
                 f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    return lines, gov


def replay_actions(pol, rep):
    """The policy's actions on the recorded ticks."""
    idx = np.asarray(pol.config["beam_indices"], dtype=int)
    out = np.zeros_like(rep["jetson_action"])
    for i in range(len(out)):
        if rep["reset"][i]:
            pol.reset()
        scan_norm = ob.normalize_scan(rep["scans"][i][idx])
        out[i] = pol(ob.build_obs(ob.init_stack(scan_norm, 1), rep["speed"][i], rep["prev_action"][i]))
    return out


def time_ticks(pol, rep, calls, cfg):
    """Times (ms) of the scan callback's work, the forward pass and the whole tick, as in policy_node."""
    idx = np.asarray(pol.config["beam_indices"], dtype=int)
    v_cmd_max, steer_max = pol.config["v_cmd_max"], pol.config["steer_max"]
    n = rep["scans"].shape[1]
    msgs = [array.array("f", s.tobytes()) for s in rep["scans"]]  # what rclpy hands the node: array('f')
    t_scan, t_fwd, t_tick = np.empty(calls), np.empty(calls), np.empty(calls)
    pol.reset()
    stack = None
    prev = ob.reset_action()
    speed_request = 0.0
    for i in range(calls):
        j = i % len(msgs)
        speed_meas = float(rep["speed"][j])
        t0 = time.perf_counter()
        # _on_scan
        ranges = np.asarray(msgs[j], dtype=np.float32)
        angles = rep["angle_min"] + np.arange(n) * rep["angle_increment"]
        t1 = time.perf_counter()
        # _tick
        ttc = min_ttc(ranges, angles, speed_meas, cfg.ttc_half_angle, rep["range_min"])
        scan_norm = ob.normalize_scan(ranges[idx])
        if stack is None or rep["reset"][j]:
            stack, prev, speed_request = ob.init_stack(scan_norm, 1), ob.reset_action(), 0.0
            pol.reset()
        else:
            stack = ob.push_scan(stack, scan_norm)
        observation = ob.build_obs(stack, speed_meas, prev)
        t2 = time.perf_counter()
        action = pol(observation)
        t3 = time.perf_counter()
        speed_cmd, steer_cmd = ob.action_to_command(action, v_cmd_max, steer_max)
        speed_request = request_speed(float(speed_cmd), cfg, v_cmd_max, speed_request)
        result = apply_safety(speed_request, steer_cmd, cfg, scan_age=0.0, odom_age=0.0, ttc=ttc)
        prev = ob.command_to_action(result.speed, result.steer, v_cmd_max, steer_max)
        t4 = time.perf_counter()
        t_scan[i], t_fwd[i], t_tick[i] = (t1 - t0) * 1e3, (t3 - t2) * 1e3, (t4 - t0) * 1e3
    return t_scan, t_fwd, t_tick


def _pcts(t):
    return f"p50 {np.percentile(t, 50):7.3f} ms   p99 {np.percentile(t, 99):7.3f} ms   max {t.max():7.3f} ms"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="", help="policy .npz for the timing (default: the one that drove the recording)")
    ap.add_argument("--calls", type=int, default=5000, help="ticks to time (default 5000 = 200 s of driving)")
    ap.add_argument("--replay", default="", help="recorded drive (default: the package's data/car_replay.npz)")
    a = ap.parse_args(argv)

    lines, gov = board_info()
    print("== board")
    for line in lines:
        print("  " + line)

    rep = load_replay(a.replay or None)
    ref_path = find_share("models", rep["policy"])
    ok = True

    print(f"== 1. same actions as the Jetson ({rep['policy']}, {len(rep['speed'])} ticks: {rep['source']})")
    acts = replay_actions(NumpyPolicy(ref_path), rep)
    diff = np.abs(acts - rep["jetson_action"])
    worst = int(diff.max(axis=1).argmax())
    same = bool(np.isfinite(acts).all() and diff.max() <= ACTION_TOL)
    ok &= same
    print(f"  largest difference from the Jetson's actions: {diff.max():.2e} (tick {worst}; allowed {ACTION_TOL:.0e})")
    print(f"  speed action   here {acts[worst, 0]:+.9f}   Jetson {rep['jetson_action'][worst, 0]:+.9f}")
    print(f"  steering       here {acts[worst, 1]:+.9f}   Jetson {rep['jetson_action'][worst, 1]:+.9f}")
    print("  PASS" if same else "  FAIL: this board computes different actions from the same inputs")

    pol = NumpyPolicy(a.policy) if a.policy else NumpyPolicy(ref_path)
    tick_ms = 1e3 / float(pol.config["policy_rate_hz"])
    cfg = SafetyConfig(max_speed=10.0, ttc_half_angle=0.0)  # the car's settings (policy_car.yaml)
    print(f"== 2. timing ({a.calls} ticks, {pathlib.Path(a.policy).name if a.policy else rep['policy']}, "
          f"{pol.config.get('arch', 'mlp')}, {len(pol.config['beam_indices'])} beams)")
    time_ticks(pol, rep, 200, cfg)  # warm-up
    t_scan, t_fwd, t_tick = time_ticks(pol, rep, a.calls, cfg)
    late = int((t_tick > tick_ms).sum())
    fast = bool(np.percentile(t_tick, 99) < tick_ms / 2 and late == 0)
    ok &= fast
    print(f"  scan message  {_pcts(t_scan)}")
    print(f"  forward pass  {_pcts(t_fwd)}   (Jetson budget: p99 < {JETSON_FORWARD_P99_MS:g} ms)")
    print(f"  whole tick    {_pcts(t_tick)}   (budget: p99 < {tick_ms / 2:g} ms, none over {tick_ms:g} ms)")
    print(f"  ticks longer than {tick_ms:g} ms: {late}")
    print("  PASS" if fast else "  FAIL: too slow for the policy's rate")
    if not fast and gov and gov != "performance":
        print(f"  the CPU governor is '{gov}': try  echo performance | sudo tee "
              "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor  and run it again")

    print("== board test: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
