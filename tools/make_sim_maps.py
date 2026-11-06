#!/usr/bin/env python3
"""Copy never-trained maps from the f1tenth_gym_jax fork into sim/maps and
write one ROS sim config per map (sim/sim_rl_<Name>.yaml), from sim_rl.yaml,
starting on the centerline's first point and heading along it.

    python3 tools/make_sim_maps.py [../f1tenth_gym_jax/rl/maps]

The maps are the fork's "unseen" set (rl/eval_sweep.py): used neither in
training nor in checkpoint selection. Start one with  MAP=<Name> tools/sim_gru.sh
"""

import math
import pathlib
import re
import shutil
import sys

import numpy as np

UNSEEN_GEN = ["EvalDuct3012", "EvalDuct3014", "EvalDuct3020", "EvalDuct3023", "EvalDuct3034", "EvalDuct3053"]
UNSEEN_RACETRACKS = ["IMS", "Melbourne", "MexicoCity", "Nuerburgring", "Sakhir", "SaoPaulo",
                     "Sepang", "Shanghai", "Sochi", "Spa", "YasMarina", "Zandvoort"]


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    src_root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else repo.parent / "f1tenth_gym_jax" / "rl" / "maps")
    template = (repo / "sim" / "sim_rl.yaml").read_text()
    out_maps = repo / "sim" / "maps"
    rows = []
    for name in UNSEEN_GEN + UNSEEN_RACETRACKS:
        src = src_root / name
        files = [src / f"{name}.png", src / f"{name}.yaml", src / f"{name}_centerline.csv"]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            print(f"skip {name}: missing {missing} (run `python -m rl.eval_sweep prepare` in the fork)")
            continue
        for f in files:
            shutil.copy2(f, out_maps / f.name)
        cl = np.loadtxt(src / f"{name}_centerline.csv", delimiter=",", comments="#")
        x0, y0 = cl[0, 0], cl[0, 1]
        theta = math.atan2(cl[1, 1] - y0, cl[1, 0] - x0)
        length = float(np.sum(np.hypot(*np.diff(np.vstack([cl[:, :2], cl[:1, :2]]), axis=0).T)))
        width = float(np.median(cl[:, 2] + cl[:, 3]))
        cfg = template
        cfg = re.sub(r"map_path: '[^']*'", f"map_path: '/root/ros2_ws/sim/maps/{name}'", cfg)
        cfg = re.sub(r"sx: [-0-9.]+", f"sx: {x0:.4f}", cfg)
        cfg = re.sub(r"sy: [-0-9.]+", f"sy: {y0:.4f}", cfg)
        cfg = re.sub(r"stheta: [-0-9.]+", f"stheta: {theta:.4f}", cfg)
        cfg = cfg.replace("#  - a held-out training map (Duct1077, 2.2 m wide generated duct track)",
                          f"#  - {name}: never used in training or checkpoint selection ({length:.0f} m, ~{width:.1f} m wide)")
        (repo / "sim" / f"sim_rl_{name}.yaml").write_text(cfg)
        rows.append((name, length, width))
    print(f"{'map':14s} {'length m':>9s} {'width m':>8s}")
    for name, length, width in rows:
        print(f"{name:14s} {length:9.0f} {width:8.1f}")


if __name__ == "__main__":
    main()
