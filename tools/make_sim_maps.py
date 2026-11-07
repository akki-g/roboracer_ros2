#!/usr/bin/env python3
"""Copy never-trained maps from the f1tenth_gym_jax fork into sim/maps and
write one ROS sim config per map (sim/sim_rl_<Name>.yaml), from sim_rl.yaml,
starting on the centerline's first point and heading along it.

    python3 tools/make_sim_maps.py [../f1tenth_gym_jax/rl/maps]

The maps are the fork's "unseen" set (rl/eval_sweep.py): used neither in
training nor in checkpoint selection. Start one with  MAP=<Name> tools/sim_gru.sh

    python3 tools/make_sim_maps.py --last-race [maps dir]

instead writes the last race's track (LastRace2, and LastRace2Rev the other way round) and
LastRace2Obs / LastRace2RevObs: the same track with parked cars (OBSTACLES of them, 0.58 x 0.31 m,
where the corridor is >= 1.1 m wide, pushed toward one wall as in the fork's obstacle training).
Needs numpy and Pillow (e.g. the fork's venv).
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


LAST_RACE = ["LastRace2", "LastRace2Rev"]
OBSTACLES = 3
CAR_LENGTH, CAR_WIDTH = 0.58, 0.31  # m, the fork's car footprint
OBSTACLE_MIN_WIDTH = 1.3  # m: room to pass (the fork's training allows 1.1, rarely right out of a hairpin)
CLEAR_RUN = 20  # centerline points (2 m) before and after a parked car without a sharp turn
MAX_TURN = 0.5  # rad of heading change over CLEAR_RUN
SWEEPER_WIDTH, SWEEPER_TURN = 1.7, 0.9  # also: wide sweeping bends (>= 1.7 m wide, < 0.9 rad over CLEAR_RUN)
MIN_GAP = 5.0  # m between parked cars along the track
START_CLEAR = 4.0  # m of free track on both sides of the start (both directions start there)


def write_config(repo, name, map_name, cl, note):
    """sim/sim_rl_<name>.yaml for map sim/maps/<map_name>, starting on the centerline's first point."""
    template = (repo / "sim" / "sim_rl.yaml").read_text()
    x0, y0 = cl[0, 0], cl[0, 1]
    theta = math.atan2(cl[1, 1] - y0, cl[1, 0] - x0)
    cfg = template
    cfg = re.sub(r"map_path: '[^']*'", f"map_path: '/root/ros2_ws/sim/maps/{map_name}'", cfg)
    cfg = re.sub(r"sx: [-0-9.]+", f"sx: {x0:.4f}", cfg)
    cfg = re.sub(r"sy: [-0-9.]+", f"sy: {y0:.4f}", cfg)
    cfg = re.sub(r"stheta: [-0-9.]+", f"stheta: {theta:.4f}", cfg)
    cfg = cfg.replace("#  - a held-out training map (Duct1077, 2.2 m wide generated duct track)", f"#  - {note}")
    (repo / "sim" / f"sim_rl_{name}.yaml").write_text(cfg)


def track_stats(cl):
    length = float(np.sum(np.hypot(*np.diff(np.vstack([cl[:, :2], cl[:1, :2]]), axis=0).T)))
    return length, float(np.median(cl[:, 2] + cl[:, 3]))


def place_obstacles(cl, n=OBSTACLES):
    """Up to n parked cars (centre x, y, yaw), >= MIN_GAP apart and away from the start, at the
    straightest spots where the corridor is >= OBSTACLE_MIN_WIDTH and the track runs without a
    sharp turn for 2 m either side (a straight run-up in both directions), or on a wide sweeping
    bend (SWEEPER_WIDTH / SWEEPER_TURN), pushed 80% of
    the way toward one wall (sides alternate; the other side if one has no room)."""
    xy, wl, wr = cl[:, :2], cl[:, 2], cl[:, 3]
    seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    length = float(seg.sum())
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    yaw = np.arctan2(t[:, 1], t[:, 0])
    turn = np.abs(np.angle(np.exp(1j * (np.roll(yaw, -5) - np.roll(yaw, 5)))))  # heading change over ~1 m
    run = np.maximum(np.abs(np.angle(np.exp(1j * (yaw - np.roll(yaw, CLEAR_RUN))))),
                     np.abs(np.angle(np.exp(1j * (np.roll(yaw, -CLEAR_RUN) - yaw)))))
    w = wl + wr
    ok = (((w >= OBSTACLE_MIN_WIDTH) & (run < MAX_TURN)) | ((w >= SWEEPER_WIDTH) & (run < SWEEPER_TURN))) \
        & (s >= START_CLEAR) & (s <= length - START_CLEAR)
    picked = []
    for i in np.flatnonzero(ok)[np.argsort(turn[ok], kind="stable")]:  # straightest first, >= MIN_GAP apart
        if all(min(abs(s[i] - s[j]), length - abs(s[i] - s[j])) >= MIN_GAP for j in picked):
            picked.append(int(i))
        if len(picked) == n:
            break
    out, side = [], 1.0
    for i in sorted(picked, key=lambda j: s[j]):
        for sd in (side, -side):
            room = (wl[i] if sd > 0 else wr[i]) - CAR_WIDTH / 2 - 0.08
            if room > 0.05:
                break
        ey = sd * 0.8 * room
        normal = np.array([-np.sin(yaw[i]), np.cos(yaw[i])])  # left
        c = xy[i] + ey * normal
        out.append((float(c[0]), float(c[1]), float(yaw[i]), float(s[i]), float(sd)))
        side = -sd
    return out


def draw_obstacles(png_in, png_out, map_yaml, cars):
    from PIL import Image, ImageDraw

    meta = dict(re.findall(r"^(\w+): *(.*)$", map_yaml.read_text(), re.M))
    res = float(meta["resolution"])
    ox, oy = (float(v) for v in meta["origin"].strip("[]").split(",")[:2])
    img = Image.open(png_in).convert("L")
    h = img.height
    draw = ImageDraw.Draw(img)
    for x, y, yaw, *_ in cars:
        c, sn = math.cos(yaw), math.sin(yaw)
        corners = [(x + c * dx - sn * dy, y + sn * dx + c * dy)
                   for dx, dy in ((CAR_LENGTH / 2, CAR_WIDTH / 2), (-CAR_LENGTH / 2, CAR_WIDTH / 2),
                                  (-CAR_LENGTH / 2, -CAR_WIDTH / 2), (CAR_LENGTH / 2, -CAR_WIDTH / 2))]
        draw.polygon([((px - ox) / res, h - (py - oy) / res) for px, py in corners], fill=0)
    img.save(png_out)


def last_race(repo, src_root):
    out_maps = repo / "sim" / "maps"
    fwd = np.loadtxt(src_root / "LastRace2" / "LastRace2_centerline.csv", delimiter=",", comments="#")
    cars = place_obstacles(fwd)
    for name in LAST_RACE:
        src = src_root / name
        for f in (f"{name}.png", f"{name}.yaml", f"{name}_centerline.csv"):
            shutil.copy2(src / f, out_maps / f)
        cl = np.loadtxt(src / f"{name}_centerline.csv", delimiter=",", comments="#")
        length, width = track_stats(cl)
        way = "the other way round" if name.endswith("Rev") else "as raced"
        write_config(repo, name, name, cl, f"{name}: the last race's track, {way} ({length:.0f} m, ~{width:.1f} m wide)")
        obs = f"{name}Obs"
        draw_obstacles(src / f"{name}.png", out_maps / f"{obs}.png", src / f"{name}.yaml", cars)
        (out_maps / f"{obs}.yaml").write_text((src / f"{name}.yaml").read_text().replace(f"{name}.png", f"{obs}.png"))
        shutil.copy2(src / f"{name}_centerline.csv", out_maps / f"{obs}_centerline.csv")
        write_config(repo, obs, obs, cl, f"{obs}: {name} with {len(cars)} parked cars ({length:.0f} m, ~{width:.1f} m wide)")
        print(f"{name:14s} {length:5.0f} m, ~{width:.1f} m wide; {obs}: parked cars at s = "
              + ", ".join(f"{c[3]:.1f} m ({'left' if c[4] > 0 else 'right'})" for c in cars))


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    args = [a for a in sys.argv[1:] if a != "--last-race"]
    src_root = pathlib.Path(args[0] if args else repo.parent / "f1tenth_gym_jax" / "rl" / "maps")
    if "--last-race" in sys.argv[1:]:
        last_race(repo, src_root)
        return
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
        length, width = track_stats(cl)
        write_config(repo, name, name, cl, f"{name}: never used in training or checkpoint selection ({length:.0f} m, ~{width:.1f} m wide)")
        rows.append((name, length, width))
    print(f"{'map':14s} {'length m':>9s} {'width m':>8s}")
    for name, length, width in rows:
        print(f"{name:14s} {length:9.0f} {width:8.1f}")


if __name__ == "__main__":
    main()
