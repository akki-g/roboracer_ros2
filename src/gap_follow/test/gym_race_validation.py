#!/usr/bin/env python3
"""Opt-in, headless F1TENTH Gym evaluation; never imports ROS or drives hardware.

Requires the user's existing f110_gym installation. The controller receives
scans only; simulator poses/maps are used solely to score the run. This models
dynamic tire behavior, steering slew and 20 ms actuator delay, with selectable
control period and scan age. Computation time is reported separately: fixed
simulation scheduling does not establish that the controller meets a deadline.

Example:
  python3 src/gap_follow/test/gym_race_validation.py --track spielberg \
    --geometry corrected --max-speed 20 --clip-sim-range-noise \
    --output /tmp/gap_spielberg

This Gym fork adds Gaussian noise after clipping scans to range_max. The
explicit --clip-sim-range-noise option restores that declared range limit;
without it, above-maximum readings remain invalid to reproduce the native
integration problem. It never changes physical-sensor input handling.

The bridge profile reproduces the current ROS bridge's scan offset (.275 m
from the Gym state origin). Corrected geometry keeps that same simulated sensor
and translates the planner's rear-axle frame by lr; it changes no hardware YAML.
Gym's low-speed kinematic fallback has a different reference-point convention
from its dynamic model; this remains a simulator limitation.
"""

import argparse
from collections import Counter, deque
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def load_controller(path):
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location('gap_gym_controller', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def body_clearance(scan_simulator, state, length, width):
    """Intersect occupied map cells with the full centered body using SAT.

    Gym's built-in wall check uses iTTC, not its full collision rectangle. Cell
    polygons prevent missing thin raster walls between body sample points. The
    maximum separating-axis gap is a lower bound on Euclidean clearance; zero
    or negative means polygon contact. Queries are capped at about one metre.
    """
    resolution = scan_simulator.map_resolution
    if abs(scan_simulator.origin[2]) > 1e-12:
        raise ValueError('This evaluator currently requires an unrotated map')
    x, y, yaw = state[0], state[1], state[4]
    radius = math.hypot(length / 2, width / 2) + 1.0
    c0, r0 = np.floor((np.array([x - radius, y - radius])
                       - scan_simulator.origin[:2]) / resolution).astype(int)
    c1, r1 = np.ceil((np.array([x + radius, y + radius])
                      - scan_simulator.origin[:2]) / resolution).astype(int)
    height, cols = scan_simulator.map_img.shape
    if c0 < 0 or r0 < 0 or c1 >= cols or r1 >= height:
        return -math.inf
    rows, columns = np.nonzero(scan_simulator.map_img[r0:r1 + 1, c0:c1 + 1] == 0)
    if not len(rows):
        return 1.0
    dx = scan_simulator.origin[0] + (columns + c0 + 0.5) * resolution - x
    dy = scan_simulator.origin[1] + (rows + r0 + 0.5) * resolution - y
    cosine, sine = math.cos(yaw), math.sin(yaw)
    local_x = dx * cosine + dy * sine
    local_y = -dx * sine + dy * cosine
    cell = resolution / 2
    rotated_cell = cell * (abs(cosine) + abs(sine))
    gap = np.maximum.reduce((
        np.abs(local_x) - length / 2 - rotated_cell,
        np.abs(local_y) - width / 2 - rotated_cell,
        np.abs(dx) - length / 2 * abs(cosine) - width / 2 * abs(sine) - cell,
        np.abs(dy) - length / 2 * abs(sine) - width / 2 * abs(cosine) - cell))
    return float(np.min(gap))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=('levine', 'spielberg'), default='levine')
    parser.add_argument('--maps', type=Path,
                        default=Path.home() / 'sim_ws/src/f1tenth_gym_ros/maps')
    parser.add_argument('--controller', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'scripts/gap_follow_standalone.py')
    parser.add_argument('--geometry', choices=('bridge', 'corrected'), default='bridge')
    parser.add_argument('--profile', choices=('default', 'sim-race'), default='default')
    parser.add_argument('--max-speed', type=float, default=20.0)
    parser.add_argument('--path-horizon', type=float)
    parser.add_argument('--lidar-range', type=float)
    parser.add_argument('--control-period', type=float, default=0.025)
    parser.add_argument('--scan-delay', type=float, default=0.0)
    parser.add_argument('--clip-sim-range-noise', action='store_true', help=(
        'Clip finite simulator readings above range_max after Gym adds noise. '
        'This explicit simulator-only correction does not alter the controller.'))
    parser.add_argument('--duration', type=float, default=180.0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--output', type=Path, default=Path('/tmp/gap_gym_validation'))
    args = parser.parse_args()
    physics_dt = 0.005
    controls_per_step = round(args.control_period / physics_dt)
    if (controls_per_step < 1
            or abs(controls_per_step * physics_dt - args.control_period) > 1e-9
            or args.scan_delay < 0 or args.duration <= 0):
        parser.error('Control period must be a positive multiple of 5 ms; delay >= 0; duration > 0')

    import pyglet
    pyglet.options['headless'] = True
    from PIL import Image
    from f110_gym.envs.f110_env import F110Env
    from f110_gym.envs.base_classes import RaceCar

    controller, controller_digest = load_controller(args.controller)
    map_path = args.maps / ('Spielberg_map' if args.track == 'spielberg' else 'levine')
    start = np.array([0.0, 0.0, 0.27 if args.track == 'spielberg' else 0.0])
    env = F110Env(map=str(map_path), map_ext='.png', num_agents=1,
                 timestep=physics_dt, lidar_dist=0.275, seed=args.seed)
    car = env.sim.agents[0]
    collision_snapshot = [None]
    native_check_ttc = car.check_ttc

    def capture_ttc_state(scan):
        # Native Gym zeros velocity AND yaw when declaring a collision. Preserve
        # the integrated pose for an independent body/contact check and plotting.
        before_reset = car.state.copy()
        result = native_check_ttc(scan)
        if car.in_collision:
            collision_snapshot[0] = before_reset
        return result

    car.check_ttc = capture_ttc_state
    # Keep the native 20 ms delay when reducing the default 10 ms physics step.
    car.steer_buffer_size = round(0.02 / physics_dt)
    overrides = {'max_speed': args.max_speed, 'lidar_offset': 0.275}
    if args.geometry == 'corrected':
        overrides.update(lidar_offset=0.275 + env.params['lr'],
                         vehicle_front=env.params['length'] / 2 + env.params['lr'],
                         vehicle_rear=env.params['length'] / 2 - env.params['lr'],
                         wheelbase=env.params['lf'] + env.params['lr'])
    if args.profile == 'sim-race':
        overrides.update(max_acceleration=9.0, braking_decel=9.0,
                         max_lateral_accel=min(8.0, env.params['mu'] * 9.81))
    for argument, parameter in (('path_horizon', 'path_horizon'),
                                ('lidar_range', 'max_lidar_range')):
        if getattr(args, argument) is not None:
            overrides[parameter] = getattr(args, argument)
    cfg = controller.GapConfig(**overrides)
    follower = controller.GapFollower(cfg)
    obs, _, _, _ = env.reset(start[None, :])
    scan_simulator = RaceCar.scan_simulator
    original_map = np.flipud(np.asarray(Image.open(map_path.with_suffix('.png')).convert('L')))
    history = deque([(0.0, np.asarray(obs['scans'][0]).copy())])
    timeline, states, requested, runtimes, clearances, actual_speeds = [], [], [], [], [], []
    reasons = Counter()
    overflow_readings = 0
    distance = 0.0
    max_excursion = 0.0
    stopped_time = 0.0
    next_report = 5.0
    outcome = 'duration_limit'
    initial_clearance = body_clearance(scan_simulator, car.state,
                                       env.params['length'], env.params['width'])
    if initial_clearance <= 0:
        raise ValueError(f'Starting body intersects the map: clearance={initial_clearance}')
    wall_start = time.monotonic()
    print(json.dumps({'event': 'start', 'track': args.track, 'geometry': args.geometry,
                      'profile': args.profile, 'controller_sha256': controller_digest,
                      'control_period': args.control_period, 'scan_delay': args.scan_delay,
                      'clip_sim_range_noise': args.clip_sim_range_noise,
                      'planner_config': asdict(cfg), 'sim_params': env.params}), flush=True)
    t = 0.0
    for _ in range(math.ceil(args.duration / args.control_period)):
        while len(history) > 1 and history[1][0] <= t - args.scan_delay + 1e-9:
            history.popleft()
        stamp, scan = history[0]
        overflow = np.isfinite(scan) & (scan > scan_simulator.max_range)
        overflow_readings += int(np.count_nonzero(overflow))
        if args.clip_sim_range_noise:
            scan = np.where(overflow, scan_simulator.max_range, scan)
        compute_start = time.perf_counter()
        command = follower.compute(scan, -car.fov / 2, scan_simulator.get_increment(),
                                   0.0, scan_simulator.max_range, dt=args.control_period)
        runtimes.append(time.perf_counter() - compute_start)
        reasons[command.reason] += 1
        action = np.array([[command.steering_angle, command.speed]])
        requested.append((command.steering_angle, command.speed,
                          command.path_clearance, t - stamp))
        for _ in range(controls_per_step):
            before = car.state[:2].copy()
            obs, _, done, _ = env.step(action)
            t += physics_dt
            physical_state = (collision_snapshot[0] if obs['collisions'][0]
                              and collision_snapshot[0] is not None else car.state)
            actual_speeds.append(float(physical_state[3]))
            distance += float(np.linalg.norm(car.state[:2] - before))
            max_excursion = max(max_excursion, float(np.linalg.norm(car.state[:2] - start[:2])))
            clearance = body_clearance(scan_simulator, physical_state,
                                       env.params['length'], env.params['width'])
            clearances.append(clearance)
            if obs['collisions'][0] or clearance <= 0:
                outcome = 'gym_collision' if obs['collisions'][0] else 'body_map_intersection'
                break
            column, row = np.floor((car.state[:2] - scan_simulator.origin[:2])
                                   / scan_simulator.map_resolution).astype(int)
            # Levine's gray unmapped area is treated as free by this old Gym
            # fork. Leaving through an open map edge is not a completed track.
            if original_map[row, column] < 250:
                outcome = 'left_mapped_track'
                break
        timeline.append(t)
        states.append(physical_state.copy())
        history.append((t, np.asarray(obs['scans'][0]).copy()))
        if outcome != 'duration_limit':
            break
        stopped_time = stopped_time + args.control_period if abs(car.state[3]) < 0.02 else 0.0
        if stopped_time >= 2.0:
            outcome = 'stopped:' + command.reason
            break
        # Gym's finish detector alone can count tiny local loops as laps.
        # Also require a substantial excursion and distance before accepting it.
        if obs['lap_counts'][0] >= 1 and distance > 30.0 and max_excursion > 10.0:
            outcome = 'lap_completed'
            break
        if t >= next_report:
            print(json.dumps({'event': 'progress', 'sim_time': round(t, 3),
                              'distance': round(distance, 2), 'actual_speed': round(float(car.state[3]), 3),
                              'max_actual_speed': round(float(np.max(np.asarray(states)[:, 3])), 3),
                              'pose': car.state[[0, 1, 4]].tolist()}), flush=True)
            next_report += 5.0
    states, requested, runtimes = np.asarray(states), np.asarray(requested), np.asarray(runtimes)
    result = {'outcome': outcome, 'track': args.track, 'geometry': args.geometry,
              'profile': args.profile, 'seed': args.seed, 'sim_seconds': t,
              'wall_seconds': time.monotonic() - wall_start, 'distance_m': distance,
              'gym_laps': float(obs['lap_counts'][0]), 'max_excursion_m': max_excursion,
              'requested_speed_limit_mps': args.max_speed,
              'max_actual_speed_mps': float(np.max(actual_speeds)),
              'mean_actual_speed_mps': float(np.mean(actual_speeds)),
              'max_command_speed_mps': float(np.max(requested[:, 1])),
              'minimum_body_cell_clearance_m': float(min(clearances)),
              'native_gym_collision': bool(obs['collisions'][0]),
              'final_body_cell_clearance_m': float(clearances[-1]),
              'collision_evaluator': 'rectangle_cell_separating_axis_test',
              'control_period_s': args.control_period, 'scan_delay_s': args.scan_delay,
              'clip_sim_range_noise': args.clip_sim_range_noise,
              'overflow_readings_seen': overflow_readings,
              'callback_ms_median_p95_max': (1000 * np.percentile(runtimes, [50, 95, 100])).tolist(),
              'callbacks_over_control_period': int(np.count_nonzero(runtimes > args.control_period)),
              'callbacks_over_300ms': int(np.count_nonzero(runtimes > 0.3)),
              'callbacks': len(runtimes), 'reasons': dict(reasons),
              'controller_sha256': controller_digest,
              'planner_config': asdict(cfg), 'sim_params': env.params,
              'requested_columns': ['steering_angle', 'speed', 'path_clearance', 'scan_age'],
              'map_path': str(map_path), 'start_pose': start.tolist(),
              'physics_dt': physics_dt, 'steering_delay_s': 0.02,
              'final_pose': states[-1, [0, 1, 4]].tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(args.output.with_suffix('.npz'), time=timeline, states=states,
                        requested=requested, callback_seconds=runtimes)
    print(json.dumps({'event': 'result', **result}), flush=True)


if __name__ == '__main__':
    main()
