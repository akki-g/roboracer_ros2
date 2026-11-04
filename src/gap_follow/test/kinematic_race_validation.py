#!/usr/bin/env python3
"""Opt-in, ROS-free race-controller checks using ray-cast walls and bicycle motion.

Run the short matrix (six cases):
    python3 src/gap_follow/test/kinematic_race_validation.py --quick
Run additional control intervals, aggressive turns, and longer approaches:
    python3 src/gap_follow/test/kinematic_race_validation.py --full

The 20 m/s setting is a ceiling, not an expected or measured physical speed.
This harness never starts ROS nodes or writes vehicle configuration files.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

import test_gap_follow_standalone as geometry
from test_gap_follow_curved_entry import narrowing_rounded_turn
from test_gap_follow_traps import dogleg


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'low_speed': {'max_speed': 0.5},
    'high_speed_ceiling': {'max_speed': 20.0},
    'aggressive_simulation_only': {
        'max_speed': 20.0, 'braking_decel': 9.0,
        'max_acceleration': 9.0, 'max_lateral_accel': 6.0,
    },
}
LIMITATIONS = [
    'Ideal bicycle motion tracks commanded speed and steering immediately; '
    'no tire slip, steering lag, motor dynamics, or braking lag is simulated.',
    'Static synthetic walls and noiseless 1081-beam scans are used; '
    'these are not measurements reconstructed from the photograph.',
    'Simulation time advances independently of computation time. A 0.025 s '
    'control interval does not establish real-time 40 Hz capability.',
    'Reported peak speeds are simulated commanded speeds, not physical-car '
    'measurements. Braking and curvature limits can keep them far below the cap.',
    'The collision body uses the existing regression estimate of 0.155 m '
    'half-width, with the configured front and rear extents. It is not a new calibration.',
    'Aggressive parameters apply only to this offline experiment and do not '
    'validate those limits for the physical car.',
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def load_controller(path):
    spec = importlib.util.spec_from_file_location('kinematic_validation_controller', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load controller: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    source = path.read_bytes()
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def cases(full):
    matrix = [('low_speed', 'narrowing', 0.1),
              ('high_speed_ceiling', 'narrowing', 0.1),
              ('aggressive_simulation_only', 'dogleg', 0.1)]
    if full:
        matrix.extend([
            ('low_speed', 'narrowing', 0.025),
            ('high_speed_ceiling', 'narrowing', 0.025),
            ('high_speed_ceiling', 'long_approach', 0.1),
            ('aggressive_simulation_only', 'narrowing', 0.1),
            ('aggressive_simulation_only', 'narrowing', 0.025),
            ('aggressive_simulation_only', 'dogleg', 0.025),
        ])
    return [(profile, scenario, interval, direction)
            for profile, scenario, interval in matrix for direction in (-1, 1)]


def run_case(controller, config, scenario, dt, direction):
    if scenario == 'dogleg':
        walls, pose = dogleg(direction), np.array([-3.0, 0.0, 0.0])
        reached_goal = lambda p: p[0] > 6.0
    else:
        start_x = -7.5 if scenario == 'long_approach' else -2.5
        walls = narrowing_rounded_turn(direction)
        pose = np.array([start_x, -float(direction), 0.0])
        reached_goal = lambda p: p[0] < -0.5 and direction * p[1] > 0.5

    # The shared fixture helpers read these two motion/sensor dimensions.
    old_dimensions = geometry.WHEELBASE, geometry.LIDAR_OFFSET
    geometry.WHEELBASE, geometry.LIDAR_OFFSET = config.wheelbase, config.lidar_offset
    commands, compute_times = [], []
    elapsed, distance = 0.0, 0.0
    outcome = 'timeout'
    began = time.perf_counter()
    try:
        follower = controller.GapFollower(config)
        for _ in range(math.ceil(35.0 / dt)):
            scan = geometry.raycast(walls, pose)
            before_compute = time.perf_counter()
            command = geometry.compute(follower, scan, dt=dt)
            compute_times.append(time.perf_counter() - before_compute)
            commands.append(command)
            if command.speed <= 1e-5:
                outcome = 'stopped:' + command.reason
                break
            for _ in range(5):
                pose = geometry.advance(pose, command, dt / 5)
                elapsed += dt / 5
                distance += command.speed * dt / 5
                if geometry.body_intersects_wall(
                        walls, pose, front=config.vehicle_front,
                        rear=config.vehicle_rear, half_width=0.155):
                    outcome = 'collision'
                    break
            if outcome == 'collision':
                break
            if reached_goal(pose):
                outcome = 'finished'
                break
    finally:
        geometry.WHEELBASE, geometry.LIDAR_OFFSET = old_dimensions

    speeds = np.array([command.speed for command in commands])
    timings = 1000 * np.asarray(compute_times)
    return {
        'scenario': scenario, 'direction': 'left' if direction > 0 else 'right',
        'control_interval_s': dt, 'outcome': outcome, 'command_count': len(commands),
        'simulated_time_s': round(elapsed, 4), 'distance_m': round(distance, 4),
        'peak_commanded_speed_mps': round(float(np.max(speeds)), 6),
        'mean_commanded_speed_mps': round(float(np.mean(speeds)), 6),
        'wall_time_s': round(time.perf_counter() - began, 3),
        'compute_wall_time_ms': {
            'median': round(float(np.median(timings)), 3),
            'p95': round(float(np.percentile(timings, 95)), 3),
            'maximum': round(float(np.max(timings)), 3),
        },
        'final_pose_x_y_yaw': [round(float(value), 6) for value in pose],
        'final_command': asdict(commands[-1]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--controller', type=Path,
                        default=PACKAGE_ROOT / 'scripts/gap_follow_standalone.py')
    parser.add_argument('--output', type=Path,
                        default=PACKAGE_ROOT / 'docs/kinematic_validation.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true', help='Six cases; the default.')
    mode.add_argument('--full', action='store_true', help='Eighteen cases including 40 Hz intervals.')
    args = parser.parse_args(argv)
    controller_path = args.controller.resolve()
    controller, controller_digest = load_controller(controller_path)
    configurations = {name: controller.GapConfig(**overrides)
                      for name, overrides in PROFILES.items()}
    try:
        source_label = str(controller_path.relative_to(PACKAGE_ROOT))
    except ValueError:
        source_label = controller_path.name
    matrix = cases(args.full)
    report = {
        'schema_version': 1, 'suite': 'full' if args.full else 'quick',
        'controller_source': source_label,
        'controller_sha256': controller_digest,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'profiles': {name: {'configuration': asdict(config),
                            'configuration_sha256': digest(asdict(config))}
                     for name, config in configurations.items()},
        'limitations': LIMITATIONS, 'expected_case_count': len(matrix),
        'complete': False, 'all_finished': False, 'results': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for number, (profile, scenario, dt, direction) in enumerate(matrix, 1):
        result = run_case(controller, configurations[profile], scenario, dt, direction)
        result['profile'] = profile
        report['results'].append(result)
        report['complete'] = number == len(matrix)
        report['all_finished'] = (report['complete'] and
                                  all(row['outcome'] == 'finished' for row in report['results']))
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f"[{number}/{len(matrix)}] {profile} {scenario} {result['direction']} "
              f"dt={dt}: {result['outcome']}; "
              f"peak={result['peak_commanded_speed_mps']:.3f} m/s", flush=True)
    return 0 if report['all_finished'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
