#!/usr/bin/env python3
"""Plot saved Gym results; reads artifacts only and never starts a simulator."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    runs = []
    for path in args.results:
        result = json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as data:
            states = data['states'].copy()
        runs.append((result, states))
    columns = min(2, len(runs))
    rows = math.ceil(len(runs) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 5 * rows),
                             squeeze=False, constrained_layout=True)
    norm = plt.Normalize(0, max(float(np.max(states[:, 3])) for _, states in runs))
    for ax, (result, states) in zip(axes.ravel(), runs):
        map_path = Path(result['map_path'])
        metadata = yaml.safe_load(map_path.with_suffix('.yaml').read_text())
        pixels = np.asarray(Image.open(map_path.with_suffix('.png')).convert('L'))
        resolution, origin = metadata['resolution'], metadata['origin']
        extent = (origin[0], origin[0] + pixels.shape[1] * resolution,
                  origin[1], origin[1] + pixels.shape[0] * resolution)
        ax.imshow(np.flipud(pixels), cmap='gray', origin='lower', extent=extent,
                  vmin=0, vmax=255, rasterized=True)
        points = states[:, :2]
        segments = np.stack((points[:-1], points[1:]), axis=1)
        line = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2.5)
        line.set_array(states[1:, 3])
        ax.add_collection(line)
        ax.plot(*points[0], marker='^', color='green', markersize=8, label='Start')
        ax.plot(*points[-1], marker='x', color='crimson', markersize=8, label='End')
        horizon = result['planner_config']['path_horizon']
        outcome = {'gym_collision': 'Gym collision flag',
                   'lap_completed': 'Lap completed',
                   'stopped:no_safe_path': 'Stopped: no safe path'}.get(
                       result['outcome'], result['outcome'])
        ax.set_title(f"{result['track'].title()} · {result['profile']} · {horizon:g} m horizon\n"
                     f"Control {1000 * result['control_period_s']:g} ms / "
                     f"scan delay {1000 * result['scan_delay_s']:g} ms\n"
                     f"{outcome} | peak {result['max_actual_speed_mps']:.2f} m/s | "
                     f"{result['sim_seconds']:.2f} s")
        # Keep short failed runs readable without hiding the traveled region.
        margin = 5.0
        ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
        ax.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.set_xlabel('Map x (m)')
        ax.set_ylabel('Map y (m)')
        ax.legend(loc='upper right', fontsize=8)
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    fig.colorbar(line, ax=list(axes.ravel()[:len(runs)]), label='Simulated actual speed (m/s)',
                 shrink=0.8)
    fig.suptitle('LiDAR-only controller in F1TENTH Gym\n'
                 '20 m/s requested ceiling; actual speeds shown. '
                 'Simulation scheduling does not establish real-time performance.\n'
                 'Native collision flags are not confirmed body impacts; see the validation report.',
                 fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
