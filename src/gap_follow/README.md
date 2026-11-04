# Standalone LiDAR gap follower

`scripts/gap_follow_standalone.py` runs directly with Python after sourcing ROS
and the car workspace. It uses live LiDAR; it does not require a map or
localization. The existing `levine_*` launch files still select `reactive_node.py`.

## Why the old controller stalled in hairpins

The old controller selected a deep ray in the widest gap, used that ray's
bearing directly as a clipped steering angle, and stopped according to a
straight corridor in front of the LiDAR. Saturated steering reduced its nominal
speed to 0.15 m/s, then front clearance scaled that toward zero. In a square
U-turn regression, it approached 0.5 m front clearance and effectively stopped.
It also discarded scan angles outside ±90° and stopped for any return within
the nearest-obstacle bubble, including side returns.

The controller now:

- Scores gaps using capped depth and distance from their edges.
- Uses the gap as a steering preference, with bicycle geometry and LiDAR offset.
- Evaluates 41 feasible steering arcs over 3 m. Observed obstacles constrain a
  rectangular vehicle footprint; the leading body must remain in observed space.
- Ranks these immediate arcs using checked travel, wall clearance, the gap
  preference and steering continuity. Each scan chooses a new steering command;
  there is no second-turn search or score for unobserved extensions.
- Uses 260° of scan data by default, including side walls, and prefers wall clearance
  and continuity between steering commands. The extra-clearance reward saturates
  at 0.35 m so a wide recess does not automatically beat the through route.
- Limits speed using curvature, free travel along the selected arc, assumed
  braking response, and acceleration. Short checked paths reduce speed through
  the braking bound. Braking and stops are applied immediately.
- Retains the optional inside-wall preference among comparable checked arcs.
- Keeps invalid ranges blocked. Positive infinity is accepted as a no-return
  measurement only when `inf_is_free:=true` is explicitly configured.

Known route-selection limits: the current single-arc controller stops in
closed-loop regressions for a narrowing rounded-apex entry and a forward
dead-end pocket with a side exit. Collision and braking checks remain in place;
removing second-turn planning reduces its ability to find a usable entry.

Obstacle returns are grouped in 3 cm cells, each bounded tightly around all of
its actual returns. These bounds are projected into every tested body pose;
the unmeasured remainder of a cell is not marked occupied. This retains thin
and singleton obstacle returns while reducing computation. Endpoint clearance
bounds certify motion between sampled poses. Ambiguous intervals within the
first 0.3 m of an arc receive up to five batched subdivisions, with the
same body, measured obstacle bounds, visibility and missing-ray checks.
Unresolved intervals remain blocked. This
avoids a false stop caused by a large, fixed sampling pad beside a clear turn,
and checks the full initial interval instead of exempting its starting pose.
These geometric checks assume static obstacles and ideal steering.
Beam lookup indices are cached by scan geometry, with a fixed memory bound;
every scan's ranges, validity and obstacle hits are evaluated afresh. The ROS
wrapper checks freshness again after computation before publishing a command.

Any positive checked travel may produce a braking-limited command. There is no
separate crawl threshold, crawl speed cap or additional stopping-distance
reserve parameter. Occupied or unobserved paths still cause a stop.
Geometric stops preserve the previous steering preference while commanding zero
speed; malformed/stale scans and watchdog stops reset it.

### Historical observations before simplification

The following recordings and timings concern the earlier continuation planner.
They explain the retained footprint checks; they are not validation results or
runtime measurements for the current single-arc controller.

The November 4 live recording distinguished two right-turn stops. At 16:36,
finer collision checks recovered a checked turn in all ten representative scans.
At 16:48, the maximum-right arc actually entered the padded-body obstacle region;
the updated planner still stopped on all three representative scans. The fix
does not make a blocked entry physically turnable. On ten matched moving scans,
median computation changed from 70.4 to 71.4 ms, with worst cases 165.6 and
170.6 ms. This is replay evidence, not a 40 Hz or physical lap-time guarantee.

A later 81-scan capture at 17:23 exposed a separate `no_safe_path`/`ok` alternation:
nearby side returns stayed outside the padded body, but filling whole grid
cells and adding their half-diagonal made them appear to overlap it. Tight
measured bounds and the finer interval checks recovered moving paths in all
81 replayed scans, versus 31 with the former cell representation. Six
representative scans also passed with a 20 m/s ceiling; this does not mean
the car drove at that speed. Matched moving-scan computation increased from
79.6 to 123.0 ms median (worst updated sample 290.4 ms on the shared host),
mostly because more continuation routes became available to check.
The footprint dimensions, visibility checks and post-compute freshness stop
remain in force. Missing inner-wall support simply disables the inside-line
preference and leaves the normal planner in control; it does not independently
trigger `no_safe_path` or permit unchecked wall-following commands.

The 17:37 capture showed a different pose: every scan contained raw returns
inside the configured padded front-left corner, before any cell aggregation.
That overlap still requires a stop. The padded rectangle is a configured
collision boundary, not proof that the physical chassis is touching a wall.

## Run and diagnose

The calibrated hardware configuration in `f1tenth_stack/config` remains the
source of truth for the car. Running this Python file starts only the planner;
bringup starts the LiDAR, joystick, mux and VESC drivers. The planner has its own
parameters and does not automatically import the hardware YAML files.

On this car, use the following setup in **each terminal**:

```bash
cd /home/kit/f1tenth_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
```

Domain 42 is the domain observed on the running car. All terminals must use the
same domain. If bringing this workspace to another car, use that car's domain.

In terminal 1, start hardware bringup if it is not already running:

```bash
ros2 launch f1tenth_stack bringup_launch.py
```

In terminal 2, run the planner against the normal mux input:

```bash
python3 src/gap_follow/scripts/gap_follow_standalone.py --ros-args \
  -p max_speed:=0.5 --log-level gap_follow_standalone:=debug
```

The speed override is a temporary test ceiling; it does not edit the calibrated
VESC settings. Running the Python file directly requires no rebuild. If it is
already running when the file changes, stop that process with Ctrl+C and restart
it to load the edits.

With your installed joystick configuration, hold **R1 alone** (button index 5)
to allow autonomous commands through. With no buttons pressed, `joy_teleop`
publishes zero-speed `/teleop` commands at a higher priority than `/drive`.
R1 suppresses that default command; after the 0.2 s mux timeout, `/drive` is
eligible. L1 (button index 4) selects manual driving and keeps joystick priority.
R1 does not override an algorithm stop or start the Python process. Avoid other
buttons during this handoff: the current default-action code suppresses its
stop command whenever any button is pressed, not exclusively for R1.

The actual command chain in this repository is:

```text
/scan -> gap_follow_standalone -> /drive -> ackermann_mux
       -> /ackermann_cmd -> ackermann_to_vesc_node -> /commands/motor/speed
```

For a dry run that cannot feed the navigation input, remap the output:

```bash
python3 src/gap_follow/scripts/gap_follow_standalone.py --ros-args \
  -r /drive:=/gap_follow_test -p max_speed:=0.5 \
  --log-level gap_follow_standalone:=debug
```

With that `/gap_follow_test` remap, R1 will not route this planner to the motors.
Remove the remap when using the normal driving command above.

If the car does not move, these read-only checks distinguish the stop location:

```bash
ros2 topic echo /drive --once --field drive
ros2 topic echo /ackermann_cmd --once --field drive
ros2 topic echo /joy --once --field buttons
```

| Observation | Meaning |
| --- | --- |
| `/drive` speed is zero | The planner is requesting a stop; read its reason in terminal 2 |
| `/drive` is positive but `/ackermann_cmd` is zero | Check joystick/mux selection; while holding R1, `buttons[5]` should be 1 and `buttons[4]` should be 0 |
| Both commands are positive | The planner and mux are passing motion; inspect VESC/driver status |
| No `/scan`, `waiting_for_scan`, or `scan_timeout` | Check bringup and matching ROS domains |

The hardware snapshots used during diagnosis had valid scan data and a connected
command chain, but repeated `no_safe_path` decisions with a 270° planning sector.
The far-left/rear scan edge at +135° reported about 0.25–0.38 m, while the side
walls were roughly 0.7–1 m away. Many of these peripheral points project inside
the new padded footprint and veto every path. This is consistent with a
self-return or an obstructed scan edge, rather than the distant track walls.

The default `forward_fov_deg` is now **260°** (±130°). This omits the outer 5° at
each rear edge while retaining both ±90° side-wall directions and the complete
front sector. It does not reduce the vehicle clearance margin or alter the
LiDAR driver's calibrated configuration. Excluded sectors remain unobserved;
real obstacles exclusively in those rear sectors cannot be checked. The exact
physical source of the peripheral return has not been established.

The setting can be made explicit at startup with `-p forward_fov_deg:=260.0`.
Restart a running Python process to load the changed default. Restoring 270°
includes the suspect rear-edge returns again.

Parameters are read-only after startup. Restart to change them. There is no
reverse recovery: if all forward arcs are blocked, `no_safe_path` commands a stop.
Other stop reasons distinguish invalid data, stale timestamps and scan timeout.
Debug logs report front distance, `path` (checked arc distance), gap bearing,
steering and commanded speed.
Geometric stop logs retain the best checked distance instead of always reporting
zero, so clearance limits can be distinguished from missing scan data.

### Live RViz and recording

From a graphical terminal, start the diagnostic helper and RViz together:

```bash
./start_rviz_ftg.sh
```

The launcher sources ROS 2 Humble and this workspace automatically, using
`ROS_DOMAIN_ID=42` unless already set. It works from any directory when called
by its path. Bringup and the standalone planner must already be running.
Closing RViz or pressing Ctrl+C also stops the helper. Monitor options can be
passed directly, with optional RViz arguments after `--`:

```bash
./start_rviz_ftg.sh --output-file /home/kit/f1tenth_diagnostics/live_monitor.jsonl
```

To start them individually instead, use two terminals with the same ROS setup
and domain:

```bash
python3 src/gap_follow/scripts/gap_follow_visualizer.py --duration 1200 \
  --output-file /home/kit/f1tenth_diagnostics/live_monitor.jsonl
```

```bash
rviz2 -d /home/kit/f1tenth_ws/src/gap_follow/config/live_monitor.rviz
```

RViz needs a graphical desktop. The helper publishes diagnostic markers only;
it does not run the planner or send driving commands. Cyan points show the scan,
yellow shows the configured padded footprint, and the dashed orange arc
illustrates the requested steering. **That arc is not a collision-checked path.**
The display compares `/drive`, mux output and measured odometry, plus scan ages
and the last planner logs. Logs arrive at approximately 1 Hz and are not
synchronized to each displayed scan. The helper's geometry options must match
any planner footprint/wheelbase overrides. No SLAM map is needed.

For replayable raw data, record separately; the JSONL file contains summaries:

```bash
ros2 bag record -o /home/kit/f1tenth_diagnostics/track_run \
  /scan /drive /ackermann_cmd /teleop /joy /odom /rosout /tf /tf_static \
  /commands/motor/speed /commands/servo/position /sensors/core \
  /gap_follow_monitor/markers
```

Choose a new output directory each run and stop the recorder with Ctrl+C before
querying its SQLite database. Reading the active database can interfere with
this installed recorder's locking behavior. Note the time and intended turn
direction at each hesitation so the raw scans and control output can be compared.

Check command ownership with `ros2 topic info /drive -v`. The supplied
`wall_follow_pkg/launch/car_autonomy.launch.py` starts both wall-follow and safety
publishers on `/drive`. Running this script alongside wall-follow gives competing
autonomous commands. The existing safety node also publishes onto that same
topic; that arrangement does not provide a latched priority brake through the
mux. Its `brake()` assigns `last_break_time` while its hold logic reads
`last_brake_time`. Those separate integration issues have not been changed here.
Use one navigation controller and validate the car's independent stop/mux behavior.

## Geometry and tuning

All distances below are metres. Keep the existing hardware calibration. The
new planner's footprint and motion-model assumptions are separate from it.
The table records current source defaults. Startup overrides may differ; read
the active values with
`ros2 param dump /gap_follow_standalone` when comparing runs.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `wheelbase` | 0.33 | Rear axle to front axle; matches the VESC odometry config |
| `lidar_offset` | 0.27 | Matches the physical car's calibrated bringup transform; simulator override is 0.275 |
| `vehicle_front` | 0.43 | Rear axle to frontmost body point; initial estimate |
| `vehicle_rear` | 0.12 | Rear axle to rearmost body point; initial estimate |
| `bubble_radius` | 0.25 | Half vehicle width plus lateral clearance margin |
| `forward_fov_deg` | 260.0 | Usable scan sector, ±130°; excludes the obstructed peripheral region |
| `max_steering_angle` | 0.35 | Calibrated road-wheel angle limit, radians |
| `max_speed` | 1.5 | Speed ceiling; use 0.5 for initial physical tests |
| `max_lateral_accel` | 4.0 | Assumed lateral acceleration limit, m/s²; current source setting, requires grip validation |
| `braking_decel` | 2.5 | Assumed effective braking deceleration, m/s² |
| `reaction_time` | 0.30 | Allowance for sensing, computation and actuation, seconds |
| `max_acceleration` | 1.0 | Commanded speed increase limit, m/s² |
| `slow_distance` | 0.10 | Checked travel needed for full nominal cruise speed |
| `path_horizon` | 3.0 | Maximum distance evaluated along each arc |
| `comfort_clearance` | 0.35 | Maximum extra wall clearance rewarded by the score |
| `inner_wall_bias` | 1.0 | Inside-line preference strength, 0 to 1; 0 disables it |
| `inner_wall_clearance` | 0.10 | Preferred extra gap beyond the padded side, in metres; range 0 to 1 |

### Prefer the inside of a turn

Add these overrides to the planner's `--ros-args` on restart to set a moderate
inside-line preference:

```bash
-p inner_wall_bias:=0.7 -p inner_wall_clearance:=0.10
```

`inner_wall_bias=0.0` restores the original route ranking. Larger values up to
1.0 give more preference to a route closer to the requested inside gap.
`inner_wall_clearance` is additional space beyond the existing padded footprint,
including occupied-cell uncertainty; it is not distance from the car's centre
or a new collision margin. Lower values request a closer line. It is a soft
target, not a promise to maintain an exact wall distance.

The unbiased arc establishes turn direction. The preference requires a
turn with a supported nearby wall on its inside; it does not assume
every visible boundary is the central divider. Only measured side-wall cells
within 1 m of the padded side qualify. The score uses checked poses on the
immediate arc. All steering candidates are evaluated against the same arc checks.

Alternatives must preserve braking room for the original arc's speed ceiling
and may lose at most 0.10 m of checked travel. The preference
cannot override footprint, visibility, braking or curvature limits. A tighter
line may still require a lower corner speed. Straight corrections, opposing
turns and missing-wall cases retain the original ranking.
Keep the footprint dimensions calibrated; use these line-preference parameters
to change positioning. Source defaults may have been edited between runs, so
use explicit bias values when comparing laps.

**Parameter migration:** remove overrides for `planning_horizon`,
`branch_distance`, `crawl_speed`, `min_path_distance` and `stop_distance`; these
parameters were removed with continuation planning and the separate crawl
policy. `slow_distance` still refers to checked arc travel, not a front ray.

At the default steering limit, rear-axle turning radius is
`wheelbase / tan(max_steering_angle) ≈ 0.90 m`. Lowering clearance thresholds
cannot make a physically tighter turn possible. Use your established calibration
and test left and right hairpins from different entry positions. Do not increase
the steering limit just to hide poor entry positioning.

The speed bound solves `v * reaction_time + v² / (2 * braking_decel) <= available`
and also enforces `v² * abs(curvature) <= max_lateral_accel`. These depend on
actual traction, braking and actuator response. They use commanded speed, not
measured odometry. `min_speed` never overrides a lower safety bound. Acceleration
uses elapsed callback time, capped at 0.1 s; speed reductions are immediate.

Raising `max_speed` only raises a ceiling. With a 3 m checked horizon,
2.5 m/s² braking assumption and 0.30 s reaction allowance, even a
fully clear straight arc permits at most about **3.19 m/s**. At full 0.35 rad
steering, `max_lateral_accel=4.0` permits about **1.90 m/s**. The earlier
`max_lateral_accel=1.5` setting permitted about 1.16 m/s at that steering angle.
The 1.0 m/s² acceleration limit can lower these values further. A 10 m/s ceiling
therefore does not request 10 m/s on this course.

Improve completion and turn entry first, then compare lap time, measured speed,
slip and latency on repeat runs. Increase acceleration or lateral acceleration
only against measured traction and response. Higher straight speeds need more
observed, collision-checked travel: at the existing braking/reaction assumptions,
5 m/s needs 6.5 m of checked travel. Do not claim stronger braking or lower
latency merely to raise the computed speed. Longer horizons increase computation
and do not solve occluded turns; the dynamic simulator's aggressive settings did
not pass all delayed-scan corner tests (see `docs/race_validation.md`).

If a stop coincides with missing returns, inspect the scan values and the
driver's convention. Enable `inf_is_free` only if positive infinity means no
obstacle within `range_max`. NaN, negative infinity, zero and out-of-range finite
values stay invalid under either setting.

For repeatable tuning, record `/scan`, `/drive`, `/odom` and `/rosout`; compare
completion rate, minimum clearance, steering oscillation, computation latency
and lap time on the same courses. Establish reliable low-speed completion before
raising `max_speed`. Profile on the car: a 40 Hz scan offers 25 ms per callback.
The watchdog cannot interrupt a blocked callback or a failed process.

## Validation and limits

```bash
python3 src/gap_follow/scripts/gap_follow_standalone.py --self-test
python3 -m unittest discover -s src/gap_follow/test -v
```

The tests raycast static line-segment tracks and integrate bicycle motion,
checking the body against walls between control steps. They cover mirrored
smooth and square hairpins, displaced corridor entries, close obstacles, scan
ordering, invalid data and speed bounds. They do not establish performance on
the physical car: steering transitions, tire slip and scan distortion are not
simulated.

Current known completion failures are the narrowing rounded-apex entry and the
forward dead-end pocket with a side exit: the controller eventually stops in
both scenarios. Ordinary smooth hairpins, rectangular hairpins and doglegs pass
their targeted regressions. The full regression command above is expected to
report the known completion failures; the smoke test alone does not cover them.

The narrowing rounded-turn regression is synthetic, not a measurement inferred
from the distorted track photograph. A separately checked, bounded-curvature
reference proves a route fits. The controller takes a shallow approach that
eventually faces the outer wall with no forward turning room. Reposition before
the bend for another entry; no reverse recovery is enabled.

The opt-in ROS integration check uses unique topics, domain 217 and localhost
discovery, regardless of the shell's car-domain settings:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/gap_follow/test/ros_smoke_test.py
```

This remains a local reactive controller. Finite-resolution scans cannot certify
empty space between beams. The rear blind region is not fully observed, and
steering arcs assume their requested steering angle is achieved immediately.
It cannot guarantee a route out of arbitrary dead ends or optimal lap times on
every track. Checking only constant-steering arcs limits its ability to choose
entries that require a later steering change, especially at pockets and tight
turns. It does not know which unseen branch belongs to the racetrack. More
general recovery requires local memory or a route reference. Physical
steering-response and braking measurements must support the model before
increasing race speed.

Historical validation includes mirrored hairpins, opposing turns, a wide/narrow/
wide S bend, broad side recesses and forward dead ends with visible side exits.
An earlier single-arc controller failed the forward-pocket cases and one
opposing-turn direction; the former continuation planner completed those
geometric rollouts. Its replay of 30 recorded turn scans yielded moving crawl
commands in all 30, compared with 19 stops under an earlier hard threshold.
These results and the figure below predate the current simplification and do
not establish its completion rate or physical lap-time performance.

![Before and after trajectories in ideal bicycle simulations](docs/turn_comparison.png)

## Higher-speed simulation

The VESC motor limits do not configure F1TENTH Gym or this planner. The updated
`speed_max` converts to 20 m/s using the existing gain and offset, but the planner
still defaults to 1.5 m/s unless explicitly overridden. A speed ceiling also
does not override the braking, curvature or visibility limits.

The opt-in benchmark below runs the installed Gym directly, imports no ROS, and
cannot feed the physical car's command topics. It uses live simulated scans for
planning; maps and simulator poses are used only to evaluate the result.

```bash
python3 src/gap_follow/test/gym_race_validation.py \
  --track spielberg --geometry corrected --profile sim-race --max-speed 20 \
  --control-period 0.1 --scan-delay 0.1 --clip-sim-range-noise \
  --output /tmp/gap_spielberg
```

This command reproduces a **stress test, not a validated race configuration**.
The simulation-only profile uses acceleration/braking 9 m/s² and lateral
acceleration 8 m/s²; it does not write hardware settings. `--geometry corrected`
accounts for Gym's vehicle-center origin while the planner uses a rear-axle
origin. The explicit noise option clips finite simulator readings above the
declared maximum after Gym adds noise; physical scan validation is unchanged.

[Recorded results and limitations](docs/race_validation.md) predate this
simplification and concern the continuation planner. They include successful
ideal-scheduled laps and failed delayed/high-speed runs. In particular, the
3 m racing profile reached 4.95 m/s and completed an ideal-scheduled lap, but
stopped at a hairpin with 100 ms scan delay. Larger horizons reached 7.91 and
11.80 m/s and triggered Gym's collision detector. These results do **not**
establish reliable 20 m/s racing or performance on every track.

For the separate ideal bicycle-model stress cases:

```bash
python3 src/gap_follow/test/kinematic_race_validation.py --quick
# --full also exercises shorter control intervals and longer approaches.
```

## Racing algorithms considered

This implementation combines ideas suitable for live LiDAR without a prebuilt
map. It is a sampled local planner, not a full implementation of MPC, DWA,
LocalMap, or a learned racing policy.

| Approach | Relevant evidence and idea | Use here |
| --- | --- | --- |
| [Disparity Extender](https://nathanotterness.com/2019/04/the-disparity-extender-algorithm-and.html) | The author reports a 2019 F1TENTH win, but also documents wide-to-narrow turns misleading the farthest-ray strategy. | Retain disparity inflation for opening proposals; check candidate steering arcs before choosing a direction. |
| [Dynamic Window Approach](https://www.cs.cmu.edu/~dfox/abstracts/colli-ieee.abstract.html) | The original approach filters candidate motions by collision avoidance and stopping ability before scoring them. | Borrow braking-admissible motion selection, adapted to Ackermann arcs. We do not implement its full dynamic-state/reachable-velocity window. |
| [LocalMap racing](https://arxiv.org/html/2401.17732v1) | A scan-derived local corridor supports planning without a prebuilt map or global localization; the paper reports 8.8% faster laps than FTG in simulation. | Explicit boundary reconstruction remains outside this controller; its arc checks use current observed space. |
| [RaceMOP](https://arxiv.org/html/2403.07129v2) | Combines a base planner and learned residual decisions; the paper identifies a remaining simulation-to-real gap. | Retain body-aware geometric checks. No learned policy is deployed; it needs training, dynamic-state inputs and real-car validation. |

For competition tuning, compare completion rate and lap-time distributions on
the same courses at the same speed ceiling. Include both directions, displaced
entries, narrow exits and misleading recesses. Increase speed only after the
planner consistently chooses the correct route; a higher ceiling cannot repair
a bad entry line. Use the validation tools' callback timings to assess the
actual host and profile; fixed simulated timesteps do not prove that the Python
controller can meet a 40 Hz deadline.

References: [F1TENTH gap-following lab](https://github.com/f1tenth/f1tenth_lab4_template)
and the [original disparity extender explanation](https://nathanotterness.com/2019/04/the-disparity-extender-algorithm-and.html).
