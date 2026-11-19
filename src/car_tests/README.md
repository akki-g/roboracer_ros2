# Car measurements

These runners publish `/drive` and record bags for grip, straight acceleration/braking/speed calibration, steering response, and guarded policy speed sweeps. Run on the car after native bringup; analyse on the Mac. They add no ROS dependencies beyond packages already used by the stack. Mapping and rollover testing are outside this package.

## Verified stack behavior — read before driving

**RB is NOT a strict deadman in the existing stack.** With no buttons pressed, `default` repeatedly publishes zero speed and steering on `teleop`. However, *any* pressed button disables `default`, including buttons other than RB/LB. Thus another button, or loss of joystick messages followed by the teleop timeout, can allow `/drive` through without RB. Every runner here independently requires fresh `/joy`, an observed release followed by a new button-5 press, and continued RB hold. Release during a run latches an abort; another press cannot restart that process. LB during a run also aborts the test; teleop itself still has higher priority.

Evidence in the existing checkout:

| Behavior | Source (`file:line`, relative to CAR root) |
| --- | --- |
| Joystick autorepeat configured at 20 Hz; default zero axis mappings; LB=4; RB=5 publishes `/dev/null` | [joy_teleop.yaml:6](../f1tenth_system/f1tenth_stack/config/joy_teleop.yaml#L6), [default:11](../f1tenth_system/f1tenth_stack/config/joy_teleop.yaml#L11), [LB:29](../f1tenth_system/f1tenth_stack/config/joy_teleop.yaml#L29), [RB:44](../f1tenth_system/f1tenth_stack/config/joy_teleop.yaml#L44) |
| No deadman config becomes `default`; default active only when sum of all buttons is zero | [joy_teleop.py:86](../f1tenth_system/teleop_tools/joy_teleop/joy_teleop/joy_teleop.py#L86), [joy_teleop.py:112](../f1tenth_system/teleop_tools/joy_teleop/joy_teleop/joy_teleop.py#L112) |
| Axis mappings publish on every active joy callback, without message-value debounce | [joy_teleop.py:201](../f1tenth_system/teleop_tools/joy_teleop/joy_teleop/joy_teleop.py#L201), [publish:248](../f1tenth_system/teleop_tools/joy_teleop/joy_teleop/joy_teleop.py#L248), [callback:411](../f1tenth_system/teleop_tools/joy_teleop/joy_teleop/joy_teleop.py#L411) |
| `/drive` priority 10, `/teleop` priority 100, both expire after 0.2 s | [mux.yaml:3](../f1tenth_system/f1tenth_stack/config/mux.yaml#L3) |
| Mux timeout masks old inputs; it does **not** itself publish a zero when all inputs expire | [topic_handle.hpp:105](../f1tenth_system/ackermann_mux/include/ackermann_mux/topic_handle.hpp#L105), [mask:181](../f1tenth_system/ackermann_mux/include/ackermann_mux/topic_handle.hpp#L181), [callback-only publish:190](../f1tenth_system/ackermann_mux/include/ackermann_mux/topic_handle.hpp#L190), [diagnostics timer:100](../f1tenth_system/ackermann_mux/src/ackermann_mux.cpp#L100) |
| Actual mux output is `/ackermann_cmd`; bridge subscribes there. Bringup's `ackermann_cmd_out` remap does not match that publisher | [ackermann_mux.cpp:90](../f1tenth_system/ackermann_mux/src/ackermann_mux.cpp#L90), [ackermann_to_vesc.cpp:61](../f1tenth_system/vesc/vesc_ackermann/src/ackermann_to_vesc.cpp#L61), [bringup_launch.py:121](../f1tenth_system/f1tenth_stack/launch/bringup_launch.py#L121) |
| ERPM = 4675 × speed + 220; servo = −0.9 × angle + 0.552; servo limits 0.15–0.95 | [vesc.yaml:5](../f1tenth_system/f1tenth_stack/config/vesc.yaml#L5), [servo limits:24](../f1tenth_system/f1tenth_stack/config/vesc.yaml#L24), [bridge conversion:69](../f1tenth_system/vesc/vesc_ackermann/src/ackermann_to_vesc.cpp#L69) |
| Smoother parameters are 9.0 m/s² and 3.2 rad/s, but the smoother node is **disabled** in bringup | [vesc.yaml:45](../f1tenth_system/f1tenth_stack/config/vesc.yaml#L45), [bringup_launch.py:136](../f1tenth_system/f1tenth_stack/launch/bringup_launch.py#L136) |
| `/odom` speed uses `(state.speed + offset)/gain` in this checkout; yaw is calculated from servo command, not a measured yaw sensor | [vesc_to_odom.cpp:100](../f1tenth_system/vesc/vesc_ackermann/src/vesc_to_odom.cpp#L100), [yaw:105](../f1tenth_system/vesc/vesc_ackermann/src/vesc_to_odom.cpp#L105) |

The configured default joystick path therefore blocks `/drive` when *all* buttons are released and fresh joy messages continue. Holding only RB permits `/drive` after teleop's 0.2 s timeout. This initial gate delay can appear in a straight run's command-to-response fit. Driver firmware timeout/braking behavior is not established by the ROS mux code.

Telemetry names and types (bringup gives these nodes no namespace or telemetry remaps: [bringup_launch.py:86](../f1tenth_system/f1tenth_stack/launch/bringup_launch.py#L86)):

| Topic | ROS type | Existing source |
| --- | --- | --- |
| `/sensors/core` | `vesc_msgs/msg/VescStateStamped` | [vesc_driver.cpp:81](../f1tenth_system/vesc/vesc_driver/src/vesc_driver.cpp#L81); `state.speed` receives raw RPM at [172](../f1tenth_system/vesc/vesc_driver/src/vesc_driver.cpp#L172) |
| `/commands/servo/position` | `std_msgs/msg/Float64` | [ackermann_to_vesc.cpp:58](../f1tenth_system/vesc/vesc_ackermann/src/ackermann_to_vesc.cpp#L58) |
| `/sensors/servo_position_command` | `std_msgs/msg/Float64` | [vesc_driver.cpp:87](../f1tenth_system/vesc/vesc_driver/src/vesc_driver.cpp#L87), clipped command echo at [334](../f1tenth_system/vesc/vesc_driver/src/vesc_driver.cpp#L334); **not measured servo position** |
| `/commands/motor/speed` | `std_msgs/msg/Float64` | [ackermann_to_vesc.cpp:57](../f1tenth_system/vesc/vesc_ackermann/src/ackermann_to_vesc.cpp#L57) |
| `/odom` | `nav_msgs/msg/Odometry` | [vesc_to_odom.cpp:75](../f1tenth_system/vesc/vesc_ackermann/src/vesc_to_odom.cpp#L75) |
| `/policy_node/debug` | `std_msgs/msg/String` | [policy_node.py:129](../rl_policy/rl_policy/policy_node.py#L129) |

`vesc_msgs` is already a dependency of the driver ([vesc_driver/package.xml:24](../f1tenth_system/vesc/vesc_driver/package.xml#L24)). No new apt dependency is needed for this package. The sensor config does not prove the actual scan rate/geometry; it requests `cluster: 1`, `skip: 0`, and ±3.14 rad ([sensors.yaml:3](../f1tenth_system/f1tenth_stack/config/sensors.yaml#L3), [18](../f1tenth_system/f1tenth_stack/config/sensors.yaml#L18)). Safety uses each received scan's angles; check live rate and beams on the car.

## Setup and common limits

Build: `tools/car_native.sh build`

In a separate terminal: `source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch f1tenth_stack bringup_launch.py`

The runner sources that same native install. It requires fresh `/scan`, `/odom`, `/sensors/core`, and `/joy`, all joystick buttons released, a `/drive` subscriber, and no competing `/drive` or private-policy publisher. Do not start a second controller while it runs. Start low, on the intended floor, with a clear route and an operator holding the controller. Check the LiDAR plane sees the wall/obstacles and the mounting is secure.

Defaults are 50 Hz `/drive`, maximum requested speed 3.5 m/s, steering ≤0.42 rad, maximum 120 s **per active run**, stale scan/core/odom/joy limit 0.25 s (monotonic receive time), anything closer than 0.35 m stops, and (for straight/steer) minimum valid forward range over ±10° must exceed `v*0.25 + v²/(2*3.0) + 0.5` metres. Invalid forward data stops. Both distances ignore LiDAR noise: readings under 5 cm (`--short-reject`) are dropped, and an obstacle must span 3 adjacent beams (`--cluster-beams`, 0.75°). The car's UST-10LX gives about 3 spurious short readings per scan, which tripped a plain any-beam minimum in 20% of scans of the 2026-11-17 recordings. `v` is the maximum of requested speed, odom speed, and conservative absolute raw-ERPM conversions using both offset signs. The straight/steer distance is measured **from the LiDAR**; the 0.265 m sensor offset is not extra clearance. Grip instead checks scan points against a finite rear-axle arc of length `stop_distance(v)`, using `tan(current_steering)/0.33` curvature and transforming laser points by +0.265 m. Its swept tube has radius `--car-half-width + --side-margin` (0.15 + 0.15 m by default), with endpoint caps. It retains `min_clearance`, drops short noise, requires `cluster_beams` adjacent obstacle hits, and fails closed when the approaching route is outside the scan or has a broad dropout. Isolated rejected beams can use a nearby valid cluster within six beams (twice `cluster_beams`). During a command transition it checks both the previous and new steering arcs. Steering is commanded, not measured; zero-speed aborts still command zero steering, so the actual steering-return/braking path and footprint need a car check. Straight and steering-step tests retain the existing straight-cone guard: rapidly changing steering in the latter makes a constant-curvature forecast less defensible. The 0.5 m margin must cover body overhang and stopping uncertainty. Zero speed uses the same bridge speed-command path as the policy, not a separate motor-current brake ([policy_node.py:303](../rl_policy/rl_policy/policy_node.py#L303), [ackermann_to_vesc.cpp:69](../f1tenth_system/vesc/vesc_ackermann/src/ackermann_to_vesc.cpp#L69)). A zero speed command still maps to 220 ERPM in the checked-in bridge; verify that the real car stops.

The commanded lateral cap is `v²*abs(tan(delta))/0.33 <= 5.0 m/s²`. The tip model is rough: SIM [`rl/car_params.py:127`](../../../f1tenth_gym_jax/rl/car_params.py#L127) has `TIP_ANGLE_DEG=(45,50)`, `TIP_ASYM_DEG=1.5`, `TIP_DYNAMIC=(0.75,0.95)`, `GRAVITY=9.81`. Its low end is `tan(45−1.5°)*0.75*9.81 ≈ 6.98 m/s²`; that is an estimate, not a verified safe dynamic threshold. The measured LiDAR mount provenance is at SIM [`rl/car_params.py:67`](../../../f1tenth_gym_jax/rl/car_params.py#L67). These tests do not alter those constants or test rollover.

Release RB to abort. Ctrl-C or SIGTERM also stops; the node keeps DDS alive and publishes zero for at least 0.5 s before exiting. Faults latch; restart the command after fixing the cause. A normal straight/steering run commands zero, waits for reported stop for 0.5 s, then prints `between`: only then release RB, reposition, and press RB for the next target. Hold RB through braking. LB still selects higher-priority manual teleop; it is not a substitute for releasing RB and all other buttons to stop.

All options: `tools/car_tests.sh straight --help`

Parameters are validated before any process starts; safety caps are never raised by the profile. `--max-speed`, `--steer-limit`, `--lat-acc-cap`, `--max-duration`, `--rate-hz`, `--cone-deg`, `--latency`, `--a-brake`, `--margin`, `--min-clearance`, `--stale-s`, `--car-half-width`, and `--side-margin` are common (the last two affect grip only). Stale timeout cannot exceed 0.25 s and steering limit cannot exceed 0.42 rad. Braking distance is a conservative planning model whose parameters still need measurement on the car.

The node remains unarmed until a recorder subscribes to both `/drive` and `/car_test/events`; wait for `Recorder ready`, then release and press RB. Direct node execution also needs a recorder. The recorder uses sqlite3, an 8 MiB cache (up to two buffers), and names bags `~/bags/<test>_<key params>_<YYYYmmdd_HHMMSS>`. `--bag-root PATH` changes the destination. It records `/scan /odom /drive /ackermann_cmd /joy /sensors/core /commands/motor/speed /commands/servo/position /sensors/servo_position_command /car_test/events /car_test/policy_drive /policy_node/debug`; nonexistent topics are discovered if they appear. Events are JSON `std_msgs/String` containing test, phase, run, targets, stop reason, and all params. A one-second heartbeat repeats current state for late discovery. Ctrl-C stops the publisher first, then the policy if present, and closes the bag with SIGINT. Forced TERM/KILL after a shutdown timeout emits a warning that bag reindexing may be needed.

## T1 — grip circle

Measures apparent tyre/floor friction from independent scan-derived rear-axle speed, path curvature, and yaw rate. A fixed steering angle starts at 1 m/s and ramps at 0.05 m/s each second, ending at 3.5 m/s or the lateral cap, then commands zero. At 0.30 rad the default lateral cap is reached at about 2.31 m/s. With 5 m/s² the maximum identifiable friction is only around `5/9.81 ≈ 0.51`: higher-grip floors may show **no slide onset**. Keep that censored result; do not raise the cap just to force sliding.

Floor space: rear-axle circle diameter `2*L/tan(abs(delta))`, about 2.13 m at 0.30 rad, plus car footprint and clearance. Plan at least roughly 4×4 m unobstructed, and more if steering gain is smaller than expected. The path-aware and all-beam minimum-clearance stops remain active throughout the turn. Setup: leave the centre and full circle clear, inspect tyres, place on the actual test floor, and choose left/right. Watch the speed ramp, phase output, and any outward drift. Release RB immediately if the path becomes unsafe.

For a duct boundary, use an inner radius of at least roughly 2 m, leaving room for the car circle, body, and clearance. Deliberately avoid a round ring: use an oval with a flat side or kink, and/or place two or three boxes at uneven positions against the inside boundary. Boxes must be at least 20 cm tall and outside the swept driving path. A perfectly round ring concentric with the car circle gives scan matching no independent rotation reference; low residuals can coexist with unobservable yaw and position. The UST-10LX scan plane is about 14 cm high, tilted about 0.7° up: 30.5 cm ducts are visible, while lower obstacles may not be. Inspect visibility at every planned range.

Left: `tools/car_tests.sh grip --direction left --delta 0.30 --v-start 1.0 --ramp 0.05 --v-cap 3.5`

Right (separate bag): `tools/car_tests.sh grip --direction right --delta 0.30 --v-cap 3.5`

Analyse: `uv run --frozen python -m rl.car_tests_fit grip ~/bags/grip_left_d0.3_v3.5_TIMESTAMP --calibration straight.json --output grip.json`

Analysis uses robust point-to-line scan odometry with keyframes and Hessian degeneracy checks. It converts the laser path to the rear axle, fits circles over sliding windows and laps, estimates low-speed steering as `atan(0.33/R)`, and separately reports left/right gain. True speed is the local-polynomial path arc-length rate; lateral acceleration is reported as both `v²/R` and `v*abs(yaw_rate)`. Both yaw and curvature must remain below 0.85 of the low-speed kinematic baseline for ≥0.5 s to mark apparent slide onset. Degenerate or discontinuous trajectories cannot produce a grip calibration. `--calibration` is unnecessary for grip v2. It reports apparent `mu` at observed onset, otherwise a conditional lower bound, never an upper friction limit. Changing gain, lateral sideslip, moving obstacles, and matching error can mimic slip; repeat both directions and inspect the physical behavior.

## T2 + T3 — straight runs toward a flat wall

Measures true speed calibration, acceleration versus speed, command-response delay, braking deceleration, and stopping distance from the zero-speed command. Point straight at a large flat wall normal to the car. Use a lane with side clearance, no people, and no shiny/transparent target. Start about 5–6 m away for low-speed runs, head-on to the flat wall, with nothing else entering the lane. Measure the starting distance with a tape **from the LiDAR front face**, not the rear axle or sensor centre. The front face is 0.025 m ahead of the centre; the centre is 0.265 m ahead of the rear axle (mount measurement confirmed 2026-11-19). Keep static asymmetric side features for full scan observability; a featureless corridor cannot constrain travel along it. Default targets are 1.5, 2.0, 2.5, 3.0 m/s, one target per RB press. Each run steps to its target, waits for reported speed to reach 90% (4 s timeout), holds 1 s, optionally cruises longer, then commands zero and waits for stop.

Initial wall distance must exceed `v²/(2*accel_estimate) + v*(hold_s+cruise_s) + stop_distance(v)`. Default acceleration estimate is 2 m/s². This is a feasibility estimate, not a promise the target is reachable; failure to reach the target or forward clearance becoming insufficient aborts. At 3 m/s with a 1 s hold, the initial requirement is >8.0 m from the sensor, plus room behind for setup. A 0.5 s optional cruise raises it to >9.5 m, close to a nominal 10 m sensor's range. Use lower speeds/shorter holds for shorter lanes. Watch the measured starting distance, acceleration phase, and brake marker. Release RB to abort; never rely solely on the wall calculation.

Default: `tools/car_tests.sh straight --targets 1.5 2.0 2.5 3.0 --hold-s 1.0`

Calibration cruise: `tools/car_tests.sh straight --targets 1.5 2.0 --hold-s 1.0 --cruise-s 1.0`

Analyse: `uv run --frozen python -m rl.car_tests_fit straight ~/bags/straight_v1.5-2.0-2.5-3.0_TIMESTAMP --output straight.json`

Analysis v2 averages the verified stationary starting scans before each run, then measures full rear-axle motion against scan keyframes. Centered local cubic derivatives yield path speed, acceleration, lateral drift, and braking yaw; stopping distance integrates true path speed from the zero command. Steady holds fit true speed versus odom and raw ERPM versus true speed, with a whole-run distance-integral cross-check. A single-speed hold only supports a conditional ERPM gain with the configured offset fixed. The old ±5° head-on wall estimator remains an explicitly labelled cross-check; it cannot establish lateral drift or yaw in unobservable geometry. Stand checks are excluded from calibration. Delay estimates include transport, lag, and smoothing; boundary actuator fits produce no parameter suggestions. The report never modifies parameters.

Add `--start-distance 5.5` (metres) to compare the averaged reference range minus 0.025 m with the tape measurement; optionally add `--end-distance`. For several bags use repeated `--start-distance BAG_NAME=5.5` entries (and likewise for end distance). A single value applies to every run in each selected bag, so use it only when those physical tape distances are the same. Different run starts within one bag need separate `--start`/`--duration` analyses. A disagreement can indicate the wrong wall, alignment, or range scale; one measurement alone does not identify which. No tape distance has been inferred from the bag.

## T6 — left/right steering steps

Measures steering delay, gain and rate separately by direction and speed. Default speeds: 1.0, 1.5, 2.0 m/s, one speed per RB press. After reaching speed, each delta 0.1, 0.2, 0.3, 0.42 rad follows centre → left → centre → right → centre, with 0.8 s steps and 0.8 s centre settling. Combinations above the lateral cap are skipped and listed in the start event (at 2 m/s, 0.42 rad is skipped).

Needs a **long, wide open lane**, with recognizable stationary LiDAR geometry, not a narrow hallway: turns displace the car sideways. Rough path length at 1.5 m/s is ~20 m for the full four-delta sequence, before braking; at 2 m/s the three allowed deltas take ~21 m. Inspect the entire swept route and leave stopping margin. If space is limited, lower speed and test one delta per bag with longer steps; analysis needs at least three usable 2 s windows per direction, so repeat measurements/lengthen steps for identifiable fits. Forward stopping and any-beam clearance remain active during turns and can legitimately end a run. Watch both turn directions and emerging drift; release RB to abort.

Default: `tools/car_tests.sh steer --speeds 1.0 1.5 2.0 --deltas 0.1 0.2 0.3 0.42`

Analyse: `uv run --frozen python -m rl.car_tests_fit steer ~/bags/steer_v1.0-1.5-2.0_d0.1-0.2-0.3-0.42_TIMESTAMP --calibration straight.json --output steer.json`

Analysis uses LiDAR yaw, not odometry yaw, and `car_fit.fit_steering`, grouped by measured speed and command direction, including return-to-centre response. Existing `car_fit.LAT_ACC_MAX=4.0` excludes likely slipping data ([SIM rl/car_fit.py:201](../../../f1tenth_gym_jax/rl/car_fit.py#L201)), even though the maneuver cap is 5.0. Fits with high residual or gain at a search boundary are retained as diagnostics but excluded from suggestions. `--speed-bins 1.0 1.5 2.0` changes analysis bins; ±0.28 m/s bands are used.

## T5 — policy sweep / crash-or-abort bags

Measures requested versus reported speed and what happened before an abort or possible crash. Use the intended, already clear driving course with escape space and a controller operator. This wrapper does not map the course. Begin at the lowest cap, inspect each run before proceeding, and hold RB only when ready. Do not deliberately continue into obstacles to create crashes.

Run: `tools/car_tests.sh policy-sweep --caps 1.5 2.0 2.5 3.0`

Equivalent bash wrapper: `tools/policy_sweep.sh --caps 1.5 2.0 2.5 3.0`

Explicit model: `tools/policy_sweep.sh --model /absolute/path/model.npz --caps 1.5 2.0`

Default model: `car_v2_25hz_u1064.npz` from the **installed** `rl_policy` share directory. For each cap the wrapper prompts, starts `policy.launch.py mode:=car max_speed:=CAP policy_path:=MODEL record:=false drive_topic:=/car_test/policy_drive`, and records its own model/cap-named bag. The new launch argument keeps normal launches on `/drive`, while sweep policy requests go through `policy_guard`, the only sweep publisher on `/drive`. Guard clipping, deadman, stale-topic, duration, and lateral limits apply. The clearance and forward-stopping-distance stops do **not**: on a walled track the policy passes ducts closer than 0.35 m and has walls inside the stopping distance in every corner (in the 2026-11-17 policy bags they would have tripped in 14% and up to 75% of moving scans), so the speed cap and RB are the protection here. Thus these are guarded policy runs; `/car_test/policy_drive` preserves the original policy requests for comparison.

Ctrl-C cleanly ends that cap, including both the guard and policy, then the next prompt appears. Enter advances only after inspection/repositioning; `q` exits. Release RB aborts and latches the current run. Ctrl-C at a prompt exits the whole sweep. Each cap has its own bag. A process failure stops its companions. The guard's 120 s default run limit can be changed with `--max-duration`.

Analyse: `uv run --frozen python -m rl.car_tests_fit policy-sweep ~/bags/policy-sweep_MODEL_cap1.5_TIMESTAMP ~/bags/policy-sweep_MODEL_cap2.0_TIMESTAMP --calibration straight.json --output policy-sweep.json`

Per cap: peak speed, command-versus-speed error, guard abort/RB release, and a heuristic possible crash (speed above 0.7 to below 0.2 m/s within 0.35 s with range <0.35 m), with the preceding 2 s of speed/commands/ranges and time within run. Several labels may describe one physical incident. A possible crash is not proof of impact; an RB release is not a crash. Wheel-derived speed is only as accurate as calibration and traction.

## Offline verification and needs a car check

Mac core/supervisor tests: `cd /Users/akshatguduru/Desktop/CARVEL/f1tenth_gym_jax && UV_CACHE_DIR=/private/tmp/car-tests-uv-cache uv run --frozen pytest -q /Users/akshatguduru/Desktop/CARVEL/roboracer_ros2/src/car_tests/test`

Analysis tests: `uv run --frozen pytest -q rl/tests/test_car_tests_fit.py rl/tests/test_scan_odometry.py`

Full RL suite: `uv run --frozen pytest -q rl/tests`

See SIM `rl/car_test_results/verification.md` for actual results and real-bag findings. The hand-built CDR fixtures cover VESC, Joy, String, alignment, invalid encoding, split bags, and changing joystick sizes. Actuator/scan synthetic bags test scale/gain, acceleration/braking/delay, left/right steering, friction-onset/no-onset, and abort heuristics. They validate the estimators against specified models, not physical tyre behavior.

**Needs a car check:** native colcon build and installed launch argument; live topic names/types/rates and DDS QoS; 50 Hz publication under KV260 CPU load; joystick disconnect and all-button behavior with wheels lifted; stop-command behavior including the 220 ERPM offset and actual motor timeout; signal handling with the real ROS CLI/recorder and metadata closure; LiDAR forward alignment, effective range, nose overhang and floor visibility; conservative braking/latency values on the actual floor; sufficient floor/lane space; model loading and guarded private output; fitted delay versus inter-topic clock/transport latency. A killed process or power failure cannot execute the 0.5 s stop tail. Inspect the real hardware fail-safe before driving.
