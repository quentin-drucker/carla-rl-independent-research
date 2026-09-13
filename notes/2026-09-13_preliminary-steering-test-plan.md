# Preliminary Steering Test Plan

> **Planning date:** September 13, 2026
>
> **Immediate reporting deadline:** September 14, 2026
>
> **Working branch:** `experiment/preliminary-steering-test`
>
> **Status:** Design and test plan; no learned steering implementation yet

## Purpose

Establish the smallest safe, interpretable test of lateral control in the existing CARLA pedestrian-hazard framework. The test will reuse last semester's straight highway route, classical speed controller, scripted pedestrian, and experiment infrastructure. It will add a desired lateral path offset before attempting reinforcement learning.

The first research question is deliberately narrow:

> Can the ego vehicle follow a commanded lateral offset on the existing straight route, hold that offset long enough to pass a hazard, and return smoothly to its original path?

This is an engineering feasibility test, not yet evidence that an RL policy makes good evasive decisions.

## Why reuse the existing straightaway

The previous highway scenario is the appropriate starting point because it holds most of the system constant while introducing one new behavior: lateral motion. Its existing route, speed control, pedestrian timing, sensors, and scenario configuration provide known reference points.

Starting on a curve, in traffic, or with several pedestrian behaviors would make it difficult to distinguish steering problems from perception, route-planning, or scenario problems.

Pedestrian trigger timing should eventually vary because it changes whether braking or steering is useful. It should not vary in the first steering run. The stages below introduce that variation only after the lateral controller works in isolation.

## Scope decisions

### Initial control boundary

The eventual RL action will be:

```text
[brake target, desired lateral offset]
```

- Brake target remains in `[0, 1]`.
- Desired lateral offset represents where the policy wants the vehicle to travel relative to the original planned route.
- The existing PI controller continues to own throttle and cruise speed.
- The existing low-level heading controller converts the offset target into CARLA's raw steering command.
- The learned policy will choose the maneuver; it will not initially control every wheel-steering correction.

For preliminary testing, the lateral offset will be scripted rather than learned. This establishes that the maneuver is mechanically possible before reward design and training are introduced.

### Initial scenario

- Map and route: retain the current `Town04_Opt` straight highway route.
- Ego: retain the current vehicle and classical cruise controller.
- Pedestrian: retain the scripted near-cross pedestrian that remains in the ego path.
- Traffic: none for the first pedestrian test.
- Weather and friction: fixed defaults.
- Sensors: retain the current LiDAR and collision sensor setup.
- Steering direction: test one direction first, then mirror it to check left/right behavior.

### Explicit non-goals for this first test

- No RL training.
- No reward redesign.
- No learned throttle or acceleration maneuver.
- No moving traffic or ethical collision tradeoff.
- No new simulator, map, sensor modality, weather, or friction study.
- No claim of generalization.
- No attempt to replace the existing route follower with raw learned steering.
- No modification or overwrite of Spring 2026 result artifacts.

## Existing integration point

`src/lane_follow.py` currently performs this sequence inside `lane_follow_step()`:

1. Find a lookahead target on `route_points_world`.
2. Compute the angle from the ego vehicle to that target.
3. Convert heading error into `steer_cmd` in `[-1, 1]`.
4. Apply `steer_cmd` together with throttle and brake through `carla.VehicleControl`.

The preliminary design should insert the lateral-offset target between steps 1 and 2. The low-level heading calculation can then steer toward an offset version of the route rather than replacing the steering controller outright.

Before implementation, define the sign convention using the route direction—for example, negative for route-left and positive for route-right—and verify it visually. Do not assume CARLA world `x` or `y` corresponds consistently to left or right.

## Test sequence

### Test 0: Preserve the known baseline

Run one windowed episode with no steering modification and visually confirm:

- The ego follows the original route.
- The pedestrian triggers at the expected location.
- The current braking behavior still operates.
- Existing debug points and telemetry appear normally.

This is only a regression reference. Do not use the evaluator's current `full_stop` count as scientific ground truth because far pedestrian clearance may still be mislabeled as a stop.

### Test 1: Lateral shift without a pedestrian

At a low target speed, command a small fixed lateral offset on the straightaway, hold it briefly, and return the target to zero.

Start conservatively:

- One direction only.
- Small offset before attempting a full lane-width movement.
- Smoothly ramp the desired offset instead of changing it instantaneously.
- Keep the normal route centerline visible for comparison.

Success means the vehicle shifts in the intended direction, remains stable, and returns without oscillation, road departure, or emergency braking.

### Test 2: Mirrored lateral shift

Repeat Test 1 in the opposite direction. This catches sign, map-orientation, and route-normal mistakes.

### Test 3: Fixed pedestrian encounter

Use one fixed ego speed and one generous pedestrian trigger time. Script the lateral maneuver; do not use RL yet.

Success means:

- No pedestrian collision.
- The ego stays on drivable road.
- The ego obtains positive clearance around the pedestrian.
- The ego returns toward its original route after passing.
- The maneuver does not create unstable steering or a throttle/brake conflict.

This empty-road case demonstrates mechanics only. It does not demonstrate intelligent maneuver selection because there is no competing hazard.

### Test 4: Small timing sweep

Only after Tests 1–3 pass, repeat the same scripted maneuver at approximately three pedestrian trigger times: early, intermediate, and late.

The purpose is to observe how available maneuver time affects clearance and stability. It is not yet a training distribution or full parameter sweep.

### Later test: Constrained escape

Add a road boundary or stationary blocking vehicle on one side. This produces the first meaningful decision case: brake, move toward the clear side, or recognize that no safe lateral maneuver is available.

This is outside the immediate preliminary test unless progress is unexpectedly fast.

## Preliminary measurements

Record these values even for scripted tests so they can later become shared evaluation metrics:

- Collision and collided actor type.
- Explicit termination reason.
- Ego speed at pedestrian trigger.
- Requested lateral offset.
- Actual lateral or cross-track position relative to the original route.
- Applied `steer_cmd`.
- Heading error and yaw rate if available.
- Minimum pedestrian clearance.
- Whether the vehicle left the drivable road or crossed an unsafe boundary.
- Maximum lateral displacement.
- Time to reach the requested offset.
- Time and distance required to return to the route.
- Steering rate, lateral acceleration, and qualitative oscillation notes.

The existing `cte_m` measures distance to the lane-center waypoint without a signed direction. It can help detect displacement but is insufficient by itself for a left/right target controller. A signed route-relative lateral error will eventually be needed.

## Pass/fail criteria for the preliminary proof of concept

The preliminary steering mechanism passes when:

1. Zero requested offset reproduces normal route following.
2. Positive and negative scripted offsets move the vehicle in the intended directions.
3. The vehicle can hold a small nonzero offset without obvious oscillation.
4. Returning the requested offset to zero returns the ego smoothly toward the route.
5. A fixed, generously timed pedestrian encounter can be cleared without collision or road departure.
6. Logs distinguish pedestrian collision, safe clearance, road departure, route recovery, and timeout.

Avoid claiming success from a single collision-free run. Repeat each deterministic case several times or note any simulator variation.

## Stop conditions

Stop and diagnose before increasing speed, offset, or scenario variation if any of the following occurs:

- Steering oscillates or saturates repeatedly.
- Offset direction changes with route orientation.
- The LiDAR corridor continues to follow the original centerline and loses the hazard in a way that changes braking unexpectedly.
- The ego leaves the drivable road.
- The classical controller immediately cancels the commanded offset.
- The route-recovery command causes a second abrupt maneuver.
- Outcome logging cannot distinguish safe pedestrian clearance from an actual vehicle stop.

## Work for the September 13 afternoon

Given the short time before Monday's report, today's achievable deliverable is design validation rather than a working RL controller.

- [x] Choose maneuver-level lateral-offset control instead of raw learned wheel steering.
- [x] Confirm the existing steering integration point in `lane_follow_step()`.
- [x] Create a feature/experiment branch from `develop`.
- [x] Write this preliminary test plan.
- [x] Run the unmodified scenario once in windowed CARLA and record whether the current route, pedestrian, LiDAR, and braking behavior look correct.
- [ ] Write a short Week 1/Week 2 tracker update using the language below.

Do not rush an implementation tonight solely to have a code change for the report. A bounded architecture decision, code-path inspection, and staged validation plan are meaningful research progress.

## Tracker-ready progress update

> Scoped the first steering extension of the CARLA pedestrian-hazard environment. I decided to preserve the existing low-level route follower and initially represent lateral control as a desired offset from the planned route, while the agent will later choose brake and lateral-offset commands. I identified the steering integration point in `lane_follow_step()` and designed a staged validation sequence: first test a scripted lateral shift and route recovery without a pedestrian, then repeat it around one fixed pedestrian encounter, and only afterward vary pedestrian trigger timing. This isolates steering mechanics before RL training and before adding blocked lanes or traffic.

## Next implementation step

After this plan is reviewed, implement the smallest deterministic interface that accepts a scripted lateral offset, transforms the route lookahead target along the route-relative left/right normal, and reports both requested offset and signed lateral error. Keep it separate from the SAC action space until Tests 1–3 pass.

## September 13 implementation record

The first deterministic interface and Tests 1–2 were completed on
`experiment/preliminary-steering-test`.

### Added implementation

- `src/route_lateral_control.py`: pure route-relative geometry with a documented
  sign convention (`+right`, `-left`).
- `lane_follow_step(..., lateral_offset_m=0.0)`: optional lateral target; the
  zero default preserves existing callers.
- Lateral telemetry: requested offset, signed actual offset, and offset error.
- `src/test4___scripted_lateral_offset.py`: low-speed, no-pedestrian manual test
  that ramps away from the route, holds, and recovers.
- `tests/test_route_lateral_control.py`: offline geometry tests.

### Verification results

Offline verification:

- Six route-geometry unit tests passed.
- Modified Python files compiled successfully.
- The manual runner's command-line interface loaded successfully.

Live CARLA verification at 15 mph:

| Test | Requested peak | Observed peak | Peak steer | Final route offset | Result |
|---|---:|---:|---:|---:|---|
| Route-right | +0.750 m | +0.717 m | 0.032 | +0.011 m | Pass |
| Route-left | -0.750 m | -0.717 m | 0.032 | -0.011 m | Pass |

The close mirror symmetry supports the route-relative sign convention, and
both tests returned to substantially less than the preliminary 0.25 m recovery
tolerance.

Existing-controller regression check:

- Used the original pedestrian scenario runner without supplying the new
  optional offset.
- Configuration: 15 mph, 60 m encounter, near crossing, 4.0 s trigger TTC,
  `step_constant` braking.
- Hazard triggered at 14.9 mph and 4.00 s nominal TTC.
- Ego recorded a full stop 0.52 s after hazard braking with no collision.

This verifies backward compatibility for one representative live scenario. It
does not repair or supersede the known evaluation/outcome limitations in the
master research summary.

### Next research test

Proceed to Test 3: connect a scripted lateral-offset schedule to one fixed,
generously timed pedestrian encounter. Before judging avoidance, decide how the
LiDAR corridor should relate to the original route versus the temporary offset
path, and record pedestrian clearance plus road-boundary status explicitly.

## Preliminary multi-corridor hazard investigation

Test 3 was implemented as a monitoring experiment before changing braking
authority. Steering creates three distinct sensing questions:

1. Is the original route blocked?
2. Does the currently commanded lateral path appear blocked?
3. Do the left and right candidate paths appear blocked before a maneuver is
   selected?

The new multi-corridor LiDAR helper projects each LiDAR return onto the original
route once and measures its signed lateral distance from several parallel
corridor centers. `lane_follow_step()` exposes the following telemetry when
monitoring is enabled:

- `d_min_original_path_m`
- `d_min_commanded_path_m`
- `d_min_left_candidate_m`
- `d_min_right_candidate_m`

The original route still owns braking. The new readings are observational only.
This separation prevents an unvalidated corridor approximation from silently
changing the established safety controller.

### Visual test

With CARLA running in windowed mode:

```powershell
Set-Location "C:\Users\qdruc\Projects\Carla Project\src"
..\venv\Scripts\python.exe -X utf8 test5___scripted_pedestrian_steering.py
```

Visual legend:

- Blue/gray, turning orange during braking: original route corridor.
- Green: commanded route-right corridor, currently clear.
- Magenta: commanded corridor contains at least one accepted LiDAR return.
- Dark blue line: original planned route.
- Pink lookahead point: steering target.
- Cyan lookahead point: unshifted route target.

The terminal prints original, commanded, left-candidate, and right-candidate
minimum distances every 0.5 seconds after the pedestrian trigger.

To mirror the scripted maneuver:

```powershell
..\venv\Scripts\python.exe -X utf8 test5___scripted_pedestrian_steering.py --lateral-offset-m -1.5
```

### First live result

Configuration: 15 mph, 60 m encounter, near pedestrian from route-left, 4.0 s
trigger TTC, exponential braking, and a scripted 1.5 m route-right target.

- Collision: no.
- Recorded outcome: `slowed_avoided`.
- Minimum pedestrian distance: 6.762 m.
- Maximum actual route offset: 1.323 m.
- Final actual route offset: +1.194 m; timed recovery did not complete.
- Original corridor detected the pedestrian as it entered the lane.
- Route-left candidate readings detected the approaching pedestrian before the
  original corridor did.
- Route-right/commanded readings remained clear during most of the avoidance
  phase.

### Interpretation and newly exposed limitation

The incomplete recovery is informative. The near-cross pedestrian remains on
the original route, while the original corridor still owns braking. The ego
therefore slowed substantially and the time-based script requested a return
before the vehicle had safely passed the pedestrian. A real maneuver controller
must not recover based on elapsed time alone; recovery should require evidence
that the hazard is behind or the original route is clear.

The current commanded and candidate corridors are constant parallel offsets.
They do not yet represent the **swept transition path** between the ego's actual
position and the requested offset. Consequently, they must not yet take over
braking or be treated as complete safety checks. A safe next step is to model a
transition corridor that begins at the ego's measured lateral offset and blends
toward the target over a bounded forward distance. Road-boundary/drivable-area
checks are also still required before lateral selection becomes autonomous.

## Repository workflow

Continue working in the existing repository on `experiment/preliminary-steering-test`, which was created from `develop`. Do not copy `src/` into another testing folder and do not extend `previous_src_tests/`; Git already preserves the experiment history.

When implementation begins:

- Put reusable control logic in an appropriately named module under `src/` rather than duplicating the whole previous scenario.
- Keep small entry-point scripts in `src/` if a dedicated manual test runner is useful.
- Store generated raw results in a new timestamped directory and never reuse the Spring 2026 result paths.
- Commit related code, configuration, documentation, and summarized results on this branch.
- Open a pull request back to `develop` when the deterministic proof of concept is understood and reproducible.
