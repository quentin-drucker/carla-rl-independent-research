# CARLA Research Summary

> **Project:** Safety-Focused Reinforcement Learning for Autonomous Driving via Parameterized Rare-Hazard Scenarios in CARLA  
> **Course/context:** COM496, junior-year spring semester, 2026  
> **Return point:** Senior year, after several months away  
> **Reconstruction date:** August 12, 2026  
> **Purpose:** The authoritative starting point for remembering, running, evaluating, and extending this project

This document reconstructs the project from four bodies of evidence: the end-of-semester presentation and its speaker notes, the full weekly research log, the semester paper draft, and the current `Carla Project` repository. It was synthesized with GPT/Codex after separately auditing the documents, the repository, and then reconciling discrepancies between them.

This is written for my future self. It records both what I believed at semester end and what a later code/results audit can actually support. Where those differ, the distinction is explicit.

## Evidence-status vocabulary

| Label | Meaning in this document |
|---|---|
| **CURRENT** | Appears to be the latest relevant implementation or intended end-of-semester artifact. |
| **CONFIRMED** | Strongly supported by direct evidence from code, retained results, and/or multiple research documents. |
| **TENTATIVE / INCONCLUSIVE** | Observed or suspected, but not established well enough to treat as a finding. |
| **SUPERSEDED / HISTORICAL** | Previously used, but later replaced, abandoned, or made obsolete. |
| **UNCLEAR / NEEDS VERIFICATION** | Available evidence cannot resolve the question. |
| **INFERRED RECOMMENDATION** | A next step suggested by the reconstruction, not necessarily something I explicitly planned during the semester. |

Do not read an archived numerical value as automatically comparable across controllers. Several SAC and fixed-profile metrics were computed with different episode horizons or measurement windows; the affected claims are called out below.

## Table of contents

1. [Quick status snapshot](#quick-status-snapshot)
2. [Project and research overview](#project-and-research-overview)
3. [Research background and motivation](#research-background-and-motivation)
4. [Research evolution and semester timeline](#research-evolution-and-semester-timeline)
5. [Current project architecture](#current-project-architecture)
6. [Repository guide](#repository-guide)
7. [Current entry points and execution flow](#current-entry-points-and-execution-flow)
8. [CARLA and development-environment setup](#carla-and-development-environment-setup)
9. [How to run the project](#how-to-run-the-project)
10. [Important commands cheat sheet](#important-commands-cheat-sheet)
11. [Research methodology](#research-methodology)
12. [Experiments and tests](#experiments-and-tests)
13. [Models, algorithms, RL, and safety components](#models-algorithms-rl-and-safety-components)
14. [Confirmed findings](#confirmed-findings)
15. [Tentative and inconclusive findings](#tentative-and-inconclusive-findings)
16. [Failed, abandoned, and superseded approaches](#failed-abandoned-and-superseded-approaches)
17. [Bugs, problems, and technical limitations](#bugs-problems-and-technical-limitations)
18. [Research and experimental limitations](#research-and-experimental-limitations)
19. [Current state at the end of previous work](#current-state-at-the-end-of-previous-work)
20. [Open questions](#open-questions)
21. [Intended and logical next steps](#intended-and-logical-next-steps)
22. [If I have not touched this project in months, start here](#if-i-have-not-touched-this-project-in-months-start-here)
23. [Important paths and files cheat sheet](#important-paths-and-files-cheat-sheet)
24. [Important terminology and concepts](#important-terminology-and-concepts)
25. [Uncertainties and things that still need verification](#uncertainties-and-things-that-still-need-verification)
26. [Source materials used](#source-materials-used)
27. [Summary: where I am now](#summary-where-i-am-now)

---

# Quick Status Snapshot

## The five-minute version

| Question | Answer |
|---|---|
| **What is this research about?** | Building a controlled CARLA framework for rare pedestrian hazards, measuring safety and comfort, comparing four fixed automatic-emergency-braking (AEB) rules, and testing whether a Soft Actor-Critic (SAC) policy can learn adaptive braking near physical stopping limits. |
| **What did I accomplish?** | A working end-to-end experimental pipeline: deterministic route generation, custom lane following, PI speed control, route-conforming LiDAR detection, scripted pedestrian intrusion, fixed AEB profiles, quantitative metrics, JSON/CSV provenance, automated sweeps, Gymnasium integration, SAC training, evaluation, and plotting. |
| **What is the current implementation?** | The top-level `src/` tree, with `test3___ped_intrusion_scenario.py` for classical runs, `carla_aeb_env.py` for RL, `train_sac.py` for current v3 training, and `sac_v3_1600k.zip` as the intended final model. |
| **What does SAC actually control?** | One continuous brake target in `[0,1]`. Steering and cruising remain classical. More importantly, classical LiDAR hazard logic gates the `HAZARD_BRAKE` mode; SAC is a learned post-detection brake shaper, not an end-to-end or independently perceiving driving policy. |
| **What works?** | The retained artifacts show that single scenarios, 800-run profile sweeps, SAC training, randomized evaluation, matched-grid evaluation, data serialization, and plotting all ran during the semester. Current operation has not yet been smoke-tested after the break. |
| **What is the strongest result?** | On the final 200-configuration grid, fixed controllers recorded 19–23% collision rates, while the archived final SAC results recorded 20%. This supports descriptive overall collision-rate proximity on that grid, not statistical equivalence or SAC superiority. |
| **What major result is not trustworthy?** | The presentation's claimed SAC high-speed/full-stop advantage. The RL evaluator misclassified far-cross pedestrian clearance as a full stop, and SAC/fixed runs used different termination horizons. A corrected rate cannot be recovered from the CSV. |
| **What remains incomplete?** | Fair cross-controller outcome/metric computation, repeated seeds, model/source provenance, corrected avoidability labels, comfort tuning, ABS/skid analysis, broader hazards/environments, PPO, and multimodal perception. |
| **Where did I leave off?** | v3-1600k was presented as the strongest final checkpoint. I thought it roughly matched the best fixed rules, performed especially well at high speed, and was less comfortable. The later audit preserves only the archived overall collision-rate proximity; the high-speed and comfort comparisons remain unresolved. |
| **What should I do first?** | Do **not** train another policy first. Preserve the artifacts, identify/hash the authoritative v3 ZIP, repair termination/outcome and metric-window inconsistencies, add provenance and seeds, then rerun a fair matched comparison. |

## One-sentence state of the research

**CURRENT / CONFIRMED:** This CARLA pedestrian-AEB research framework was demonstrably functioning in April 2026 and retains an intended final six-observation SAC model plus useful archived experiments. Its current machine/runtime health still needs a smoke test, and its headline cross-controller full-stop, speed, comfort, and clearance comparisons require a corrected evaluation rerun before they can be treated as findings.

## Files to open first

1. This document.
2. [`src/train_sac.py`](src/train_sac.py) — current v3 sampler and training definition.
3. [`src/carla_aeb_env.py`](src/carla_aeb_env.py) — RL episode lifecycle, observations, reward call, and termination.
4. [`src/lane_follow.py`](src/lane_follow.py) — actual classical controller, hazard detection state, fixed profiles, and RL brake integration.
5. [`src/test3___ped_intrusion_scenario.py`](src/test3___ped_intrusion_scenario.py) — current classical scenario runner.
6. [`src/eval_sac_on_sweep.py`](src/eval_sac_on_sweep.py) — the matched evaluator and location of the central outcome defect.
7. [`src/runs/20260417_202355/sweep_summary.csv`](src/runs/20260417_202355/sweep_summary.csv) — final fixed-profile data.
8. [`src/runs/20260417_202355/sac_on_sweep_results.csv`](src/runs/20260417_202355/sac_on_sweep_results.csv) — final SAC data, with the caveats in this document.
9. [`WORKFLOW.md`](WORKFLOW.md) — useful historical run notes, but stale in several important places.

---

# Project and Research Overview

## Topic

The project investigated **safety-focused reinforcement learning for autonomous-driving emergency braking under parameterized rare-hazard conditions in CARLA**. The concrete completed hazard was a pedestrian entering the ego vehicle's route under controlled combinations of speed, time-to-crossing, encounter distance, crossing depth, and detection headway.

Despite the broad “autonomous driving” title, the implemented learning problem is deliberately narrow:

- The ego follows a predefined route with classical waypoint steering.
- A classical PI controller handles cruise speed.
- LiDAR plus route geometry determines whether a hazard is active.
- SAC chooses only the brake target after/during the hazard response.

It is therefore best described as **learned longitudinal AEB within a hybrid classical/RL driving stack**, not end-to-end autonomous driving.

## Original goals and how they evolved

The initial direction was broad: learn CARLA, control a vehicle, attach sensors, and eventually use reinforcement learning in safety-critical scenarios. The work became progressively more concrete:

1. Establish reliable low-level vehicle control.
2. Add sensor-in-the-loop obstacle detection and braking.
3. Convert a one-off simulation into a controlled, repeatable pedestrian scenario.
4. Quantify safety and comfort rather than judge runs visually.
5. Compare several fixed AEB responses.
6. Wrap the scenario as an RL environment and train SAC.
7. Diagnose failures near physical stopping limits and modify state/reward/training distributions.
8. Compare the final learned policy with fixed rules on a matched configuration grid.

The paper is dated April 18 and reflects an unfinished intermediate stage. Its Introduction first says RL was “being considered” or “presently underway,” then inconsistently calls learned-policy comparison part of the “current implementation.” It has an Introduction, Related Works, and references, but no Methods, Results, Discussion, or Conclusion. Neither tense supports completed empirical results. The repository and May 4 final presentation show that SAC was subsequently implemented, trained, and evaluated.

The earliest January planning notes also proposed a substantially different RL scope: RGB input, discrete actions, rule-based surrounding traffic, and explicit exclusion of continuous control/sensor fusion. That plan is **SUPERSEDED**. The completed project instead uses engineered LiDAR/TTC state features and a continuous SAC brake action.

## Final research objective

No source gives one formal, frozen hypothesis. The most faithful reconstruction is:

> Can a controlled, parameterized rare-hazard framework in CARLA support reproducible comparison of hand-designed and reinforcement-learned emergency-braking policies; can SAC learn an adaptive braking response; and how does its safety–comfort tradeoff compare with fixed AEB rules near physical stopping limits?

The first half—building the framework—was clearly accomplished. The second half produced promising but methodologically mixed evidence. The project did **not** demonstrate that SAC categorically outperforms fixed AEB.

## Why CARLA was used

CARLA provided:

- A controllable vehicle and pedestrian physics environment.
- Maps, road waypoints, spawn points, and global route planning.
- Configurable weather and vehicle physics.
- LiDAR, collision sensors, and spectator visualization.
- Synchronous fixed-step execution for controlled experiments.
- The ability to create dangerous pedestrian encounters without real-world risk.
- A practical bridge from a scripted control baseline to a Gymnasium RL environment.

The presentation described the *framework pattern* as generalizable or simulator-agnostic. That portability was not demonstrated. The current implementation is tightly coupled to CARLA APIs, a hard-coded CARLA installation, `Town04_Opt`, fixed spawn indices, CARLA actors, and CARLA route/physics objects.

---

# Research Background and Motivation

Average driving performance can hide failure at the boundary where safety margins become very small. The original framing emphasized understanding how an autonomous agent behaves when the safety margin becomes “slim to none,” rather than merely making average behavior safer.

The methodological problem was therefore not just controller design. It was how to create difficult but interpretable cases repeatedly enough to ask:

- Did the vehicle detect the hazard?
- Was collision avoidance physically possible at trigger time?
- Did the controller stop, slow and avoid, or collide?
- How close did the vehicle come to the pedestrian?
- How quickly did it stop?
- Was braking smooth or abrupt?
- Do different brake laws trade comfort for safety differently?
- Can a learned policy adapt across scenarios better than one fixed brake curve?

The parameterized scenario was intended to make rare-event evaluation systematic. Instead of waiting for a random pedestrian conflict, the code explicitly controls ego speed, route location, pedestrian crossing geometry, trigger TTC, walker speed and side, detection headway, and—in configuration support—weather and friction.

This is valuable even if SAC ultimately does not win. A reproducible hazard generator, consistent baseline set, metrics, and artifact trail are themselves a research contribution and a basis for later work.

## Literature positioning recorded in the paper

This is a source-aware map of the paper's Related Works, not a new independent literature review. Verify the cited papers and bibliography before reusing this section in a publication.

- **CARLA as the research substrate:** Dosovitskiy et al. introduced CARLA as a controlled open simulator with configurable actors, traffic, weather, and sensors. This supported safe repetition of events that would be dangerous or impractical to stage physically.
- **Broad or nominal learned driving:** Conditional Imitation Learning, NoCrash/the behavior-cloning limitations study, Learning by Cheating, and TransFuser represent work on general route-following, imitation, privileged learning, and sensor fusion. The paper used them as contrast: this project focuses on behavior during a narrow rare-hazard event rather than aggregate driving competence.
- **RL in CARLA:** The paper cites a review covering DQN, SAC, and PPO and recurring problems such as reward design, robustness, safety guarantees, and generalization. It also cites PPO-based intersection handling as an example of a narrower tactical RL task.
- **Scenario-based safety evaluation:** SafeBench and ISS-Scenario motivate unified safety-critical scenarios, parameterized scenario libraries, batch testing, and systematic discovery of failures instead of relying only on average driving scores.
- **Closest cited precedent:** Gutiérrez, Arango, and Gómez-Huelamo studied an unexpected pedestrian intrusion in CARLA using an Euro-NCAP-style validation protocol. The overlap is controlled pedestrian entry and safety-centered evaluation; this project's intended niche was parameterized scenario sweeps plus matched comparison of fixed and learned braking.

The paper positioned this work at the intersection of **CARLA-based RL** and **structured rare-event validation**. That remains a useful framing, but the completed evidence is limited to one CARLA pedestrian scenario family.

---

# Research Evolution and Semester Timeline

## 1. CARLA and sensor plumbing — before January 26

The earliest script spawned an ego vehicle, attached an RGB “dashcam,” saved periodic images, and temporarily used autopilot to validate CARLA interaction. This answered a setup question: could Python control the CARLA server, create actors, and receive sensor data?

- **Artifact:** `previous_src_tests/test0_ego_camera/`
- **Status:** **HISTORICAL** infrastructure validation, not a research controller.

## 2. Map and spawn exploration — January 26 to February 1

A map/spawn-point explorer enumerated available maps and visualized spawn points. The purpose was to choose repeatable geometry before implementing direct control.

- **Artifact:** `previous_src_tests/helper0_map_spawnpoint_explorer/`
- **Status:** **HISTORICAL**, still useful if a future study changes maps.

## 3. Custom lane following and speed control — February 2–8

The project replaced autopilot with custom control:

- Lane-center waypoint following.
- Heading-error steering toward a lookahead waypoint.
- A PI longitudinal controller for a target MPH.
- Synchronous fixed-step execution.
- Early lane locking using road/lane metadata.
- Telemetry for lane, heading, speed, steering smoothness, and distance.

The goal was a controllable baseline upon which safety behavior could be layered. Lane locking and distance accounting were acknowledged as imperfect.

- **Artifact:** `previous_src_tests/test1_lane_follow_waypoints/`
- **Legacy:** evolved into `src/lane_follow.py` and supporting utilities.

## 4. Early LiDAR braking — February 9–22

A roof LiDAR and steering-aware forward cone were added. The system mapped minimum forward distance to a continuous brake command, then applied ramp-up/ramp-down limits and persistence to reduce jitter. Telemetry and plotting were modularized from an approximately 1,100-line script.

The log initially suggested reliable behavior above roughly 20 mph, then corrected that impression the next week: reliability was mainly in controlled cases at or below approximately 20 mph, with smoothness and high-speed robustness still unresolved.

- **Artifact:** `previous_src_tests/test2_braking_via_lidar/`
- **Status:** **SUPERSEDED** by the route-corridor detector and current state machine.

## 5. State-machine braking and first pedestrian scenario — February 23 to March 1

Speed-scaled trigger distance, a panic threshold, and a four-state longitudinal controller were introduced:

```text
CRUISE → HAZARD_BRAKE → STOP_HOLD → RECOVER
```

This prevented PI wind-up and throttle/brake conflict. Work began on a controlled pedestrian intrusion, but CARLA's walker AI/NavMesh did not spawn or navigate reliably in the chosen setting.

The immediate research direction also became clearer: formalize TTC, minimum distance, acceleration/jerk, and repeatable scenario parameters.

## 6. Planning checkpoint — March 2–9

Midterm workload limited coding. The week was primarily a planning checkpoint that prepared the large spring-break consolidation.

## 7. Structured experiment pipeline — March 9–22

This was the major infrastructure pivot.

- The PI speed controller was stabilized by removing an approach taper, clamping the integrator, adding overspeed coast/light-brake behavior, and slew-limiting commands.
- The forward cone was replaced by a route-conforming **lane noodle** that filters LiDAR points near the planned route polyline.
- `GlobalRoutePlanner` produced one deterministic route used by steering, LiDAR filtering, and encounter placement.
- NavMesh walker control was abandoned for deterministic per-tick `WalkerControl`.
- `ScenarioConfig` centralized experimental parameters and JSON serialization.
- `RunResult` and sweep code stored configuration/result provenance.
- `brake_test.py` and later calibration tools estimated simulator stopping behavior.
- TTC, minimum pedestrian distance, jerk, trigger speed, time-to-stop, and categorical outcomes were recorded.

At this point, the project had become a research pipeline rather than a single demonstration.

## 8. Four fixed AEB profiles — March 23–30

Four hand-designed brake-target mappings were added:

- `proportional_ramp`
- `step_constant`
- `cautious_ramp`
- `exponential`

They shared the same hazard signal and ramp limiter, making it possible to study safety/comfort tradeoffs without changing the rest of the scenario.

## 9. Early sweeps, first SAC, and avoidability — March 30 to April 6

An early 192-run fixed-profile sweep was completed, and plotting tools were added.

The first Gymnasium wrapper exposed five observations and one continuous brake action. A fixed far-cross training setup produced a degenerate policy that coasted because the pedestrian usually cleared without collision. This was an informative failure: the training distribution supplied little collision pressure.

Training was redesigned to randomize speed, TTC, walker behavior, encounter distance, and side, with near crossings weighted more heavily. An urgent-braking reward was added. A physics-inspired avoidability classifier was built using a simplified stopping-distance equation and empirically chosen effective deceleration.

## 10. Randomized SAC evaluation — April 6–13

The 800k randomized SAC checkpoint was evaluated, and the avoidability taxonomy expanded from binary labels to:

- `avoidable`
- `borderline-avoidable`
- `borderline-impossible`
- `impossible`

The weekly entry contains an internal inconsistency: it refers to a 30-scenario evaluation but later reports counts totaling 97. It also explicitly warns that one randomized evaluation was not concrete evidence. PPO was considered but not implemented.

## 11. Final fixed sweep and SAC v2 — April 13–20

The fixed-profile grid expanded to 800 runs. When the machine shut down mid-sweep, `resume_sweep.py` re-ran and patched 101 interrupted rows.

The five-observation SAC model was continued to 1.6 million steps and evaluated on the same 200 unique configurations as the fixed profiles. Its perceived weakness in borderline-avoidable cases motivated v2:

- Added a sixth `stopping_feasibility_norm` observation.
- Increased collision penalty from 200 to 300.
- Doubled urgent-braking reward from 3 to 6.
- Reduced smoothness penalty from 0.2 to 0.05.
- Added a distance-sensitive stopping-margin bonus.

v2-1600k appeared more aggressive and was initially treated as the primary learned result. Continuing to v2-2400k produced mixed or degraded metrics and was not selected as final.

## 12. SAC v3 and final interpretation — April 20–27

A headway-distribution blind spot was identified as a suspected reason for poor v2 behavior: training had used only 2.5-second brake headway while evaluation included 2.5 and 5.0 seconds.

v3 was trained from scratch with two simultaneous changes:

- Fifty percent borderline-targeted TTC sampling.
- Randomized headway from `{2.5, 5.0}`.

The weekly log and presentation treated `sac_v3_1600k.zip` as the strongest final checkpoint. At semester end, I believed it:

- recorded a 20% overall collision rate,
- roughly matched the fixed profiles on safety,
- achieved a substantial high-speed full-stop advantage,
- and paid for that safety with the worst jerk/comfort.

The later reconstruction revises this interpretation:

- The retained matched CSV records 40/200 SAC collisions.
- The high-speed/full-stop claim is **not established** because of an outcome-labeling and episode-horizon defect.
- The comfort comparison is **not established** because jerk windows differ.

## 13. Presentation and handoff — late April to May

The final presentation framed the work as an end-to-end rare-hazard experiment pipeline and v3 as the final policy. The advisor raised skidding/wheel lock and the lack of ABS as important caveats, particularly for jerk interpretation. The final weekly entry records code organization and workflow notes for returning later.

`WORKFLOW.md` is a related April 17 handoff artifact. The final weekly entry records later organization and future-self notes, but the exact mapping to this file is unclear. In any case, it predates v3 and the final evaluation, so it must be read alongside the corrections in this document.

---

# Current Project Architecture

## System boundary

The current system is a **hybrid classical/RL AEB stack**:

```text
ScenarioConfig
    │
    ├─ CARLA world: Town04_Opt, weather, physics
    ├─ Ego: Tesla Model 3, fixed start/end route
    └─ Pedestrian: scripted near/far crossing
             │
             v
GlobalRoutePlanner route polyline
    ├─ waypoint/lookahead steering
    ├─ pedestrian encounter placement
    └─ route-conforming LiDAR “lane noodle”
             │
             v
Classical PI cruise + hazard detector + state machine
             │
             ├─ fixed profile computes brake target, or
             └─ SAC supplies continuous brake target
                         │
                         v
                  shared ramp limiter
                         │
                         v
                 CARLA VehicleControl
                         │
                         v
telemetry → metrics → RunResult/config JSON → sweep CSV → plots
```

## CARLA world and actors

The principal scripts connect to `localhost:2000`, load `Town04_Opt`, enable synchronous mode at `fixed_delta_seconds = 0.02`, spawn a Tesla Model 3, attach LiDAR and collision sensors, build a route, and spawn a pedestrian relative to a route encounter point.

The current fixed route uses:

- Ego spawn index: `242`
- End marker spawn index: `168`
- Global-route sampling resolution: `2.0 m`
- Steering lookahead: `6 m`

These indices are map/version-specific and should not be assumed portable.

## Route following and cruise control

`src/lane_follow.py` combines:

- Heading-error steering toward the planned route.
- Steering gain and smoothing/limits.
- PI target-speed control.
- Integrator clamping and decay.
- Overspeed coasting/light braking.
- Slew-limited throttle and cruise braking.
- State transitions among cruise, hazard braking, stop hold, and recovery.

The learned policy does not replace this system.

## LiDAR perception and hazard activation

`src/lidar_sensor.py` configures a roof-mounted ray-cast LiDAR:

| Setting | Value |
|---|---:|
| Range | 50 m |
| Rotation frequency | 50 Hz |
| Channels | 32 |
| Points per second | 400,000 |
| Upper FOV | +10° |
| Lower FOV | −30° |

`src/lidar_utils.py` transforms returns into world coordinates and filters them against the route corridor. Principal corridor settings in the current scenario/environment are:

- Sample step: `1.25 m`
- Half-width: `1.4 m`
- Minimum forward distance: `2.5 m`
- Nominal maximum route distance: `80 m`

The 80 m route corridor does not extend actual sensor range; LiDAR still cannot detect beyond 50 m.

Hazard trigger distance is conceptually:

```text
trigger_distance = 5.0 m + ego_speed_mps × brake_headway_s
```

At high speed and a five-second headway, this exceeds LiDAR range. The effective trigger is therefore sensor-limited.

## Pedestrian scenario

The pedestrian is spawned relative to the encounter waypoint and commanded every tick with deterministic `WalkerControl` rather than CARLA walker AI.

- `near`: the pedestrian stops around lane center, remaining an obstacle.
- `far`: the pedestrian continues across and clears the ego lane.
- `trigger_ttc_s`: controls when the crossing begins relative to nominal ego arrival.
- `encounter_distance_m`: controls where along the route the crossing occurs.
- Walker speed, starting side, delay, weather, friction, and simulation length are configurable.

The configuration is serialized for sweep runs, although deterministic behavior was asserted rather than established through formal repeated-run testing.

## Fixed brake profiles

All profiles map normalized hazard-zone penetration to a target brake value and then pass through a common rate limiter:

| Profile | Target mapping | Intended tradeoff |
|---|---|---|
| `proportional_ramp` | `penetration` | Progressive baseline |
| `step_constant` | `1.0` after detection | Maximum immediate response |
| `cautious_ramp` | linear, capped at `0.5` | Comfort-biased/weak braking |
| `exponential` | `penetration²` | Gentle early, steep later |

Because even `step_constant` passes through the common ramp limiter, it is not literally an instantaneous full wheel-brake command.

## SAC environment

`src/carla_aeb_env.py` adapts the same scenario machinery to Gymnasium:

- `reset()` samples/installs a `ScenarioConfig`, rebuilds actors, and returns a six-value state.
- `step(action)` advances one synchronous CARLA tick.
- `action[0]` is clamped to a brake target in `[0,1]`.
- `lane_follow_step()` still computes steering, speed state, LiDAR hazard status, mode transitions, and the applied ramped brake.
- Reward is computed from safety, comfort, and efficiency components.
- Episodes end on collision, actual full stop, far-cross completion, or timeout/truncation conditions.

The far-cross termination path is central to the known result bug discussed later.

## Output and result handling

Classical sweeps create:

```text
src/runs/<timestamp>/
├── run_0000/
│   ├── config.json
│   └── result.json
├── run_0001/
│   ├── config.json
│   └── result.json
└── sweep_summary.csv
```

Matched SAC evaluation writes `sac_on_sweep_results.csv` into the same sweep directory. Plot scripts read the two summary CSVs and write PNGs alongside them.

Important weakness: `eval_sac_on_sweep.py` overwrites the matched SAC CSV rather than versioning it. Raw v2 matched rows have apparently been overwritten; only their plots remain.

---

# Repository Guide

## Top level

| Path | Status | Purpose and warning |
|---|---|---|
| `MASTER_CARLA_RESEARCH_SUMMARY.md` | **CURRENT / IMPORTANT** | This reconstructed master reference. |
| `src/` | **CURRENT / IMPORTANT** | Latest source, model ZIPs, checkpoints, outputs, and run artifacts. |
| `previous_src_tests/` | **HISTORICAL / EXPERIMENTAL** | Preserves the progression from camera test through lane following, early LiDAR braking, and an April 5 test3 snapshot. Do not mistake it for the current implementation. |
| `notes/` | **HISTORICAL** | Early project/RL planning notes. Useful for intent, not current truth. |
| `venv/` | **CURRENT / IMPORTS VERIFIED** | Retained Windows virtual environment. Key dependencies and project modules imported successfully on August 12, 2026; a live CARLA episode still needs a smoke test. |
| `WORKFLOW.md` | **HISTORICAL / PARTLY STALE** | End-of-semester future-self notes. Useful overview, but its model defaults, training length, output names, and some commands no longer match current code. |

No `.git` repository, `requirements.txt`, lockfile, package definition, or automated installer was found.

## Current source areas

### Core scenario and controller

| Path | Status | Purpose |
|---|---|---|
| `src/test3___ped_intrusion_scenario.py` | **CURRENT** | Principal fixed-profile single-scenario runner and classical experiment implementation. |
| `src/scenario_config.py` | **CURRENT** | Dataclass containing pedestrian, ego, weather, friction, timing, and provenance parameters. |
| `src/lane_follow.py` | **CURRENT** | Steering, PI speed control, state machine, profile logic, ramping, and RL brake override. |
| `src/lidar_sensor.py` | **CURRENT** | LiDAR blueprint and queue setup. |
| `src/lidar_utils.py` | **CURRENT** | LiDAR parsing and route-corridor hazard filtering. |
| `src/walker_utils.py` | **CURRENT** | Scripted pedestrian spawn and motion. |
| `src/carla_session.py` | **CURRENT** | CARLA connection, world loading, and synchronous-mode helpers. |
| `src/spawning.py` | **CURRENT** | Vehicle/obstacle spawn helpers; current ego blueprint is Tesla Model 3. |
| `src/spectator.py` | **CURRENT utility** | Spectator follow/glide/free camera behavior. |
| `src/run_result.py` | **CURRENT** | Per-run result schema and serialization. |
| `src/run_stats.py`, `src/telemetry_plotting.py` | **CURRENT support** | Per-tick statistics and visualization. |

### RL and evaluation

| Path | Status | Purpose |
|---|---|---|
| `src/carla_aeb_env.py` | **CURRENT** | Six-observation Gymnasium environment around the CARLA scenario. |
| `src/rl_reward_design.py` | **CURRENT implementation/reference** | State construction, reward calculation, current constants, and extensive design comments. |
| `src/train_sac.py` | **CURRENT** | Fresh v3 training: 1.6M steps, targeted/random TTC, randomized headway. Header comments still contain older v2/fixed settings. |
| `src/continue_sac.py` | **CURRENT 6-D utility with broken default** | Can continue a compatible six-observation v2/v3 ZIP when passed explicitly, but its default five-observation `sac_aeb_rand_800k` is incompatible with the current six-observation environment. It resets entropy and does not restore the old replay buffer, so it is not a faithful resume. |
| `src/eval_sac.py` | **CURRENT script with stale default and outcome bug** | Seeded 100-episode randomized evaluation. Default model remains `sac_aeb_rand_800k`. |
| `src/eval_sac_on_sweep.py` | **CURRENT script with outcome/protocol bug** | Runs SAC deterministically on unique configurations from a profile sweep. Defaults to old `sac_aeb_rand_1600k`; explicitly pass v3. |
| `src/inspect_brake_curves.py` | **CURRENT diagnostic, partly broken** | Records SAC and analytical profile commands. Its output `ego_speed` field is populated with TTC in current code. |

### Sweeps, calibration, and analysis

| Path | Status | Purpose |
|---|---|---|
| `src/sweep.py` | **CURRENT** | Builds and runs the final 800-run fixed-profile grid; also contains an optional seeded random sweep. |
| `src/resume_sweep.py` | **CURRENT** | Detects `CRASHED` rows and reruns/patches them. |
| `src/brake_test.py` | **CURRENT diagnostic** | Direct full-brake stopping tests across speed/friction. |
| `src/brake_calibration.py` | **CURRENT diagnostic** | Measures stopping performance at different brake commands. |
| `src/avoidability.py` | **CURRENT but approximate** | Four-level stopping-margin classifier using nominal speed and fixed `a_eff = 3.5 m/s²`. |
| `src/validate_avoidability.py` | **CURRENT diagnostic** | Compares classifier expectations with step-constant outcomes. |
| `src/compare_braking_profiles.py` | **CURRENT offline analysis** | Fixed-profile descriptive plots. |
| `src/plot_eval.py` | **CURRENT offline analysis** | Randomized SAC evaluation plots. |
| `src/plot_unified_comparison.py` | **CURRENT but consumes flawed labels** | SAC/profile outcome and speed plots. |
| `src/plot_comfort_safety.py` | **CURRENT but cross-protocol metrics are not fair** | Comfort/safety distributions and radar plot. |
| `src/plot_speed_ttc_comparison.py` | **CURRENT** | Five-controller speed×TTC visualization. |
| `src/plot_presentation_summary.py` | **CURRENT plotting utility** | Presentation summary figure; has a hard-coded `SAC v2` display prefix that can produce misleading text such as `SAC v2 (v3 1600k)`. |
| `src/plot_stopping_distance.py` | **CURRENT offline diagnostic** | Stopping-distance reference from calibration CSV. |
| `src/load_town4.py` | **LOW-VALUE / INEFFECTIVE FOR MAIN FLOW** | Loads non-Opt `Town04`, but main environments reload `Town04_Opt`. |

## Models and retained outputs

### Model lineage

| Model | State dimension | Interpretation |
|---|---:|---|
| `src/sac_aeb_200k.zip` | 5 | Early/fixed-scenario lineage; exact experiment mapping is unclear. |
| `src/sac_aeb_rand_300k.zip` | 5 | **UNCLEAR** early five-input artifact. Its name suggests randomized training, but it may map to the documented fixed-far 300k experiment; the ZIP does not preserve its sampler. |
| `src/sac_aeb_rand_800k.zip` | 5 | Randomized 800k checkpoint used in early evaluation. |
| `src/sac_aeb_rand_1600k.zip` | 5 | Continued five-input checkpoint used in initial matched comparison. |
| `src/sac_v2_1600k.zip` | 6 | Added stopping feasibility and revised reward. Intermediate. |
| `src/sac_v2_2400k.zip` | 6 | Continued v2; mixed/degraded result. Superseded. |
| `src/sac_v3_1600k.zip` | 6 | **CURRENT intended final model.** Fresh training with targeted TTC and both headways. |
| `src/checkpoints/` | mixed | Hundreds of intermediate checkpoints, roughly 1 GB total. Useful for lineage studies, not normal resumption. |

The top-level v3 ZIP and nominal 1.6M callback checkpoint contain different policy weights. The retained matched CSV stores a model label but no hash, so the exact evaluated bytes are **UNCLEAR**.

### Run directories

- `src/runs/20260417_202355/` — **CURRENT / MOST IMPORTANT** final fixed sweep, final overwritten matched SAC CSV, and v2/v3 comparison plots.
- `src/runs/20260405_025448/` — important earlier 192-run/matched comparison family; exact role should be read with the weekly log.
- Earlier March and early-April directories — development sweeps, small grids, interrupted runs, or validation experiments. Preserve them as history; do not pool them blindly.
- Several timestamp directories are incomplete/interrupted and contain little or no usable output.

The `previous_src_tests/test3_rare_hazard_scenario/` copy contains duplicated April 5-era code and run data. It is an archive, not an independent current dataset.

---

# Current Entry Points and Execution Flow

## Classical single run

`python test3___ped_intrusion_scenario.py`

Execution approximately follows:

1. Construct a `ScenarioConfig` near the bottom of the script.
2. Connect to CARLA and load `Town04_Opt`.
3. Enable 50 Hz synchronous mode and apply weather.
4. Spawn ego, LiDAR, collision sensor, and scripted walker.
5. Build the global route and encounter geometry.
6. Each tick:
   - update pedestrian motion,
   - consume LiDAR,
   - compute route-following and speed control,
   - update hazard state and chosen brake profile,
   - apply vehicle control,
   - accumulate telemetry/metrics.
7. Continue through the post-cross settling period or another end condition.
8. Return a `RunResult`; sweep callers serialize it.

## Fixed-profile sweep

`python sweep.py`

The current bottom-of-file grid expands to:

```text
5 speeds × 2 encounter distances × 2 crossing depths ×
5 trigger TTCs × 4 profiles × 2 headways = 800 runs
```

Each run calls the same classical scenario implementation and writes config/result JSON. The sweep finally writes `sweep_summary.csv`.

## SAC training

`python train_sac.py`

The current main block:

1. Creates `CarlaAEBEnv(config_fn=sample_config)`.
2. Creates SB3 `SAC("MlpPolicy", ...)`.
3. Trains for 1,600,000 simulator steps.
4. Saves every 20,000 steps under `src/checkpoints/`.
5. Saves the final model as `src/sac_v3_1600k.zip`.

This is the latest training path even though the module header still describes older v2 settings.

## Randomized SAC evaluation

`python eval_sac.py sac_v3_1600k`

This seeds Python and NumPy with 42 and runs 100 deterministic-policy episodes using `sample_config()`. It writes a model-specific CSV under current code. CARLA and Gymnasium/environment randomness are not comprehensively seeded.

The script contains the far-clearance/full-stop outcome defect. Use it only after correcting that logic if the goal is new authoritative results.

## Matched SAC evaluation

`python eval_sac_on_sweep.py sac_v3_1600k runs/20260417_202355/sweep_summary.csv`

The script takes the `proportional_ramp` rows as one copy of the 200 unique scenario configurations, runs SAC deterministically on each, and writes `sac_on_sweep_results.csv` into the sweep directory. It reconstructs only a subset of `ScenarioConfig`, omitting friction, sun/cloud overrides, post-trigger delay, and random seed. Calling it “matched” is justified for the retained final grid because those omitted values were defaults, but not for arbitrary future sweeps that vary them.

This is the intended apples-to-apples comparison path, but it is not currently apples-to-apples because SAC and fixed runners terminate and measure several metrics differently.

Operationally, the evaluator retains results in memory until all episodes finish, saves no partial progress, has no exception-safe `env.close()`, and overwrites its destination CSV. A crash can therefore lose the whole new evaluation and leave the CARLA world dirty.

## Older entry points to avoid confusing with current work

- Everything under `previous_src_tests/` is historical.
- `continue_sac.py` defaults to a five-observation model that is incompatible with the current six-observation environment; pass a compatible six-observation model explicitly if deliberately using it.
- `eval_sac.py` defaults to `sac_aeb_rand_800k`, not v3.
- `eval_sac_on_sweep.py` defaults to `sac_aeb_rand_1600k`, not v3.
- `inspect_brake_curves.py` defaults to `sac_v2_1600k`, not v3.
- `load_town4.py` does not configure the main environment's final map.

Always pass explicit model and data paths.

---

# CARLA and Development-Environment Setup

## Verified from the retained environment and code

| Component | Retained/current value |
|---|---|
| OS/shell assumption | Windows + PowerShell |
| CARLA | 0.9.16 |
| Hard-coded CARLA root | `C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16` |
| CARLA executable | `C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe` |
| Server | `localhost:2000` |
| Python | 3.12.10 in retained `venv` |
| Stable-Baselines3 | 2.8.0 |
| Gymnasium | 1.2.3 |
| PyTorch | 2.11.0 CPU build |
| NumPy | 2.4.2 |
| pandas | 3.0.2 |
| Matplotlib | 3.10.8 |
| pygame | 2.6.1 |
| pynput | 1.8.1 |
| Principal map | `Town04_Opt` |
| Fixed timestep | 0.02 s / 50 Hz |

The scripts append CARLA's `PythonAPI` and `PythonAPI/carla` directories to `sys.path`; there is no normal packaged dependency configuration.

## Reconstructed or needs verification

- **VERIFIED August 12, 2026:** The retained interpreter reports Python 3.12.10, and CARLA 0.9.16, Gymnasium, SB3, PyTorch, NumPy, pandas, Matplotlib, pygame, pynput, `GlobalRoutePlanner`, and core project modules import successfully in read-only checks.
- **VERIFIED path only:** The hard-coded CARLA executable exists. **NEEDS VERIFICATION:** It launches, accepts a client connection, loads `Town04_Opt`, and completes an episode successfully.
- **VERIFIED import only:** CARLA's Python API imports through the retained setup. Live client/server interoperability remains part of the smoke test.
- **NEEDS VERIFICATION:** CPU-only PyTorch was intentional. SAC's small MLP can run on CPU, but wall-clock training performance should be measured before another long run.
- **NEEDS VERIFICATION:** Current source still produces the same behavior as April; there is no Git commit or clean environment manifest.
- **UNCLEAR:** A complete installation command. No `requirements.txt` or lockfile exists, so do not reconstruct the environment by guessing versions when the existing `venv` may still work.

## Known environment quirks

- CARLA must be running before scripts connect.
- Wait approximately 15 seconds after starting the simulator.
- Main scripts reload `Town04_Opt`; preloading `Town04` is ineffective.
- Headless `-RenderOffScreen` mode was used for long training and sweeps.
- Scripts assume they are executed from `src/` because models, `runs/`, checkpoints, and outputs use relative paths.
- OneDrive paths and large checkpoint/run trees may introduce sync overhead.
- Synchronous mode improves control over simulator time but does not prove bitwise deterministic physics.

---

# How to Run the Project

These commands are verified against current paths and code defaults, but no CARLA episode was launched during the reconstruction. Treat the first run as a smoke test.

**Write/overwrite warning:** The commands below are operational procedures, not all read-only actions. `train_sac.py` can overwrite the final v3 ZIP and colliding checkpoint names; evaluators, calibration, brake-curve inspection, and plotters overwrite their named CSV/PNG outputs. Work from copied/versioned artifact directories after the first smoke test.

## 1. Start CARLA

Open PowerShell terminal A.

Headless, recommended for sweeps/training:

```powershell
& "C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe" -RenderOffScreen
```

Windowed, useful for a smoke test:

```powershell
& "C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe"
```

Wait around 15 seconds. Leave this terminal/process running.

## 2. Activate the project environment

Open PowerShell terminal B.

```powershell
Set-Location "C:\Users\qdruc\OneDrive\Desktop\Carla Project"
.\venv\Scripts\Activate.ps1
Set-Location .\src
```

Optional read-only environment check:

```powershell
python --version
python -c "import carla, gymnasium, stable_baselines3, torch; print('imports OK')"
```

## 3. Safest first smoke test

Run one visible classical pedestrian scenario before training or sweeping:

```powershell
python test3___ped_intrusion_scenario.py
```

The bottom-of-file `ScenarioConfig` is the active single-run configuration—not the dataclass defaults. It currently selects 35 mph, a 120 m encounter, left-side near crossing, 2.8 s trigger TTC, 1.8 m/s walker, `exponential` braking, `ClearSunset`, 0° sun altitude, and a 13 s ceiling. `plot_after=True` may block until plot windows are closed. The script always calls `load_world("Town04_Opt")`, replacing the current world and destroying actors owned by other clients.

## 4. Run a fixed-profile sweep

```powershell
python sweep.py
```

Warning: the current active grid is 800 runs and can take hours. For a first smoke test, do not edit the source unless deliberately beginning new development; inspect or design a smaller, separately tracked validation plan first.

`resume_sweep.py` is a narrow repair utility, not a general resume system. It only retries rows whose `error` field is exactly `CRASHED`; it does not discover missing rows/directories or other partial states. Its no-argument “latest” selection is lexicographic by folder name, not modification time. Use it only on a backed-up working copy and pass an explicit directory.

For an active copied sweep with exact `CRASHED` rows:

```powershell
python resume_sweep.py runs\<working-copy-timestamp>
```

The final `runs\20260417_202355\sweep_summary.csv` already has 800 rows and zero `CRASHED` rows; there is nothing to resume there. The utility reconstructs only part of `ScenarioConfig` and omits friction, sun/cloud overrides, post-trigger delay, and random seed. That was compatible with the final grid's defaults but is unsafe for arbitrary future sweeps. It overwrites run JSON and later rewrites the CSV in place, so interruption can leave them inconsistent.

## 5. Train current SAC v3

```powershell
python train_sac.py
```

Current behavior is 1.6 million steps, a checkpoint every 20,000 steps, and final output `sac_v3_1600k.zip`.

**Do not do this as the first resumption action.** It can overwrite/confuse the intended final model and will not fix the evaluation validity problems.

## 6. Evaluate SAC on randomized scenarios

```powershell
python eval_sac.py sac_v3_1600k
```

This is the syntactically correct current command. The evaluation outcome logic must be repaired before treating new full-stop counts as authoritative.

## 7. Evaluate SAC on the final matched grid

```powershell
python eval_sac_on_sweep.py sac_v3_1600k runs\20260417_202355\sweep_summary.csv
```

**Data-preservation warning:** this overwrites `runs/20260417_202355/sac_on_sweep_results.csv`. Copy the entire archived run directory to a new, explicitly named experiment directory before any rerun. Do not overwrite the only retained final raw SAC comparison.

## 8. Offline analysis

These do not need a running CARLA server. Run them only against copied/versioned outputs if preservation matters, because they overwrite PNGs in `src/` or the selected run directory. `compare_braking_profiles.py` also opens interactive plot windows and may wait until they are closed.

```powershell
python compare_braking_profiles.py runs\20260417_202355\sweep_summary.csv
python plot_unified_comparison.py runs\20260417_202355\sweep_summary.csv
python plot_comfort_safety.py runs\20260417_202355\sweep_summary.csv
python plot_presentation_summary.py runs\20260417_202355\sweep_summary.csv
python plot_stopping_distance.py
```

The unified/comfort presentation plots will faithfully reproduce flawed SAC labels or incomparable metric windows. Use them as historical artifacts until evaluation is corrected.

To plot a newly generated randomized v3 evaluation:

```powershell
python plot_eval.py eval_results_sac_v3_1600k.csv
```

`plot_eval.py` without an argument reads the provenance-unclear generic CSV. Likewise, `plot_speed_ttc_comparison.py` without arguments mixes generic randomized SAC data with the latest fixed sweep; it is an unmatched historical visualization, not the final matched comparison.

## 9. Calibration and diagnostics

With CARLA running:

```powershell
python brake_test.py
python brake_calibration.py
python inspect_brake_curves.py sac_v3_1600k 5
```

Offline classifier validation:

```powershell
python validate_avoidability.py runs\20260417_202355\sweep_summary.csv
```

`inspect_brake_curves.py` currently writes TTC into a column named `ego_speed`; correct that before using its CSV scientifically.

## 10. Stopping and cleanup

- Use `Ctrl+C` in the training/evaluation terminal to interrupt Python. `train_sac.py` attempts to preserve an interrupted checkpoint.
- Stop the CARLA process from its terminal with `Ctrl+C`, or close the windowed simulator normally.
- Several scripts lack exception-safe `env.close()`/world cleanup, and even the main scenario's early setup path can fail before all cleanup variables exist. An interrupt can leave actors or synchronous mode active. Use a dedicated CARLA instance and restart it after a failed/interrupted script before trusting later runs.
- Do not delete or “clean” archived runs/checkpoints until they have been backed up and inventoried.

---

# Important Commands Cheat Sheet

Run from project root unless the command includes `Set-Location .\src`.

```powershell
# Project environment
Set-Location "C:\Users\qdruc\OneDrive\Desktop\Carla Project"
.\venv\Scripts\Activate.ps1
Set-Location .\src

# CARLA, usually in a separate terminal
& "C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe" -RenderOffScreen

# One current classical scenario
python test3___ped_intrusion_scenario.py

# Fixed profiles
python sweep.py
# Only for a backed-up active sweep containing exact CRASHED rows:
python resume_sweep.py runs\<working-copy-timestamp>

# Current intended SAC model
python train_sac.py
python eval_sac.py sac_v3_1600k
python eval_sac_on_sweep.py sac_v3_1600k runs\20260417_202355\sweep_summary.csv

# Physics and behavior diagnostics
python brake_test.py
python brake_calibration.py
python validate_avoidability.py runs\20260417_202355\sweep_summary.csv
python inspect_brake_curves.py sac_v3_1600k 5

# Offline plots
python compare_braking_profiles.py runs\20260417_202355\sweep_summary.csv
python plot_unified_comparison.py runs\20260417_202355\sweep_summary.csv
python plot_comfort_safety.py runs\20260417_202355\sweep_summary.csv
```

Most commands after the environment/CARLA startup write or overwrite artifacts. Do not run this block mechanically against the archived final directory; use the detailed warnings in [How to Run the Project](#how-to-run-the-project).

There is no project-specific Git workflow because this directory is not currently a Git repository.

---

# Research Methodology

## Intended methodology

The intended research design was:

1. Construct a controlled rare-hazard scenario.
2. Parameterize conditions that affect severity.
3. Run repeated/multi-condition experiments in synchronous simulation.
4. Measure safety and comfort quantitatively.
5. Establish hand-designed AEB baselines.
6. Train an RL policy on varied scenarios.
7. Compare RL and baselines on the same scenario configurations.
8. Separate agent failures from physically impossible cases.

Most of this structure was implemented. “Repeated” ultimately meant many parameter combinations, not repeated stochastic replicates of the same configuration or multiple independently trained policies.

## Scenario variables

The final fixed grid varied:

| Variable | Values |
|---|---|
| Target speed | 22, 28, 35, 40, 45 mph |
| Encounter distance | 60, 120 m |
| Crossing depth | near, far |
| Trigger TTC | 1.8, 2.2, 2.6, 3.0, 3.5 s |
| Brake profile | proportional, step, cautious, exponential |
| Brake headway | 2.5, 5.0 s |

Walker speed was fixed at `1.8 m/s`, walker side was `left`, brake ramp-up was the `4.0 units/s` `ScenarioConfig` default, weather was `ClearSunset`, friction was CARLA default, and the hard ceiling was 30 simulator seconds. This produced 800 controller runs and 200 unique scenario configurations.

The v3 training sampler instead used continuous randomization:

- Target speed: uniform 22–50 mph.
- Trigger TTC: 50% broad uniform 1.5–3.5 s; 50% sampled to target the nominal borderline-avoidable margin for the chosen speed.
- Walker speed: uniform 1.0–2.5 m/s.
- Encounter distance: nominal lead distance plus a uniform 30–80 m buffer, capped at 160 m.
- Crossing: near/far weighted 2:1.
- Side: left/right uniformly sampled.
- Brake headway: 2.5 or 5.0 s.
- Weather: `ClearNoon`.
- Default road friction.
- Episode ceiling: 30 seconds.

## Baselines

The four fixed brake profiles were the principal baselines. There was no:

- random-policy baseline,
- untrained-network baseline,
- CARLA Traffic Manager AEB baseline,
- production AEB implementation,
- PPO implementation,
- or human/controller benchmark.

## Safety and comfort metrics

Metrics included:

- Collision flag.
- Outcome category: full stop, slowed/avoided, or collision.
- Ego speed at trigger.
- Minimum TTC.
- Minimum pedestrian distance.
- Time to full stop.
- Maximum and mean jerk.
- Reward and reward components for RL.
- Physics/avoidability label and stopping margin during evaluation.

The metric ideas are appropriate, but their implementation differs between the classical and RL runners. Cross-controller jerk, minimum distance, time-to-stop, and full-stop conclusions are therefore not currently valid without rerunning under one shared protocol.

## Avoidability model

The core simplified calculation is:

```text
required stopping distance = v₀² / (2 × a_eff)
available distance         = v₀ × trigger_TTC
margin                     = available − required
```

with `a_eff = 3.5 m/s²` and an uncertainty band of ±8 m:

- margin ≥ 8 m: `avoidable`
- 0–8 m: `borderline-avoidable`
- −8–0 m: `borderline-impossible`
- below −8 m: `impossible`

This is an interpretive heuristic. Current evaluation computes it from configured target speed before the run, not actual speed at trigger. It omits sensor-range clipping, exact trigger/detection behavior, and brake-ramp delay; for far crossings it uses a fixed `3.3 m / walker_speed` clearance heuristic rather than the exact scripted path, post-trigger delay, or measured crossing state. Several other physics effects are also omitted.

## Randomness and reproducibility

Positive features:

- Fixed 0.02 s synchronous stepping.
- Scripted pedestrian control.
- Explicit fixed grid.
- Per-run config/result JSON for classical sweeps.
- `eval_sac.py` seeds Python and NumPy with 42.
- Random-sweep helper accepts a seed.

Weaknesses:

- `train_sac.py` does not set Python, NumPy, Gymnasium, SAC, PyTorch, or CARLA seeds comprehensively.
- No repeated training seeds.
- No repeated same-config determinism study.
- No held-out scenario registry.
- No source commit or environment lockfile.
- Model hash is not stored in result CSVs.
- Evaluations generally contain one pass per configuration.

The project supports structured replay better than an ad hoc demonstration, but “fully repeatable” is too strong.

---
# Experiments and Tests

This catalog groups small development tests when they answered the same research question. Dates and interpretations come from the weekly log; paths and implementation details come from the repository.

## Experiment family 1: CARLA, camera, map, and control prerequisites

**Purpose:** Establish that CARLA could be controlled from Python and that a repeatable ego-driving baseline was feasible.

**Research questions:**

- Can a Python client create/control an ego actor and receive sensor data?
- Which maps/spawn points are suitable?
- Can custom control replace autopilot and follow waypoints at a target speed?

**Relevant artifacts:**

- `previous_src_tests/test0_ego_camera/`
- `previous_src_tests/helper0_map_spawnpoint_explorer/`
- `previous_src_tests/test1_lane_follow_waypoints/`

**Procedure and measurements:** Spawned an ego, collected RGB frames, enumerated map spawn points, then implemented heading-error waypoint steering and PI speed control with lane/speed/steering telemetry.

**Result:** The client/sensor/control pipeline worked and became the foundation for the later safety scenario.

**Status:** **CONFIRMED / HISTORICAL.** These were engineering prerequisites rather than final experiments. The code was superseded, but the progression matters.

## Experiment family 2: Early LiDAR obstacle braking

**Purpose:** Determine whether a simple forward sensor could intervene on top of classical lane following and whether continuous/ramped braking was stable.

**Relevant artifacts:**

- `previous_src_tests/test2_braking_via_lidar/test2___braking_via_lidar.py`
- Associated historical `test2___*.py` modules.

**Setup:** A roof LiDAR filtered points in a steering-aware forward cone. Minimum obstacle distance produced a continuous hazard-brake target, with rise/fall limits and brief persistence.

**Measured/observed:** Hazard distance, brake target, ramped/applied command, vehicle speed, and later plots of per-tick behavior.

**Result:** The approach worked in selected controlled cases but exhibited noise/jitter and did not establish robust high-speed performance. An early claim of reliability above roughly 20 mph was corrected the following week to mostly controlled cases at or below that speed. Later state-machine and route-corridor work reportedly improved behavior toward 40 mph, but this was not a retained formal benchmark.

**Status:** **SUPERSEDED / TENTATIVE.** Useful proof of concept; do not cite its speed impressions as established findings.

## Experiment family 3: Pedestrian-scenario repeatability and geometry

**Purpose:** Replace ad hoc obstacle tests with a configurable rare-hazard episode.

**Research question:** Can the same route-relative pedestrian conflict be generated across controlled severity settings?

**Relevant files:**

- `src/test3___ped_intrusion_scenario.py`
- `src/scenario_config.py`
- `src/walker_utils.py`
- `src/lidar_utils.py`
- `src/lane_follow.py`
- `src/run_result.py`

**Key implementation changes:**

- Global route rather than local `wp.next()` progression.
- Encounter point defined by route distance.
- Walker starts/ends relative to lane geometry.
- Deterministic per-tick `WalkerControl` rather than NavMesh navigation.
- Near and far crossing variants.
- JSON config and result serialization.
- Fixed-step simulation and automated sweep execution.

**Result:** The repository contains many completed runs with serialized configurations and outcomes, demonstrating that the pipeline was operational. The log says repeated same-config behavior appeared consistent.

**Status:** **CONFIRMED implementation; TENTATIVE deterministic-repeatability claim.** No formal repeated-run variance test was retained.

## Experiment family 4: Full-brake capability and avoidability calibration

**Purpose:** Estimate CARLA's stopping envelope and distinguish policy mistakes from cases considered physically infeasible.

**Relevant files/data:**

- `src/brake_test.py`
- `src/brake_calibration.py`
- `src/brake_calibration_results.csv`
- `src/plot_stopping_distance.py`
- `src/avoidability.py`
- `src/validate_avoidability.py`

**Procedure:**

- Accelerate the Tesla Model 3 toward target speeds.
- Apply direct fixed brake levels and record stopping distance/time.
- Explore friction settings in the direct-brake diagnostic.
- Compare the simplified stopping model with fixed `step_constant` scenario outcomes.

**Documented interpretation:** Direct full-brake tests measured roughly `6.96–7.56 m/s²` raw deceleration, much stronger than the scenario system's effective stopping behavior because the shared ramp limiter consumes time/distance. `a_eff = 3.5 m/s²` was selected as a scenario-level approximation. The weekly log initially claimed 20/24 validation matches.

**Repository re-audit:**

- The calibration CSV shows brake `1.0` produced the shortest stopping distance at every tested speed; it does not support a hypothesized sub-lock brake optimum.
- Actual initial speeds did not always equal requested values, especially at nominal 45/50 mph.
- Current validation against the final step-constant data is closer to 90/100 near-cross cases classified consistently, with ten “avoidable” step collisions. The exact calibration story changed as data and code evolved.

**Status:** **CONFIRMED as a useful diagnostic; TENTATIVE as a physics classifier.** It is not ground truth and should not be used to declare a collision “not the agent's fault” without qualification.

## Experiment family 5: Four fixed-profile comparison

**Purpose:** Establish interpretable AEB baselines and explore safety/comfort tradeoffs.

**Research question:** Does one fixed brake-target curve dominate, or do aggressive/gentle profiles trade collision avoidance against smoothness and stopping behavior?

**Relevant files:**

- `src/lane_follow.py`
- `src/sweep.py`
- `src/compare_braking_profiles.py`
- `src/runs/20260417_202355/sweep_summary.csv`

**Final setup:**

```text
speed:             22, 28, 35, 40, 45 mph
encounter distance: 60, 120 m
crossing:           near, far
trigger TTC:        1.8, 2.2, 2.6, 3.0, 3.5 s
headway:            2.5, 5.0 s
profile:            proportional, step, cautious, exponential
walker speed:       1.8 m/s
weather:            ClearSunset
```

**Final results:**

| Profile | Collisions | Recorded full stops | Slowed/avoided | Collision rate |
|---|---:|---:|---:|---:|
| Step constant | 38 | 144 | 18 | 19% |
| Proportional ramp | 40 | 135 | 25 | 20% |
| Exponential | 42 | 123 | 35 | 21% |
| Cautious ramp | 46 | 113 | 41 | 23% |

**Interpretation:** Step constant was descriptively safest by collision count; cautious ramp was weakest. No single fixed profile obviously dominated all safety and comfort measures. Because these four controllers share one runner and outcome protocol, their internal comparison is the cleanest empirical result in the repository.

**Caveats:**

- One run per controller/configuration; no replicate variability.
- No inferential statistics.
- `step_constant` still passes through a ramp limiter.
- Some high-headway values exceed LiDAR range.
- Results apply to one route, vehicle, hazard family, walker speed, weather, and default friction.

**Status:** **CONFIRMED descriptive experiment, CURRENT relevance.**

## Experiment family 6: Fixed far-cross SAC and degenerate coasting

**Purpose:** First test of the CARLA-to-Gymnasium/SAC pipeline.

**Research question:** Can SAC learn braking from one controlled scenario?

**Likely artifacts:** `src/sac_aeb_200k.zip` and/or `src/sac_aeb_rand_300k.zip`; exact model-to-log mapping is **UNCLEAR**.

**Setup:** The early environment used five observations and a fixed far-cross scenario.

**Result:** The policy learned to coast because the pedestrian generally cleared the lane before arrival, so collision pressure was absent or weak. This was not a mysterious RL failure; it exposed a poorly chosen training distribution/reward situation.

**Response:** Randomize scenario parameters, increase near-cross frequency, repair encounter-distance triggering, and add urgent-braking reward.

**Status:** **FAILED / SUPERSEDED, but scientifically useful.** It showed that task distribution and reward incentives matter more than simply adding an RL algorithm.

## Experiment family 7: Randomized five-observation SAC

**Purpose:** Train a policy across a broader hazard distribution after the coasting failure.

**Five-input model lineage:**

- `sac_aeb_rand_300k.zip` — sampler provenance **UNCLEAR** despite its filename.
- `sac_aeb_rand_800k.zip`
- `sac_aeb_rand_1600k.zip`

**Five observations:** ego speed, obstacle proximity, TTC urgency, previous brake, hazard flag.

**Training changes:** Continuous speed/TTC/walker/encounter randomization, two-to-one near-cross weighting, larger replay buffer and batch size in later runs, and more explicit reward for braking in urgent TTC states.

### Provenance-uncertain randomized evaluation

`src/eval_results.csv` contains 100 rows with the presentation's physics-stratified rates. It cannot be assigned confidently to this five-observation lineage—or to v3—because it has no model field and predates the current evaluator revision:

| Stored physics label | Collisions / total |
|---|---:|
| Avoidable | 0/35 |
| Borderline-avoidable | 21/49 |
| Borderline-impossible | 3/5 |
| Impossible | 11/11 |

**Interpretation at the time:** Failures concentrated in difficult categories, suggesting learned, nonrandom behavior.

**Later qualification:**

- The CSV has no model column and cannot be bound to v3 or another checkpoint.
- Sixteen of 64 stored full-stop labels are far-cross rows without stop-time evidence, consistent with the known clearance conflation but not source-version-proven to have exactly the current termination path.
- The avoidability classifier is approximate.
- There is no untrained/random-policy control.
- The “unseen” status of the evaluation episodes is not provable.

**Status:** **CONFIRMED archived pattern; INCONCLUSIVE policy/generalization claim.**

## Experiment family 8: Six-observation SAC v2

**Purpose:** Address poor performance around the nominal stopping boundary.

**Research hypothesis:** Giving the policy explicit stopping feasibility and shifting reward weight toward urgency/collision avoidance should improve tight-but-avoidable cases.

**Relevant artifacts:**

- `src/rl_reward_design.py`
- `src/carla_aeb_env.py`
- `src/sac_v2_1600k.zip`
- v2 plots in `src/runs/20260417_202355/`

**Changes:** Sixth state feature, higher collision and urgent-braking weights, lower action-smoothness weight, larger stop bonus, and distance-based margin bonus.

**Documented v2-1600k interpretation:** Larger pedestrian margin and faster full stops, but higher jerk; overall collision approximately within the profile range. These raw v2 matched rows no longer survive because the shared SAC CSV was overwritten.

**Status:** **SUPERSEDED / PARTIALLY AUDITABLE.** The model and plots survive, but not the raw matched rows needed to reproduce the claims independently.

## Experiment family 9: Continued SAC v2-2400k

**Purpose:** Determine whether another 800k steps improved v2.

**Artifact:** `src/sac_v2_2400k.zip` and named presentation/comparison plots.

**Documented result:** Overall and borderline collision behavior worsened while some high-speed/full-stop behavior appeared stronger. It was retained as a secondary example of diminishing returns/instability rather than chosen as final.

**Explanations considered:** Empty replay buffer during continuation, policy drift, entropy behavior, and later a mismatch between training/evaluation headways.

**Audit conclusion:** No retained training curves or replay buffer isolate the cause. The stored model's entropy coefficient does not establish the claimed “entropy collapse.”

**Status:** **SUPERSEDED; causal diagnosis INCONCLUSIVE.**

## Experiment family 10: SAC v3-1600k

**Purpose:** Correct the suspected v2 headway blind spot and overrepresent the borderline region without abandoning broad training coverage.

**Artifact:**

- `src/train_sac.py`
- `src/sac_v3_1600k.zip`
- `src/runs/20260417_202355/sac_on_sweep_results.csv`
- v3-named plots in the same run directory.

**Setup:** Six observations; 1.6 million fresh steps; 50% broad/50% borderline-targeted TTC; headway sampled from 2.5 or 5.0 seconds; near/far weighted 2:1.

**Documented result:** The log and presentation reported 20% overall collisions, 20% borderline-avoidable collisions, a 45% full-stop rate at 45 mph, 1.47 s mean time-to-stop, 7.40 m mean minimum distance, and 12.71 m/s³ mean jerk. v3 was declared the strongest final checkpoint.

**Audited result:**

- `40/200 = 20%` collision is present in the archived matched CSV.
- `8/40 = 20%` borderline-avoidable collision is present under stored nominal-speed labels.
- Reclassification with logged trigger speed changes six labels and yields about `10/38 = 26.3%`; that remains heuristic.
- Only 115 of 148 stored “full stop” rows have actual stop times.
- The other 33 are far-clearance episodes mislabeled as stops.
- At 45 mph, four actual stops were observed before termination; 14 additional “stops” are censored far-clearance cases. A comparable full-stop rate is unknowable from retained data.
- `1.474 s` is the mean among the 115 observed stops, not an unbiased all-scenario comparison.
- `7.405 m` and `12.714 m/s³` reproduce the archived pipeline, but their measurement windows differ from fixed profiles.

**Status:** **CURRENT intended final model; mixed evidence.** Overall archived collision count is confirmed descriptively. High-speed, fastest-stop, largest-margin, and worst-comfort comparisons are inconclusive.

## Experiment family 11: Matched 200-configuration SAC-versus-profile comparison

**Purpose:** Evaluate all controller families on the same scenario parameter combinations.

**Relevant files:**

- `src/eval_sac_on_sweep.py`
- `src/plot_unified_comparison.py`
- `src/plot_comfort_safety.py`
- `src/plot_speed_ttc_comparison.py`
- `src/plot_presentation_summary.py`
- `src/runs/20260417_202355/`

**Strong part:** The SAC evaluator reconstructs the same 200 unique configuration values used in the fixed grid, and the model acts deterministically.

**Broken part:** Configuration matching did not guarantee protocol matching:

- RL ends immediately on far-cross clearance; fixed runs continue three seconds.
- RL labels every noncollision termination as a full stop.
- Jerk accumulation begins at different phases.
- Minimum-distance sampling uses different conditions.
- Time-to-stop in RL is censored by early far termination.

**Status:** **INCONCLUSIVE for most cross-controller metrics.** The experiment design remains the correct idea, but the implementation needs one shared episode/outcome/metric protocol and a rerun.

## Experiment family 12: Brake-curve inspection

**Purpose:** Inspect how SAC's applied brake evolves compared with analytical fixed-profile targets on the same states.

**Files:** `src/inspect_brake_curves.py`, `src/brake_curves_data.csv`, `src/brake_curves_5episodes.png`.

**Potential value:** Reveals pre-braking, oscillation, delay, and ramp behavior more directly than aggregate outcomes.

**Current problem:** The saved `ego_speed` column is populated using TTC in the current implementation. Model default is also v2, not v3.

**Status:** **PARTLY BROKEN / NEEDS REPAIR before reuse.**

---

# Models, Algorithms, RL, and Safety Components

## Soft Actor-Critic

SAC was chosen because the action—brake target—is continuous. The presentation also framed SAC as appropriate for smooth control and sample reuse through off-policy learning.

Current SB3 construction in `train_sac.py`:

| Setting | Value |
|---|---|
| Algorithm | Stable-Baselines3 SAC |
| Policy | `MlpPolicy` |
| Action dimension | 1 |
| Replay buffer | 100,000 |
| Batch size | 512 |
| Learning rate | `3e-4` |
| Target smoothing coefficient `tau` | `0.005` |
| Entropy coefficient | `auto` |
| Training steps | 1,600,000 |
| Checkpoint interval | 20,000 |
| Explicit seed | None |

Architecture details not explicitly overridden use SB3 2.8.0 defaults. Do not claim a custom network shape without inspecting the model metadata or SB3 default for that version.

## Observation space

All six observations are normalized to `[0,1]`:

| # | Observation | Why it exists |
|---:|---|---|
| 1 | Ego speed | Braking need depends strongly on speed. |
| 2 | Obstacle proximity | Represents penetration into the LiDAR trigger zone. |
| 3 | TTC urgency | Combines distance and speed into time pressure. |
| 4 | Previous applied brake | Gives the policy action-history context for smoother control. |
| 5 | Hazard flag | Explicit binary indication that classical detection is active. |
| 6 | Stopping feasibility | Encodes available distance relative to simplified required stopping distance. |

The sixth feature was the principal v2 state redesign. When no hazard exists, feasibility defaults to “safe.” Its calculation inherits the same `a_eff = 3.5 m/s²` approximation as the classifier.

## Action and actual authority

The policy outputs a brake target in `[0,1]`. That target replaces the fixed-profile target and passes through the shared ramp limiter.

However:

- Classical code computes the hazard signal.
- Classical code owns route steering.
- Classical code owns cruise throttle/brake.
- In `CRUISE`, hazard-ramp brake state is cleared.
- The SAC action becomes the actual applied hazard brake only in `HAZARD_BRAKE`/related state-machine behavior.

Therefore, SAC does not learn when an arbitrary raw scene is hazardous. It learns a brake response conditioned on engineered LiDAR/TTC/hazard features and a classical mode gate.

## Reward formulation

The live reward implementation has three reported components:

### Safety

- One-time collision penalty: `-300`.
- Per-time TTC danger penalty below 3 seconds.
- Positive urgent-braking term proportional to brake strength and TTC danger.

### Comfort

- Jerk penalty during hazard response: weight `0.05`.
- Brake-command change penalty: weight `0.05`.

### Efficiency/outcome

- Small cruise-speed tracking bonus: weight `0.05`.
- One-time full-stop bonus: `+20`.
- Pedestrian-distance margin bonus: up to approximately `+10` through `tanh(distance/5)`.

The presentation diagram says “Safety + Comfort,” but current code clearly includes efficiency/outcome terms. The more accurate description is **safety + comfort + efficiency/outcome shaping**.

The reward was deliberately shifted toward decisive braking after five-input SAC underperformed in the nominal borderline region. This creates a genuine multi-objective tension: collision avoidance, stopping margin, command smoothness, jerk, and unnecessary braking may not be optimized by the same weights.

## Classical safety state machine

The four main states are:

- `CRUISE`: PI speed control; hazard ramp cleared.
- `HAZARD_BRAKE`: throttle suppressed; fixed-profile or SAC target is ramped/applied.
- `STOP_HOLD`: holds the vehicle once nearly stopped with hazard present.
- `RECOVER`: releases braking and returns gradually toward cruise; can re-arm immediately.

Consecutive-clear logic/hysteresis was introduced to avoid mode flapping from momentary LiDAR noise. This is a classical safety envelope around the RL policy.

## Collision and outcome logic

Collision uses a CARLA collision sensor, with additional proximity logic in the scenario code. Near and far crossings do not receive identical fallback treatment. Low-impulse far-cross contact may therefore be undercounted.

An actual RL full stop is tracked when speed falls below approximately `0.3 m/s`. The evaluation bug arises later: evaluator code infers “full stop” from `terminated and not hit`, even though `terminated` can mean far pedestrian clearance.

## Comfort metric

Jerk is derived from a low-pass-filtered speed signal:

- Speed EMA alpha around `0.25`.
- Acceleration from the first finite difference.
- Absolute jerk from the second finite difference at 50 Hz.

The filtering is sensible in principle because double differences amplify PhysX tick noise. The problem is not simply the formula; it is that RL and fixed-profile scripts begin collecting jerk over different phases and the RL initialization still appears to create a common extreme peak.

---

# Confirmed Findings

These are the strongest defensible conclusions after reconciling documents, code, and raw outputs. “Confirmed” does not imply external validity or statistical significance.

## 1. A complete parameterized hazard-experiment pipeline was built

**Conclusion:** The project progressed far beyond a demonstration. It can define a scenario, run CARLA, control the ego, trigger a scripted pedestrian, detect hazards, apply a controller, collect metrics, serialize provenance, automate grids, train/evaluate SAC, and generate analysis plots.

**Evidence:** Current source modules, many run-level JSON files, sweep CSVs, model ZIPs, plots, weekly chronology, and final presentation.

**Limitation:** The code is CARLA- and machine-specific; “simulator-agnostic” describes a conceptual pattern, not the implementation.

## 2. Route-conforming perception solved an important geometric problem

**Conclusion:** The early straight/steering-aware cone was replaced by a LiDAR corridor projected onto the planned route, aligning steering, hazard filtering, and encounter placement to one route geometry.

**Evidence:** Weekly log description and current `lidar_utils.py`, `lane_follow.py`, and scenario/environment constants.

**Limitation:** No retained false-positive benchmark quantifies the improvement; the implementation change is confirmed, its magnitude is qualitative.

## 3. Scripted pedestrian control replaced unreliable NavMesh motion

**Conclusion:** Direct `WalkerControl` made the pedestrian path configurable and removed reliance on walker AI/NavMesh in the final scenario.

**Evidence:** Weekly log and current `walker_utils.py`/scenario execution.

**Limitation:** Formal cross-run determinism was not tested statistically.

## 4. The four fixed profiles exhibit a real descriptive tradeoff in the final grid

**Conclusion:** In the retained 800-run grid, step constant recorded the fewest collisions and most full stops; cautious ramp recorded the most collisions and fewest full stops. The controllers did not all produce the same safety/outcome behavior.

**Evidence:** `src/runs/20260417_202355/sweep_summary.csv`:

- Step constant: 38 collisions, 144 full stops, 18 slowed/avoided.
- Proportional: 40, 135, 25.
- Exponential: 42, 123, 35.
- Cautious: 46, 113, 41.

**Limitation:** One route/hazard/vehicle and one run per configuration; no confidence intervals. The word “safest” should mean lowest recorded collision rate on this grid, not universal superiority.

## 5. The initial fixed far-cross RL setup supplied the wrong learning pressure

**Conclusion:** A fixed far-cross pedestrian often cleared without collision, allowing SAC to learn coasting rather than useful braking. The training distribution was redesigned in response.

**Evidence:** Detailed Week 9 diagnosis and subsequent `sample_config()` implementation.

**Limitation:** Exact early model artifact mapping is unclear.

## 6. v3-1600k is the intended final semester policy

**Conclusion:** `sac_v3_1600k.zip` is the model generation intentionally used for the final presentation and comparison.

**Evidence:** Presentation slide 7/notes, Weeks 12–13, current `train_sac.py`, model timestamp, and final CSV label.

**Limitation:** The exact ZIP bytes used by the evaluator are not cryptographically bound to the CSV.

## 7. Final SAC archived collision count is 40/200

**Conclusion:** The retained matched v3 CSV records 40 collisions over the 200 configuration grid, or 20%.

**Evidence:** Direct archived-row count.

**Limitation:** This is one deterministic-policy pass with no replicated policy seeds. The SAC runner differs from the fixed runner in episode horizon, termination, and metric collection behavior, so “exactly equal protocol” is too strong.

## 8. Final fixed collision rates and archived SAC collision rate are numerically close

**Conclusion:** On the retained grid, recorded collision rates were 19–23% for fixed rules and 20% for SAC.

**Evidence:** Both final CSVs.

**Proper wording:** “SAC's archived overall collision rate was in the same numerical range as the four fixed profiles.”

**Do not say:** “SAC statistically matched the best profile,” because no equivalence test, repeated policy seeds, or uncertainty estimate exists.

## 9. Only 115 final SAC episodes demonstrably stopped before termination

**Conclusion:** Of 148 rows labeled as full stops, 115 contain `time_to_stop_s`; 33 do not and are all far-cross clearances.

**Evidence:** CSV plus environment/evaluator termination code.

**Limitation:** Some censored far-clearance cases might have stopped if allowed the same three-second tail as fixed runs. Therefore 115 is an observed-before-termination count, not necessarily the fair eventual-stop total.

## 10. The reported SAC numerical summaries can be reproduced as pipeline statistics

The retained matched CSV reproduces:

- Mean minimum pedestrian distance: approximately `7.405 m`.
- Noncollision mean jerk: approximately `12.714 m/s³`.
- Mean time-to-stop among the 115 timed stops: approximately `1.474 s`.
- Stored borderline-avoidable collision count: `8/40 = 20%`.

**Limitation:** Reproduction shows that the presentation numbers came from the data. It does not make cross-controller interpretations valid. Measurement-window, censoring, and classifier problems remain.

---

# Tentative and Inconclusive Findings

## 1. “SAC had a high-speed full-stop advantage” — not established

**What seemed meaningful:** Presentation/log plots showed an approximately 45% SAC full-stop rate at 45 mph versus roughly 15–22.5% for fixed profiles.

**Why uncertain:** At 45 mph, archived SAC rows contain:

- 4 timed actual stops,
- 14 far-cross clearances mislabeled as full stops,
- 18 collisions,
- 4 other far truncated/slowed outcomes.

Four stops were directly observed, establishing a 10% observed-before-termination rate. Fourteen clearance terminations and four other far-cross truncations leave eventual stopping unresolved; no defensible upper bound or fair comparable rate can be recovered from the CSV.

**Needed:** Rerun all controllers with one termination reason and common post-event horizon.

## 2. “SAC stopped fastest” — not established comparatively

**What was observed:** Mean `1.47 s` among 115 SAC rows with stop times, numerically lower than profile summaries.

**Why uncertain:** SAC far episodes terminate at pedestrian clearance. Slower eventual stops are selectively censored, while fixed runs continue for three seconds. The SAC mean is conditional on stopping early enough to be observed.

**Needed:** Shared horizon; compute time-to-stop for every controller from the same event origin; report censored/non-stopping cases separately.

## 3. “SAC was least comfortable” — plausible, but not demonstrated by the current comparison

**What was observed:** Archived SAC noncollision mean jerk `12.71 m/s³`, versus profile summaries around 8–10 or 9.3–9.9 depending on the documented aggregation.

**Why uncertain:** SAC accumulates jerk from near the start of the episode, including pre-trigger acceleration, while fixed runs accumulate after trigger. All final SAC rows share an identical extreme maximum jerk around `408.16`, suggesting a common initialization transient survived the attempted warm-up exclusion.

**Needed:** Compute filtered jerk over exactly the same trigger-to-end window, separate onset jerk from steady braking, and consider skid/wheel-lock telemetry.

## 4. “SAC provided the largest pedestrian safety margin” — not cleanly comparable

**What was observed:** SAC archived mean minimum distance was around 7.40 m.

**Why uncertain:** SAC and fixed runners sample minimum distance under different ahead/post-trigger conditions and different episode tails. The metric is center-to-center distance, not necessarily bumper-to-pedestrian clearance.

**Needed:** One geometric definition and one sampling window.

## 5. “Failures concentrated near physical limits, proving meaningful learning” — suggestive, not proven

**What was observed:** Generic randomized evaluation showed 0/35 collisions in stored `avoidable` cases and 11/11 in `impossible` cases, with intermediate rates in borderline groups.

**Why uncertain:** Checkpoint provenance is missing; classifier is simplified; there is no random/untrained baseline; outcome labels are partly defective; evaluation scenarios were not tracked as a formal holdout.

**Needed:** Corrected evaluation with model hash, actual-speed labels, explicit holdout generation, and baseline policies.

## 6. “v3 fixed the headway blind spot” — plausible, not causally isolated

**What was observed:** v3 documentation reports improved overall/borderline results compared with v2-2400k.

**Why uncertain:** v3 simultaneously changed headway distribution, TTC distribution, and initialization by retraining from scratch. Raw v2 matched rows were overwritten. No ablation separates the changes.

**Needed:** Controlled training runs that vary one factor at a time across multiple seeds.

## 7. “v2-2400k degraded because of entropy collapse or replay-buffer reset” — unsupported causally

**What was observed:** Continued training appeared worse on several stored plots/log summaries.

**Why uncertain:** No TensorBoard curves, replay buffer, or training manifest survives. Retained entropy-coefficient values do not establish collapse. Headway distribution was another confound.

**Needed:** Treat this only as a historical hypothesis unless the behavior is deliberately reproduced with logging.

## 8. “The framework is fully repeatable and simulator-agnostic” — overclaimed

**What supports it:** Synchronous ticks, scripted walker, explicit grid, and serialized configurations improve repeatability. The conceptual separation among scenario, metrics, and policy could be ported.

**Why uncertain/incorrect as stated:** Unseeded training, no repeat-run study, hard-coded CARLA paths/types/map/spawns, no other simulator, and no portability layer.

**Needed:** Repetition tests plus either more careful language (“structured CARLA framework”) or an actual second environment/abstraction.

---

# Failed, Abandoned, and Superseded Approaches

This history prevents repeating old dead ends without understanding why they changed.

| Approach | Why it was tried | What happened | Final status / lesson |
|---|---|---|---|
| RGB camera + autopilot | Validate CARLA and sensor plumbing | Worked as an initial test | **HISTORICAL.** Not part of final perception/control. |
| Local/Town03 waypoint experiments | Learn direct route following | Produced a custom baseline but had lane/intersection limitations | **SUPERSEDED** by `Town04_Opt` global-route system. |
| Steering-aware forward LiDAR cone | Filter forward obstacles simply | Caught roadside/adjacent structure on curves and caused hazard blips | **SUPERSEDED** by route-conforming lane noodle. |
| CARLA walker AI/NavMesh | Move a pedestrian naturally | Spawn/navigation unreliable in chosen setting | **ABANDONED** for deterministic `WalkerControl`. |
| Single proportional brake rule | Initial continuous AEB response | Useful baseline but could not represent alternative tradeoffs | **SUPERSEDED** by four profiles. |
| Fixed far-cross SAC | Simplest initial RL task | Policy learned to coast because hazard often cleared | **FAILED/SUPERSEDED.** Training distribution must contain consequence-bearing cases. |
| Five-observation SAC | First functional RL formulation | Learned behavior but appeared weak near stopping boundary | **SUPERSEDED** by stopping-feasibility state. |
| Training only at 2.5 s headway | Simplified early sampling | Suspected evaluation blind spot at 5 s | **SUPERSEDED** by v3 headway randomization. |
| Binary avoidable/impossible labels | Separate agent fault from physics | Too brittle around the boundary | **SUPERSEDED** by four margin bands, which remain approximate. |
| v2-2400k as continued improvement | Test more training | Mixed/degraded documented result | **SUPERSEDED** by fresh v3. |
| Software ABS / wheel-lock handling | Improve realism and interpret jerk | A `use_abs` software-pulsing branch and intended ABS sweep dimension were briefly implemented in archived code, but no retained config/result contains `use_abs`; the branch was removed from final source | **ABANDONED/INCOMPLETE.** The implementation existed, but no completed experiment is evidenced. |
| PPO comparison | Stability/algorithm baseline | Discussed in log/presentation | **NOT IMPLEMENTED.** No PPO artifact exists. |
| Radar/RGB fusion | Broader perception | Listed as future work | **NOT IMPLEMENTED.** Early RGB capture is not multimodal RL. |
| Stalled vehicle/cut-in hazards | Generalize hazard framework | Listed as future work | **NOT IMPLEMENTED.** Only pedestrian intrusion was completed. |

---

# Bugs, Problems, and Technical Limitations

## Known current problems

### 1. Far-cross clearance is mislabeled as full stop

**Severity:** Critical to reported results.

`carla_aeb_env.py` terminates an episode when the far pedestrian reaches its endpoint. `eval_sac.py` and `eval_sac_on_sweep.py` infer full stop from noncollision termination. This converts pedestrian clearance into a vehicle stop.

**Impact:** 33/148 final matched “full stops” and 16/64 generic random-eval “full stops” lack actual stop evidence.

**Fix direction:** Return/store explicit `termination_reason` and define `full_stop` only from the environment's speed-threshold flag.

**Historical sibling bug already recognized in the code:** The comment immediately above `ped_crossed` documents an earlier near-cross failure: reaching the lane-center endpoint used to terminate the episode before a collision could register, producing an `impossible + full_stop` misclassification. The current code fixed that case by allowing `ped_crossed` termination only for far crossings. The remaining defect is the same underlying semantic mistake one layer later: evaluators assume every noncollision `terminated` episode means the ego stopped. Far-cross clearance may reasonably remain a terminal event, but it must be reported as `pedestrian_cleared`, not `actual_stop`. If eventual-stop rate is a comparison metric, all controllers must additionally receive the same post-clearance observation horizon.

### 2. SAC and fixed-profile episode horizons differ

**Severity:** Critical to full-stop/time comparisons.

Far SAC episodes end immediately at clearance; fixed runs continue for a three-second settling tail.

**Impact:** Eventual SAC stopping is censored, and collision/opportunity windows differ.

**Fix direction:** One shared post-event tail and one runner-independent outcome state machine.

### 3. Jerk measurement windows differ

**Severity:** Critical to comfort claims.

Fixed profiles accumulate after the pedestrian trigger; SAC accumulates from tick 11 onward, including pre-trigger acceleration. An identical maximum jerk across all SAC rows suggests a systematic startup transient.

**Fix direction:** Define a common analysis window, initialize filters from actual initial speed, and save per-tick traces for validation.

### 4. Minimum-distance definitions differ

**Severity:** High for safety-margin claims.

SAC records center-to-center pedestrian distance every tick; fixed code uses a post-trigger/ahead condition.

**Fix direction:** Specify one geometric clearance definition and measure it over the same event window.

### 5. Avoidability uses nominal target speed

**Severity:** High for physics-stratified conclusions.

The classifier is computed before the run using target MPH even though actual trigger speed can differ, especially at shorter encounter distances. Reclassification changes category counts and the final borderline rate.

**Fix direction:** Label after trigger using logged speed and actual available distance/detection state; preserve both nominal and measured labels.

### 6. Nominal headway exceeds LiDAR range

**Severity:** Conceptual/experimental.

At 35 mph, five-second headway plus base distance is approximately 83.2 m; at 45 mph it is around 105.6 m. LiDAR range is 50 m. Even 2.5-second headway at 45 mph gives about 55.3 m.

**Impact:** `brake_headway_s` does not produce the nominal detection distance in these cases. Interpret it as an upper desired threshold clipped by sensing.

### 7. `inspect_brake_curves.py` corrupts a labeled field

**Severity:** High for that diagnostic only.

The output column named `ego_speed` is assigned TTC. Existing `brake_curves_data.csv` should not be used as a speed trace.

### 8. Possible far-cross collision undercount

**Severity:** Uncertain but potentially important.

Collision-sensor impulses may miss some low-contact events, while the extra proximity fallback is near-cross-specific. Far-cross episode termination can also reduce the window for late contact.

**Fix direction:** Use a geometry-based clearance/contact criterion consistently for near/far and retain collision sensor impulse traces.

### 9. Matched SAC output is destructive

**Severity:** High for provenance.

`eval_sac_on_sweep.py` writes the same filename in the sweep directory. Raw v2 rows were overwritten by later evaluation.

**Fix direction:** Write timestamped/model-hash experiment directories and refuse overwrite by default.

Other fixed-name outputs also overwrite silently or reuse colliding names:

- `train_sac.py`: `sac_v3_1600k.zip` and v3 checkpoint names.
- `eval_sac.py sac_v3_1600k`: `eval_results_sac_v3_1600k.csv`.
- `brake_calibration.py`: `brake_calibration_results.csv`.
- `inspect_brake_curves.py sac_v3_1600k 5`: `brake_curves_data.csv` and `brake_curves_5episodes.png`.
- Plotters: existing PNGs in `src/` or the selected run directory; `plot_stopping_distance.py` always writes `src/stopping_distance_reference.png`.

### 10. Defaults and documentation are stale

Examples:

- `train_sac.py` header describes v2/older fixed settings, while its main block trains v3.
- `carla_aeb_env.py` comments say five observations, while `STATE_DIM` is six.
- `WORKFLOW.md` says 800k current training and old model defaults.
- `eval_sac.py`, `eval_sac_on_sweep.py`, `continue_sac.py`, and `inspect_brake_curves.py` default to older models; `continue_sac.py`'s five-observation default is structurally incompatible with the current six-observation environment.
- Current evaluation writes a model-specific CSV although workflow text says generic `eval_results.csv`.
- Workflow `inspect_brake_curves` positional arguments are wrong.
- `plot_presentation_summary.py` hard-codes `SAC v2` text around an auto-detected model label.
- `load_town4.py` loads a map that main code immediately replaces.

### 11. No source/environment/model provenance chain

There is no Git history, requirements lockfile, source snapshot stored with training, model hash in CSV, or run manifest linking all components.

**Impact:** Current source strongly resembles the code that trained v3, but exact identity cannot be proven.

### 12. Incomplete seeding

Random evaluation seeds Python/NumPy; training does not comprehensively seed Python, NumPy, SB3, Gymnasium, PyTorch, and CARLA. `ScenarioConfig.random_seed` is not a full simulator seed.

### 13. Checkpoint ambiguity and storage volume

There are hundreds of checkpoints consuming roughly 1 GB. The final top-level model and same-step callback checkpoint differ. Pruning now would destroy evidence; inventory/hash them before any cleanup.

## Historical problems that were solved or mitigated

- Unstable/undershooting PI cruise → controller redesign and slew limiting.
- Throttle/brake conflict and integral wind-up → explicit hazard state machine.
- LiDAR false positives on curves → route-corridor filtering.
- NavMesh walker failure → scripted walker control.
- Fixed-far SAC coasting → randomized and near-weighted training distribution.
- Encounter firing before cruise speed → encounter-distance buffer.
- Overnight sweep interruption → `resume_sweep.py`.
- Five-input urgency ambiguity → sixth stopping-feasibility observation.

These fixes are supported by code/history, but not all were separately benchmarked.

---

# Research and Experimental Limitations

These are validity limitations rather than ordinary software defects.

## Scope and generalizability

- One CARLA version.
- One vehicle blueprint.
- One main map and route.
- One ego start/end pair.
- One hazard family: pedestrian intrusion.
- Two crossing-depth modes, not diverse pedestrian behavior.
- Final grid uses one walker speed, one weather preset, and default friction.
- No real-world validation.
- CARLA vehicle dynamics do not represent production ABS behavior in a validated way.

The framework can *support* more variation, but configuration capability is not evidence that those variations were studied.

## Experimental replication

- One run per controller/configuration in the final grid.
- One retained training run per principal model generation.
- No multiple policy seeds.
- No confidence intervals or hypothesis/equivalence tests.
- No formal deterministic replay study.

The 800-run count sounds large, but the effective replicated sample size for a specific controller/configuration is one.

## Fairness of comparison

- Fixed controllers share a common runner; their comparison is relatively clean.
- SAC shares configuration values but not identical outcome/metric code.
- Fixed controllers receive a common brake ramp; SAC also receives it, but the policy observes prior ramped brake and is classically hazard-gated.
- The SAC policy trained on a continuous/random distribution related to, but not identical with, the fixed grid.

“Matched scenarios” is true at the parameter level, not fully true at the evaluation-protocol level.

## Avoidability assumptions

- Constant deceleration approximation.
- Nominal rather than measured trigger speed in stored labels.
- No explicit perception delay, actuator delay beyond the simplified effective value, road grade, skid dynamics, or sensor clipping.
- Far crossing treated simplistically.
- Threshold bands chosen heuristically.

Avoidability is best used to stratify difficulty for diagnosis, not to absolve policy failures categorically.

## Metric validity

- Center-to-center minimum distance is not physical clearance.
- Jerk is sensitive to filtering, initialization, and interval definition.
- “Full stop” did not consistently mean vehicle stop.
- Time-to-stop excludes censored failures/non-stops and uses inconsistent horizons.
- Collision detection may vary by crossing type/contact impulse.
- Radar charts combine normalized metrics whose underlying definitions are not all comparable.

## Baseline coverage

There is no untrained SAC, random action, PPO, standard AEB, or alternative RL baseline. The four hand-designed profiles are useful but all depend on the same engineered hazard detector and ramp logic.

## Interpretation and causal attribution

Multiple changes were commonly made between model generations: state, reward, sample distribution, headway, initialization, and training duration. This supports iterative engineering but weakens claims about which change caused an outcome.

## Documentation contradictions

- The April 18 paper is internally mixed-tense: it says RL is being considered/presently underway, then calls learned-policy comparison part of the current implementation. Because it has no Methods or Results, neither sentence establishes completed RL; the May 4 deck and repository supersede it.
- The deck says reward was safety + comfort; code includes efficiency/outcome terms.
- The deck says the framework is simulator-agnostic/generalizable; code is CARLA-specific.
- The deck says the setup is fully repeatable; reproducibility mechanisms are incomplete.
- The log's Week 10 “30 episode” discussion includes totals summing to 97.
- The deck's high-speed/full-stop conclusion comes from the evaluator's label bug.
- The presentation references alignment with standardized/NHTSA-style evaluation goals, but this is framing rather than a cited or demonstrated standards-compliance result.

---

# Current State at the End of Previous Work

## What currently exists

**CURRENT / CONFIRMED:**

- A structured `src/` implementation of the pedestrian-intrusion experiment.
- CARLA 0.9.16 configuration for `Town04_Opt` at 50 Hz.
- Custom route steering and PI cruise control.
- Route-conforming LiDAR hazard detection.
- Scripted pedestrian motion with near/far variants.
- Four fixed AEB profiles.
- Config/result schemas and automated sweeps.
- Direct brake calibration and approximate avoidability tooling.
- Six-observation Gymnasium environment.
- SAC training/checkpoint/evaluation scripts.
- Intended final `sac_v3_1600k.zip`.
- Final 800-row profile CSV and 200-row SAC matched CSV.
- Multiple analysis and presentation plots.
- Historical snapshots showing how the system evolved.

## What worked by semester end

Retained artifacts show successful execution of:

- Single classical scenarios.
- Multi-hour automated sweeps.
- Recovery of interrupted sweep rows.
- Fixed-profile comparison.
- SAC training through 1.6M/2.4M-step model generations.
- Randomized policy evaluation.
- Matched-config policy evaluation.
- Output serialization and offline plotting.

This reconstruction did not relaunch CARLA, so the phrase “works” means it demonstrably ran in April 2026—not that every entry point is currently smoke-tested on August 2026 machine state.

## What partially works

- Avoidability labels organize difficulty but are not physics truth.
- SAC evaluation runs but misclassifies some outcomes.
- Matched evaluation shares configs but not one metric/termination protocol.
- Brake-curve diagnostic runs in principle but contains a bad column assignment.
- Workflow notes are useful but stale.
- Configuration supports weather/friction, but final comparative evidence does not cover them.

## What appears broken or scientifically unsafe

- SAC full-stop count.
- High-speed full-stop comparison.
- Cross-controller jerk/comfort comparison.
- Cross-controller time-to-stop comparison.
- Cross-controller minimum-distance comparison.
- Exact provenance of final evaluated v3 bytes.
- Generic random-evaluation checkpoint identity.

## Most relevant retained results

1. `src/runs/20260417_202355/sweep_summary.csv` — strongest clean comparative artifact among fixed profiles.
2. `src/runs/20260417_202355/sac_on_sweep_results.csv` — important final SAC artifact, but must be interpreted using the audit corrections.
3. `src/sac_v3_1600k.zip` — intended final learned policy.
4. `src/eval_results.csv` — historical randomized-evaluation pattern with uncertain model provenance.
5. `src/brake_calibration_results.csv` — simulator braking diagnostic.

## What I believed when I stopped

At semester end, I believed v3 was the strongest checkpoint, roughly matched or beat most fixed rules on safety, more than doubled high-speed full-stop performance, and remained least comfortable. I considered comfort reward tuning the main unresolved policy issue.

## What I should believe now

- v3 remains the intended final checkpoint.
- Its archived 20% overall collision count is real and numerically near the profile range.
- SAC superiority is not established.
- High-speed advantage is unresolved, not confirmed or disproved.
- Comfort disadvantage is plausible but not validly measured against profiles yet.
- The immediate bottleneck is evaluation correctness and provenance, not a lack of another trained model.

## Older files generally to ignore during normal work

- `previous_src_tests/test0_ego_camera/`
- `previous_src_tests/test1_lane_follow_waypoints/`
- `previous_src_tests/test2_braking_via_lidar/`
- `previous_src_tests/test3_rare_hazard_scenario/` unless reconstructing April 5 lineage.
- Early March run directories unless answering a historical question.
- v2 and five-input models unless running a controlled lineage/ablation study.

Never delete them merely because they are not current; they are the only available history in the absence of Git.

---

# Open Questions

## Explicitly left open during the semester

1. Can comfort/smoothness be improved without sacrificing collision avoidance?
2. Should the current safety–comfort tradeoff be accepted or framed as multi-objective optimization?
3. How much do wheel lock/skidding and missing ABS distort stopping and jerk?
4. Does the policy generalize across weather, friction, maps, and broader pedestrian behavior?
5. How would SAC compare with PPO?
6. Can the framework support stalled vehicles and cut-ins?
7. Would LiDAR + radar + RGB improve perception/generalization?
8. Why did v2-2400k degrade, and what training instability or distribution issue mattered?

## Open questions from Week 2 steering work (added 2026-09-18)

**Should braking authority ever be allowed to follow a scripted evasive steering path
instead of always braking for a hazard in the original lane, and if so, how?** Two
independent hand-coded attempts on `experiment/preliminary-steering-test` (see PRs #3
and #4) each failed a different way:

1. A single-width LiDAR corridor check produced a dangerous false "clear" reading (2m
   pedestrian clearance, 0.37s TTC, zero braking engaged in one live validation run).
2. A hardened retry (an additional, wider corroborating corridor) fixed that specific
   danger but introduced tick-to-tick flip-flopping between which corridor governs
   braking, causing real drive-mode instability and a measurable quality regression
   (successful route recovery dropped from 4/12 to 1/12 runs in the validation matrix,
   one run's outcome got worse, jerk nearly doubled).

Both attempts were reverted; the classical controller currently always brakes for the
original, unswerved lane's own hazard reading, exactly as before this investigation.
**Recommended framing for whoever picks this up (see the branch's own worklog for full
numbers):** two different failure modes from two different hand-coded fixes suggests
this is a genuinely hard design problem to solve as hand-tuned classical logic. Consider
instead exposing the swept-path LiDAR clearance and map-drivability signals as
*observations* to the future steering+braking SAC policy and letting it learn when to
trust them, rather than continuing to hand-code an override rule in `lane_follow.py`.
This is a design decision for the eventual expanded `carla_aeb_env.py`, not yet made.

## Questions revealed by reconstruction

1. What is SAC's true full-stop rate under the same three-second tail as fixed profiles?
2. Does any high-speed advantage remain after correcting labels and censoring?
3. Is SAC actually less comfortable under an identical jerk interval/filter initialization?
4. Which exact v3 ZIP generated the final CSV?
5. Does current source exactly match the training/evaluation source?
6. Which checkpoint generated `eval_results.csv`?
7. How often are far-cross collisions missed?
8. How do results change when avoidability uses actual trigger speed and measured available distance?
9. Does a five-second headway have any effect once the 50 m sensor limit is active?
10. Are identical classical configurations repeatable across multiple CARLA launches?
11. Which v3 improvement came from targeted TTC, headway coverage, or fresh initialization?
12. Does the current retained environment run without repair after the break?

---

# Intended and Logical Next Steps

## Previously intended next steps

These are supported by the weekly log and/or final presentation; the source is noted so planned work is not confused with reconstruction advice.

1. **Retune reward for comfort while preserving safety.** Final deck slide 10/Q&A and the late log identify comfort shaping as open work.
2. **Measure skidding/wheel lock and its effect on jerk.** This comes from Week 13 advisor feedback. Building or validating an ABS controller was not clearly committed as a next step.
3. **Expand scenario diversity.** Weather, friction, and maps come from final deck slide 10/Q&A; TTC and walker variation were also continuing themes in the log.
4. **Add hazard families.** Stalled vehicles and cut-ins were named in the final Q&A.
5. **Compare SAC with PPO.** Final deck slide 10 and the log proposed this algorithm baseline; it was never implemented.
6. **Explore multimodal perception.** LiDAR with radar and RGB was final-deck future work.
7. **Continue diagnosing borderline-avoidable cases.** Weeks 11–12 made this the principal learned-policy difficulty region.

## Reasonable new next steps

These are **INFERRED RECOMMENDATIONS** from the reconciliation, ordered by priority.

### Priority 0: Preserve the semester-end evidence

Before running anything:

- Back up the whole repository.
- Hash all top-level models and relevant final checkpoints.
- Make `src/runs/20260417_202355/` read-only or copy it to an archival location.
- Record current file timestamps and environment package versions.
- Initialize version control for future work, without rewriting historical artifacts.

Why first: current scripts overwrite results, and there is no other provenance chain.

### Priority 1: Build one authoritative evaluation protocol

Create a shared evaluation/outcome layer used by fixed and RL controllers:

- Explicit termination reason: `collision`, `actual_stop`, `pedestrian_cleared`, `timeout`.
- Same post-event horizon.
- Same collision detection and geometry fallback.
- Same trigger-relative metric window.
- Same minimum-clearance geometry.
- Same jerk filter initialization.
- Same time-to-stop definition.
- Actual trigger-speed avoidability labels.

Why before new training: otherwise new policies will inherit misleading comparisons.

### Priority 2: Add reproducibility/provenance

Every run should store:

- Model SHA-256 and model path.
- Source revision/commit.
- Full configuration.
- Python/CARLA/package versions.
- Python/NumPy/PyTorch/SB3/environment seeds.
- Controller type and reward version.
- Outcome reason.
- Per-tick or minimally sufficient raw trace.

Use new timestamped directories and refuse overwrites.

### Priority 3: Reproduce the final claim set

After protocol repair:

1. Smoke-test one easy and one hard near/far scenario.
2. Repeat selected identical configs across CARLA launches.
3. Rerun the 200 matched configurations for all five controllers.
4. Use multiple simulation repeats where stochasticity appears.
5. Compute uncertainty intervals.
6. Reassess overall collision, high-speed stops, borderline cases, jerk, clearance, and time-to-stop.

This converts the current historical story into a defensible baseline for senior-year research.

### Priority 4: Establish policy baselines and ablations

- Random/untrained SAC policy.
- Current v3 versus v2 under corrected evaluation.
- Headway-randomization ablation.
- Borderline-oversampling ablation.
- Multiple training seeds.
- PPO only after the evaluation harness is trustworthy.

### Priority 5: Choose the next research contribution

Then choose one bounded direction:

- Multi-objective safety/comfort reward tuning.
- Distributional/generalization study across friction/weather/maps.
- Sensor-gating versus agent-owned hazard decision.
- Additional hazard families.
- ABS/skid-aware braking.

Do not attempt all simultaneously; the semester history shows that changing many variables together makes causal interpretation difficult.

---

# If I Have Not Touched This Project in Months, Start Here

## Conceptual reset

Remember these four points:

1. The project is a **parameterized pedestrian-AEB framework**, not full autonomous driving.
2. SAC outputs only a brake target; route control and hazard activation remain classical.
3. `sac_v3_1600k.zip` is the intended final model.
4. The main unfinished work is **fair evaluation**, not simply more training.

## First 60 minutes

1. Read [Quick Status Snapshot](#quick-status-snapshot), [Current State](#current-state-at-the-end-of-previous-work), and [Bugs](#bugs-problems-and-technical-limitations).
2. Open, in order:
   - `src/train_sac.py`
   - `src/carla_aeb_env.py`
   - `src/lane_follow.py`
   - `src/test3___ped_intrusion_scenario.py`
   - `src/eval_sac_on_sweep.py`
3. Inspect the two final CSVs without editing them.
4. Confirm the CARLA executable and `venv` still exist.
5. Back up/hash the final model and result directory before running any evaluator.

## First smoke-test session

```powershell
& "C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe"
```

In a second terminal:

```powershell
Set-Location "C:\Users\qdruc\OneDrive\Desktop\Carla Project"
.\venv\Scripts\Activate.ps1
Set-Location .\src
python --version
python -c "import carla, gymnasium, stable_baselines3, torch; print('imports OK')"
python test3___ped_intrusion_scenario.py
```

Use windowed CARLA for this first run so route, pedestrian, detection corridor, and braking can be visually checked.

## Before changing code

- Decide how the authoritative v3 model will be identified.
- Make a new version-controlled branch/repository state.
- Write a small corrected-evaluation specification.
- Do not overwrite `sac_on_sweep_results.csv`.
- Do not trust presentation plots as ground truth for full stops or comfort.

## First development task

Implement and test explicit termination/outcome reasons and a shared metric window. Validate them manually on at least:

- Near crossing + actual stop.
- Near crossing + collision.
- Far crossing + pedestrian clears while ego is still moving.
- Far crossing + ego stops before clearance.
- Timeout/truncation.

Only after these cases are correct should the 200-config comparison be rerun.

---

# Important Paths and Files Cheat Sheet

## Core code

| Path | Remember this |
|---|---|
| `src/test3___ped_intrusion_scenario.py` | Classical single-run scenario and fixed-controller metric lifecycle. |
| `src/carla_aeb_env.py` | RL environment, including the problematic far-cross termination. |
| `src/lane_follow.py` | Actual control authority: steering, PI cruise, hazard modes, profiles, SAC override, ramp. |
| `src/train_sac.py` | Current v3 sampler and 1.6M-step training main block. |
| `src/rl_reward_design.py` | Six-state construction and exact current reward constants. |
| `src/scenario_config.py` | Full experiment parameter schema. |
| `src/sweep.py` | Current 800-run fixed grid. |
| `src/eval_sac.py` | Seeded randomized evaluator; old default and outcome bug. |
| `src/eval_sac_on_sweep.py` | Matched evaluator; output overwrite and outcome bug. |
| `src/avoidability.py` | Approximate nominal-speed physics labels. |

## Final artifacts

| Path | Remember this |
|---|---|
| `src/sac_v3_1600k.zip` | Intended final policy; exact evaluated bytes not proven. |
| `src/runs/20260417_202355/sweep_summary.csv` | Final 800-row fixed-profile dataset. |
| `src/runs/20260417_202355/sac_on_sweep_results.csv` | Final 200-row SAC dataset; contains mislabeled far-clearance stops. |
| `src/eval_results.csv` | Generic 100-episode result; model provenance unclear. |
| `src/brake_calibration_results.csv` | Direct brake calibration data. |
| `src/checkpoints/` | Large model lineage archive. |

## Historical evidence

| Path | Remember this |
|---|---|
| `previous_src_tests/test0_ego_camera/` | Initial camera/autopilot plumbing. |
| `previous_src_tests/helper0_map_spawnpoint_explorer/` | Map/spawn exploration. |
| `previous_src_tests/test1_lane_follow_waypoints/` | Early direct controller. |
| `previous_src_tests/test2_braking_via_lidar/` | Early cone-based LiDAR AEB. |
| `previous_src_tests/test3_rare_hazard_scenario/` | April 5-era snapshot, not current. |
| `notes/CARLA_Self_Driving_RL_Project_Notes_v1.md` | Early concept notes; many ideas were superseded. |
| `WORKFLOW.md` | Related April 17 handoff artifact, partially stale; exact mapping to the final Week-14 organization work is unclear. |

## Original research materials outside the repository

| Source | Path |
|---|---|
| Final presentation | `C:\Users\qdruc\OneDrive\Conn-OneDrive\y3 Spring 2026\COM496-Spring2026\Safety-Focused RL for Autonomous Driving in CARLA-Quentin’s MacBook Air.pptx` |
| Weekly log | `C:\Users\qdruc\Downloads\Quentin Drucker - 2026s COM496 Weekly Progress.xlsx` |
| Paper draft | `C:\Users\qdruc\Downloads\Paper_v2_COM496.docx` |

---

# Important Terminology and Concepts

| Term | Meaning here |
|---|---|
| **AEB** | Automatic Emergency Braking. The actual completed research scope. |
| **Ego vehicle** | The controlled Tesla Model 3. |
| **Rare hazard** | A deliberately generated safety-critical event, here a pedestrian intrusion. |
| **SAC** | Soft Actor-Critic, an off-policy continuous-control RL algorithm. |
| **Gymnasium environment** | Interface exposing `reset`, `step`, observation/action spaces, reward, and episode endings to SB3. |
| **Lane noodle** | Route-conforming corridor used to retain LiDAR points near the ego's planned path. |
| **Trigger TTC** | Configured time until nominal ego arrival at the encounter when the pedestrian is triggered. This is not necessarily the live LiDAR TTC. |
| **Live TTC / TTC urgency** | Approximate obstacle distance divided by ego speed, normalized for the policy/reward. |
| **Brake headway** | Speed-scaled desired hazard-detection time gap. Actual detection is capped by 50 m LiDAR range. |
| **Penetration** | Normalized depth into the trigger zone: zero near outer edge, one near panic distance. |
| **Near crossing** | Pedestrian moves to around lane center and remains in the ego path. |
| **Far crossing** | Pedestrian continues across and can clear the lane. |
| **Stopping feasibility** | Available obstacle distance relative to simplified required stopping distance, normalized for v2/v3 state. |
| **Avoidability margin** | `v×TTC − v²/(2a_eff)` under the simplified classifier. |
| **Borderline-avoidable** | Stored margin from 0 to 8 m: nominally possible but close to the assumed boundary. |
| **Jerk** | Rate of change of acceleration; used as a comfort proxy. Highly sensitive to filtering/window definition. |
| **Synchronous mode** | The Python client explicitly advances CARLA at fixed 0.02 s ticks. |
| **Ramp limiter** | Limits brake-command change per second; applied to both fixed-profile and SAC brake targets. |
| **Hazard gating** | Classical LiDAR/state logic determines when learned hazard braking is applied. |
| **Full stop** | Should mean ego speed below the stop threshold. Some archived evaluator rows incorrectly use this label for far pedestrian clearance. |
| **Matched grid** | Same 200 scenario parameter combinations for SAC and fixed profiles; not currently the same complete evaluation protocol. |

---

# Uncertainties and Things That Still Need Verification

| Uncertainty | Why unresolved | How to resolve |
|---|---|---|
| Exact v3 model bytes used for final CSV | CSV records a name, not a hash; top-level and callback ZIP weights differ | Hash models, inspect any external notes/logs, otherwise designate an authoritative model and rerun |
| Current source equals training source | No Git/source manifest embedded in SB3 ZIP | Treat as strongly likely but unproven; establish version control for all future results |
| True high-speed SAC full-stop rate | Far clearance censored and mislabeled | Correct common protocol and rerun |
| True comparative SAC comfort | Different jerk windows/initialization | Common event window and trace validation |
| True comparative time-to-stop | Selective censoring | Common tail and survival/non-stop-aware reporting |
| True comparative pedestrian clearance | Different distance definitions/windows | Shared geometry and window |
| Model behind `eval_results.csv` | No model field; file predates current output naming | Search external copies/notes; otherwise preserve as unknown and rerun |
| Whether evaluation scenarios were held out | Training scenario identities were not logged | Create explicit seeded held-out manifest for future work |
| Cause of v2-2400k degradation | Several changes/confounds; no curves or buffer | Controlled multi-seed reproduction/ablation |
| Cause of v3 improvement | Headway, TTC distribution, and fresh initialization changed together | Factorial or one-change-at-a-time ablation |
| Accuracy of `a_eff = 3.5` | Simplified and data/calibration story evolved | Recalibrate by speed/friction/headway using actual trigger state |
| Far-cross collision undercount | Sensor/proximity logic differs; early termination | Add uniform geometric collision/clearance audit |
| Same-config repeatability | Claimed observationally, never quantified | Repeat selected configs across fresh simulator launches |
| Current environment health | Not launched during reconstruction | Perform the smoke-test procedure |
| Exact early run/model lineage | No Git and some retrospective log edits | Use timestamps/metadata only; preserve uncertainty if no external archive exists |
| 300k fixed-far artifact identity | Log/model names do not align cleanly | Inspect external backups/training terminal logs if any |
| Realism of skidding/ABS behavior | CARLA model not validated against production ABS | Add wheel-slip telemetry and controlled physics study; avoid real-world claims |

---

# Source Materials Used

## Final presentation

`Safety-Focused RL for Autonomous Driving in CARLA-Quentin’s MacBook Air.pptx`

- Final deck last-modified May 4, 2026.
- Seventeen physical slides, including expanded 9A–9E result panels.
- Speaker notes contain important methodological and interpretive claims.
- Primary source for what I believed and presented at semester end.
- Numerical claims were checked against retained CSVs where possible.

## Weekly research log

`Quentin Drucker - 2026s COM496 Weekly Progress.xlsx`

- Weeks 0–14, including goals, time, detailed progress, failures, pivots, model versions, and future plans.
- Primary source for chronology and why decisions changed.
- Some entries were updated retrospectively and contain inconsistencies; later weeks supersede early impressions.

## Semester paper draft

`Paper_v2_COM496.docx`

- Dated April 18, 2026 in the document.
- Contains title, Introduction, Related Works, and references.
- Useful for motivation, literature framing, and early system description.
- Does not contain final Methods, Results, Discussion, Conclusion, figures, or empirical tables.
- Its “RL being considered/underway” language predates final SAC implementation; it is not evidence for final results.

## Current repository

`C:\Users\qdruc\OneDrive\Desktop\Carla Project`

- Current source in `src/`.
- Historical code in `previous_src_tests/`.
- Early notes in `notes/`.
- Model ZIPs and checkpoints.
- Calibration/evaluation CSVs.
- Per-run JSON and sweep summaries.
- Generated comparison plots.
- Stale-but-useful `WORKFLOW.md`.

## Evidence precedence used here

1. Direct current code and raw retained results for what was implemented/calculated.
2. Weekly log for chronology and historical interpretation.
3. Final presentation for semester-end narrative and claimed conclusions.
4. Paper for motivation and early framing, not final empirical evidence.

This order is not absolute: current code may have changed after an experiment, and raw results may not capture source provenance. Those cases are marked unclear rather than silently resolved.

## Quick claim-to-source map

| Claim family | Principal supporting sources |
|---|---|
| Motivation and original framing | Paper Introduction ¶¶1–4; final deck slide 2 and notes; early weekly entries |
| Lane following, PI control, LiDAR corridor, and pedestrian framework | Paper Introduction ¶¶3–4; final deck slides 3–4; Weeks 2–7; current core source |
| Four fixed profiles and 800-run design | Final deck slides 5–6; Weeks 8–11; `sweep.py`; final `sweep_summary.csv` |
| SAC formulation and final v3 lineage | Final deck slide 7; Weeks 9–13; `train_sac.py`; model metadata/artifacts |
| Randomized evaluation claims | Final deck slide 8; Weeks 10–12; provenance-unclear `eval_results.csv`; evaluator audit |
| Matched comparison claims and corrections | Final deck slide 9/9A–9E; Weeks 11–12; final two CSVs; environment/evaluator code |
| Semester-end conclusion | Final deck slide 10 and Q&A; Weeks 12–14 |
| Previously intended future work | Final deck slide 10/Q&A; Weeks 11–13 |
| Current validity warnings | Repository code/data reconciliation performed in August 2026 |

---

# Summary: Where I Am Now

The project successfully built a structured research platform for controlled pedestrian hazards in CARLA. It includes custom driving control, route-aligned LiDAR perception, deterministic pedestrian scripting, multiple AEB baselines, safety/comfort metrics, sweep automation, RL integration, trained SAC models, and retained experimental artifacts. That is the main confirmed accomplishment.

The cleanest empirical result is the fixed-profile comparison: on the final grid, step constant recorded the lowest collision rate at 19%, followed by proportional at 20%, exponential at 21%, and cautious at 23%. These are descriptive results for one simulated route/hazard configuration family, not universal rankings.

The intended final learned policy is SAC v3-1600k. Its archived matched evaluation records a 20% collision rate, numerically within the fixed-profile range. That supports a cautious statement that the learned controller reached similar recorded overall collision frequency on this grid. It does not prove statistical equivalence or superiority.

The main semester-end headline—that SAC stopped much more often at high speed while being less comfortable—cannot presently be treated as established. Far-cross clearance was mislabeled as full stop, SAC episodes were censored earlier, and comfort/clearance metrics used different windows. These are fixable evaluation problems, but they require a rerun; no arithmetic correction to the existing summary can reconstruct the missing counterfactual tail behavior.

The project therefore did not end at “SAC won” or “SAC failed.” It ended with:

- a useful and extensible experimental framework,
- a trained adaptive brake policy with promising archived collision behavior,
- meaningful evidence that scenario and training-distribution design matter,
- clear safety/comfort and physical-limit research questions,
- and an evaluation/provenance layer that must be repaired before the next scientific claim.

The most important continuation direction is to preserve the current evidence, unify outcome and metric computation, rerun a reproducible matched baseline, and only then choose whether the senior-year research contribution should be comfort-aware RL, generalization, new hazards, alternative algorithms, or skid/ABS-aware braking.
