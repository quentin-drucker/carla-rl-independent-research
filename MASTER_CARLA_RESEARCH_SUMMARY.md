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
| **What is the strongest result?** | On the final 200-configuration grid, fixed controllers recorded 19–23% collision rates, while the archived final SAC results recorded 20%. This supports descriptive overall collision-rate proximity on that grid, not statistical equivalence or SAC superiority. **Superseded 2026-10-04:** under the common protocol v3 makes contact in 36.5% of the grid; under the same protocol the fixed profiles score 30-38%; **SAC ranks second-worst of five** (Week 5 comparison). |
| **What major result is not trustworthy?** | The presentation's claimed SAC high-speed/full-stop advantage, and the archived 20% SAC collision rate. **The Phase 1 re-audit (2026-10-04) found v3 makes contact in 73/200 (36.5%) matched scenarios, not 40/200;** the old evaluator and collision checks hid far-cross hits and slow frontal hits. Fixed-profile rates must be re-scored the same way before any SAC-vs-fixed claim. |
| **What remains incomplete?** | Fair cross-controller outcome/metric computation, repeated seeds, model/source provenance, corrected avoidability labels, comfort tuning, ABS/skid analysis, broader hazards/environments, PPO, and multimodal perception. |
| **Where did I leave off?** | v3-1600k was presented as the strongest final checkpoint. I thought it roughly matched the best fixed rules, performed especially well at high speed, and was less comfortable. The later audit preserves only the archived overall collision-rate proximity; the high-speed and comfort comparisons remain unresolved. |
| **What should I do first?** | Do **not** train another policy first. Preserve the artifacts, identify/hash the authoritative v3 ZIP, repair termination/outcome and metric-window inconsistencies, add provenance and seeds, then rerun a fair matched comparison. |
| **Which vehicle physics? (Week 4, 2026-10-01)** | **CARLA's default physics stays primary. Built-in Chrono is a NO-GO for the main experiment and a LIMITED-GO for separate robustness studies only:** it cannot hold a braked car still at low speed in CARLA 0.9.16's bridge. See [the Week 4 decision](#week-4-decision-carlas-built-in-chrono-physics-is-not-the-research-backend-added-2026-10-01). |
| **Does steering help, and how much? (Week 5, 2026-10-10)** | **Scripted full braking plus steering aimed into the "passage" avoids a stationary pedestrian with as little as 0.5 s of warning; braking alone needs 1.2-1.4 s. At 0.4 s or less every option hits.** The passage computation matched its offline check on the live server. This is the reference for the learned steering policy. See "Week 5 (Phase 2, step 1)". |

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
| `src/carla_session.py` | **CURRENT** | CARLA connection, world loading, synchronous-mode helpers, and exception-safe `cleanup_session()` (Week 4). |
| `src/spawning.py` | **CURRENT** | Vehicle/obstacle spawn helpers; current ego blueprint is Tesla Model 3. |
| `src/spectator.py` | **CURRENT utility** | Spectator follow/glide/free camera behavior. |
| `src/run_result.py` | **CURRENT** | Per-run result schema and serialization. |
| `src/run_stats.py`, `src/telemetry_plotting.py` | **CURRENT support** | Per-tick statistics and visualization. |

### RL and evaluation

| Path | Status | Purpose |
|---|---|---|
| `src/carla_aeb_env.py` | **CURRENT** | Six-observation Gymnasium environment around the CARLA scenario. Since Phase 1 (2026-10-04): `collision_signal="legacy"` (default, last semester's semantics, verified unchanged live) or `"geometric"` (adds oriented-footprint pedestrian contact); every collision component is reported in `info`; supports a `"stationary"` pedestrian. |
| `src/rl_collision_signal.py` | **CURRENT** | Pure per-tick collision signal behind `collision_signal`; verified against all 39 test26 runs (`tests/fixtures/test26_contact_cases.json`, built by `src/extract_contact_fixture.py`). |
| `src/eval_policy_encounters.py`, `src/rl_eval_protocol.py` | **CURRENT — use for every reported RL number** | Encounter-protocol evaluation of a learned policy: same onset and 8 s window as test26, `encounter_metrics` scoring, runs past env termination, full provenance (model SHA-256, git state, versions), never overwrites. Keeps the legacy label for auditing only. |
| `src/test27___fixed_profile_rescore.py`, `src/fixed_rescore_protocol.py` | **CURRENT** | Re-runs the archived 4-profile x 200-config matched sweep with traces, scored by the common protocol; per-run reproduction check; `--resume`, `--relaunch-carla`. |
| `src/analyze_controller_comparison.py` | **CURRENT** | SAC vs. fixed profiles on matched scenarios under one protocol: contacts by crossing/speed/avoidability, 0.188 m sensitivity, exact paired McNemar. |
| `src/analyze_rl_reaudit.py` | **CURRENT** | Offline comparison of archived vs. reproduced-legacy vs. protocol outcomes for an evaluation run, plus pedestrian-radius sensitivity. |
| `src/rl_reward_design.py` | **CURRENT implementation/reference** | State construction, reward calculation, current constants, and extensive design comments. |
| `src/train_sac.py` | **CURRENT** | Fresh v3 training: 1.6M steps, targeted/random TTC, randomized headway. Header comments still contain older v2/fixed settings. |
| `src/continue_sac.py` | **CURRENT 6-D utility with broken default** | Can continue a compatible six-observation v2/v3 ZIP when passed explicitly, but its default five-observation `sac_aeb_rand_800k` is incompatible with the current six-observation environment. It resets entropy and does not restore the old replay buffer, so it is not a faithful resume. |
| `src/eval_sac.py` | **LEGACY — labels unreliable (warning header since 2026-10-04)** | Seeded 100-episode randomized evaluation. Default model remains `sac_aeb_rand_800k`. |
| `src/eval_sac_on_sweep.py` | **LEGACY — labels unreliable (warning header since 2026-10-04)** | Runs SAC deterministically on unique configurations from a profile sweep. Defaults to old `sac_aeb_rand_1600k`; explicitly pass v3. |
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

### Physics-backend diagnostics (Week 4)

| Path | Status | Purpose |
|---|---|---|
| `src/physics_backend.py` | **CURRENT (diagnostic support)** | Pure, offline-tested backend selection: Chrono template validation and hashing, `--chrono` server-flag check, evidence-based backend label, non-overwriting run directories. |
| `src/test24___chrono_backend_smoke.py` | **RESEARCH EVIDENCE** | Backend-selectable smoke test (`--backend default\|chrono`). Default completes; Chrono aborts at the initial-state gate (held-brake creep). |
| `src/test25___chrono_hold_diagnostic.py` | **RESEARCH EVIDENCE** | Eight-case stationary-hold diagnostic behind the Week 4 NO-GO decision. Not part of the main research path. |
| `src/encounter_metrics.py` | **CURRENT** | Common, controller-independent evaluation protocol: fixed post-onset window, explicit outcome, oriented-footprint contact/clearance, onset-referenced stopping, route recovery, drivability. Use instead of `RunResult` outcomes for controller comparisons. |
| `src/steer_brake_baseline.py`, `src/test26___steer_brake_baseline.py` | **CURRENT** | Oracle-onset three-mode baseline (no_intervention / brake_only / brake_steer) under default physics, and its live runner (`--rescore` re-scores saved traces). |
| `docs/MANUAL_CARLA_DRIVING.md` | **CURRENT** | Commands for driving a car by hand in CARLA, with or without Chrono. |

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

**Superseded as a performance claim (2026-10-04):** the Phase 1 re-audit reproduced these 40 labels exactly, but the common protocol finds **73 contacts (36.5%)** on the same grid. See [Phase 1 re-audit](#phase-1-week-5-trustworthy-rl-environment-and-the-v3-re-audit-added-2026-10-04). 40/200 remains correct as a description of the archived CSV only. **Under the same protocol the fixed profiles score 30-38%, and SAC ranks second-worst** ([Week 5 comparison](#week-5-sac-v3-versus-the-fixed-profiles-under-one-protocol-added-2026-10-05)).

**Evidence:** Direct archived-row count.

**Limitation:** This is one deterministic-policy pass with no replicated policy seeds. The SAC runner differs from the fixed runner in episode horizon, termination, and metric collection behavior, so “exactly equal protocol” is too strong.

## 8. Final fixed collision rates and archived SAC collision rate are numerically close

**Conclusion:** On the retained grid, recorded collision rates were 19–23% for fixed rules and 20% for SAC.

**Status (2026-10-04): no longer supportable as a comparison.** The Phase 1 re-audit found v3 makes contact in 36.5% of the grid under the common protocol; the fixed-profile rates came from the same flawed collision checks and have not yet been re-scored. **Done 2026-10-05:** under one protocol, fixed profiles 30-38%, SAC v3 36.5%, second-worst; SAC is significantly worse than step_constant and proportional_ramp on paired scenarios (see the Week 5 comparison).

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

## Week 3 physical-limits finding: low-speed braking transient is largely independent of commanded brake level (added 2026-09-26)

**What was observed:** `src/test9___physical_limits_braking.py`, built on the new
`src/physics_harness.py`/`src/trace_schema.py` shared harness (`experiment/physical-limits-suite`
branch), ran the Tesla Model 3 on `Town04_Opt` at a matched actual entry speed of ~34.1 mph
(requested 35 mph) under four constant brake levels (0.25, 0.50, 0.75, 1.00), CARLA-default
tire friction. At normal speed (>= 5 m/s), peak deceleration scaled with commanded brake
level as expected: approximately -6.1, -6.8, -22.3, and -27.1 m/s² for 0.25/0.50/0.75/1.00
respectively. Below approximately 5 m/s, a sharp deceleration transient appeared in every
run and reached approximately -27 m/s² **regardless of commanded brake level** (-27.1,
-27.2, -27.1, -27.0 m/s² for the same four brake levels) -- i.e. a 0.25 brake command
produced essentially the same terminal deceleration as a 1.00 command once the vehicle
slowed below that threshold. Stopping distance and time still scaled with brake level as
expected overall (25.8 m/2.68 s at 0.25 down to 17.6 m/1.76 s at 1.00), since the low-speed
transient is brief and occurs near the end of the stop.

**Evidence status:** **CONFIRMED** for the tested conditions (reproduced identically across
3 separate live runs in the same session). **NOT YET TESTED** across other speeds,
frictions, or a fresh CARLA server launch -- do not generalize beyond Tesla Model 3 /
`Town04_Opt` / default friction / ~34 mph entry yet.

**Why this matters:** Directly relevant to the advisor's Sep 22 questions ("is throttle/brake
control symmetric", "does max braking do better than partial braking assuming no skidding").
Within this tested range, higher brake commands ARE meaningfully stronger during normal-speed
braking (a real signal for a combined action-space design), but that distinction
disappears near the end of every stop in this vehicle model regardless of command. This is
reported strictly as an observed CARLA vehicle-model characteristic, **not** a skid or
wheel-lock claim: CARLA 0.9.16's Python API exposes no validated per-wheel slip/lock signal
(`wheel_slip_signal="not_measurable"` in every logged case), and CARLA does not model a
validated production ABS system (existing limitation, restated here rather than re-derived).
**Next:** broaden to more speeds/frictions and repeat across a fresh CARLA launch before
treating the ~5 m/s threshold value itself as anything more than descriptive of this run.

## Week 3 physical-limits finding: full-lock steering response and the Sep 22 advisor question (added 2026-09-26)

**What was observed:** `src/test10___physical_limits_steering_lock.py` (same
`experiment/physical-limits-suite` branch/harness) ran the Tesla Model 3 through 12 coasting
(zero throttle) full-lock turns: 3 entry speeds (15/30/45 mph) x left/right x step/ramp
steering command. Across every case that sustained the turn (i.e. did not stop early), the
achieved inner-front-wheel steer angle was consistently ~68.6-70.0° and the turn radius was
consistently ~2.64-2.66 m, essentially independent of entry speed and direction. Coasting
through a full-lock turn also produced severe speed loss from cornering drag alone: 5 of 12
cases scrubbed off effectively all speed within 1.2-1.7 s of applying full lock, well before
the 5 s maneuver window ended (this timing was NOT symmetric between left and right at the
same speed/command in this single session -- **since resolved, not a real directional
asymmetry, see the 2026-09-27 follow-up below**). Body slip angle (velocity heading vs. yaw --
a measurable whole-body kinematic quantity) grew progressively during sustained turns,
reaching 20-29° in the cases that did not stop early.

**Two measurement bugs were caught and fixed during this run, both by inspecting raw traces
rather than trusting summary numbers, and are recorded here because they affect how any
future consumer of `get_wheel_steer_angle()` on this API should read it:**
1. Reading only the front-left wheel's steer angle initially made left and right turns look
   asymmetric (46.7° vs 68.8° at the same speed) despite every other measured quantity
   (yaw rate, turn radius, slip angle) being symmetric. This is Ackermann steering geometry
   (the inner front wheel of a turn steers further than the outer one), not a real vehicle
   asymmetry -- fixed by reading FL and FR independently and reporting whichever has the
   larger magnitude.
2. A single 0.02 s tick showing speed drop from 6.38 to 2.02 m/s (implying over 20g of
   deceleration -- physically impossible) corrupted the slip-angle calculation to >150° in
   several cases. This is the same class of low-speed CARLA physics-engine artifact already
   found in the braking finding above (there, an anomalous deceleration; here, a velocity-
   direction discontinuity), both landing below roughly 5 m/s residual speed. Any future
   kinematic-derivative metric computed from this vehicle model's low-speed state should be
   treated with the same caution.

**Evidence status:** **CONFIRMED** (directly measured) for the achieved wheel angle/turn
radius/drag-induced speed loss pattern, single live session, Tesla Model 3 / `Town04_Opt` /
default friction only -- **NOT YET REPEATED** across a fresh CARLA launch. **No skid,
wheel-lock, understeer, or oversteer claim is made** -- CARLA 0.9.16's Python API exposes no
validated per-wheel slip signal, and this substudy did not attempt to define a measurable
proxy strong enough to support those specific labels.

**Why this matters:** Directly answers part of the advisor's Sep 15/22 question ("is
over/understeer or skidding modeled in CARLA at all") as far as this vehicle model's Python
API can answer it: turning response (radius, wheel angle) is stable and speed-independent at
full lock, but the model does NOT preserve speed through a coasting full-lock turn -- cornering
drag alone can bring the vehicle to a near-stop in ~1-2 seconds. This is directly relevant to
any future steering-plus-braking action space: a policy that swerves hard without also
managing throttle/brake will lose speed rapidly as an emergent consequence of this vehicle
model, not because of any explicit penalty. **Next:** repeat across a fresh CARLA launch;
rollover (2.3) and throttle/brake symmetry (2.5) substudies remain unstarted.

## Week 3 follow-up: the steering-lock near-stop asymmetry is a resolved artifact, not a directional bias (added 2026-09-27)

**What was done:** Quentin asked to revisit unexplained gaps from earlier Week 3 work rather
than move on. Inspected the existing test10 traces (no live CARLA needed) tick-by-tick for
the starkest asymmetric pair, 45mph_left_step vs. 45mph_right_step: both cases track
numerically identically (same speed, mirror-image yaw) through 1.02s, then diverge sharply --
left's speed drops from 7.89 to 0.94 m/s within 8 ticks while right decays smoothly and never
approaches near-stop in the 5s window. The exact divergence tick showed speed going 6.38 ->
2.02 m/s in ONE 0.02s tick (-218 m/s^2, >20g) -- the same tick already documented in
`physics_harness.py` as a low-speed artifact, initially just re-found rather than new.

Built `physics_harness.detect_low_speed_snap_events()` (pure, offline-tested, flags any tick
with |accel_mps2| > 100 -- well above the milder ~27 m/s^2 low-speed braking transient already
characterized separately) and ran it across **all 12** existing test10 traces, not just the
2 inspected by hand.

**Result:** The detector found this snap event in exactly 4 of the 12 cases -- matching 4 of
the original 5 `near_stop_onset` cases -- at DIFFERENT residual speeds (4.56, 4.68, 6.38, 7.82
m/s: no fixed threshold) and in BOTH directions (3 left, 1 right: not a clean directional
split). The 5th case (15mph_left_step) showed zero snap events: its speed decayed completely
smoothly for the full 5s window, reaching near-stop late (4.98s) simply because 15mph doesn't
carry enough energy to stay above 0.5 m/s for a full 5 seconds under continuous cornering
drag alone -- an unrelated, unremarkable mechanism.

**Evidence status: CONFIRMED** (measured directly from existing traces, single session,
Tesla Model 3 / Town04_Opt / default friction) -- the previously-flagged left/right
near-stop-onset asymmetry is fully explained by two separate, now-understood mechanisms, not
one unexplained directional bias. The snap phenomenon is best read as a sharp, narrow-window
transition in CARLA's underlying tire/vehicle model that a trajectory either crosses or
narrowly avoids depending on its exact evolving state (two near-identical mirror-image
trajectories diverge at the single tick one of them crosses it) -- **not** evidence that this
vehicle model handles left and right turns differently. The 4:1 left:right split in this small
sample is consistent with chance, not a real asymmetry.

**Why this matters:** Prevents a future reader (or this project's own advisor report) from
mis-citing the near-stop timing asymmetry as a genuine steering-direction bias in this vehicle
model, when the actual, better-supported explanation is a known artifact class already
documented for an unrelated reason (the braking-transient finding). `detect_low_speed_snap_
events()` is now a reusable, tested harness utility any future substudy can call to
auto-flag this artifact class rather than re-discovering it by hand. **Not yet done:** the
exact mechanism inside CARLA's tire model is still unknown (no per-wheel slip signal exists to
investigate further); this analysis reused the existing dataset rather than a fresh live run,
so it does not by itself satisfy the still-outstanding fresh-CARLA-launch repeat gate for
test10.

## Week 3 physical-limits finding: rollover boundary testing is confounded by roadside infrastructure above ~45 mph at the current test location (added 2026-09-26)

**What was observed:** `src/test11___physical_limits_rollover.py` reused test10's coasting
full-lock setup and increased speed through a bounded, predeclared sequence (30/45/60/75/90
mph x left/right, step steering) at the repository's standard spawn point (`SPAWN_INDEX=242`,
`Town04_Opt`). 7 of 10 cases (all of 60/75/90 mph, plus one 45 mph case) ran off the paved
shoulder and collided with fixed roadside infrastructure (a guardrail, once a light pole)
within 1.1-1.7 seconds -- before any intrinsic vehicle rollover dynamics could develop. This
happens because the vehicle's turning arc during the high-speed portion of a full-lock turn
is much wider than the tight ~2.65 m radius the steering-lock substudy measured once the
vehicle has already slowed down; at highway entry speed that wide arc leaves the drivable
surface. The 3 cases that were NOT confounded (30 mph both directions, 45 mph right) showed
maximum roll under 1.5 degrees and the vehicle stayed upright throughout -- no rollover or
near-miss observed in that narrow tested range. Notably, even the 7 guardrail-impact cases
stayed upright despite a real, speed-scaling vertical bounce (vertical excursion grew from
~0.02 m at low speed to ~0.78 m at 90 mph) -- no tip-over occurred in any of the 10 runs,
though this is reported only as "no tip-over in these specific impacts," not as a general
claim, since the impact itself is exactly the confound preventing a clean rollover test.

**Evidence status:** **CONFIRMED** (measured directly) that this specific spawn location
cannot cleanly test intrinsic rollover dynamics above ~45 mph -- the maneuver runs off the
road into fixed infrastructure first. **NOT YET ANSWERED**: whether this vehicle model can
roll over from steering alone at 60-90 mph, since that speed range was never actually tested
by its own dynamics here. **NOT YET REPEATED** across a fresh CARLA launch.

**Why this matters:** This is a methodology finding as much as a vehicle-dynamics one --
future physical-limits work involving full-lock or otherwise wide-radius maneuvers at highway
speed needs either an open, unobstructed spawn location, or must explicitly detect and label
a map-geometry collision as its own outcome (as this substudy now does) rather than let it
silently corrupt a following-dynamics measurement. **Next:** find or construct an open spawn
location in `Town04_Opt` (or elsewhere) with enough lateral clearance to actually test
60-90 mph rollover dynamics, rather than treating "no rollover observed" from the confounded
result as evidence of anything about that speed range.

## Week 3 follow-up: open-location rollover retest answers the question the confounded result couldn't (added 2026-09-27)

**What was done:** Quentin asked to revisit this specific gap and whether a different
map/spawn point might have more room. Systematic search found Town04_Opt is guardrail-lined
on essentially every road tried (8 locations tested across the highway loop and side streets,
all hit a guardrail/pole within 0.6-1.6s at 60-90 mph), and Town05_Opt scored no better on an
openness scan. Town03_Opt eventually yielded a clean location -- but no single spawn point
was open in both turn directions, so `src/test15___physical_limits_rollover_open_location.py`
uses two spawn points (90 for right turns, 92 for left turns, ~20 m apart, facing roughly
opposite directions), each independently live-verified collision-free across the full target
speed range before being used for data collection. Also introduced
`physics_harness.set_instant_entry_velocity()` (offline-tested pure trig +
`carla.Actor.set_target_velocity()`) in place of the throttle-ramp method for this substudy,
since the ramp covers 100+ meters reaching 90 mph -- far more room than a short street segment
offers, and irrelevant here since this substudy only cares about the coasting phase.

**Result:** All 6 cases (60/75/90 mph x left/right) came back clean -- **zero collisions,
zero confounds.** Every case: `rollover=not_observed_in_tested_range`, upright, no collision.
Maximum roll across all 6 cases was 4.29 degrees (90 mph, right turn); every other case stayed
under 1.0 degree. The 90 mph right-turn case (the highest-roll case, and a brand-new location)
was manually repeated and reproduced exactly (max roll 4.3 deg, final roll 0.4 deg, z-range
0.187 m, upright, no collision both times) -- not a one-off.

**Evidence status: CONFIRMED** (measured directly, single live session, Town03_Opt spawns
90/92, repeated once for the boundary case) -- across the full bounded 60-90 mph range, in
both directions, on an unconfounded surface, this vehicle model does not approach the 45-deg
early-stop threshold (let alone the 60-deg rollover-confirm threshold) at any tested speed.
This is the direct answer to the advisor's original question that the Town04_Opt result could
not provide: full-lock coasting alone, at these speeds, on this vehicle/tire model, does not
produce rollover.

**Scope caveat (important, not a footnote):** this result is on **Town03_Opt**, a different
map from the rest of the physical-limits suite (Town04_Opt) -- surface friction, camber, and
road geometry are not otherwise confirmed identical between maps. This finding should be
reported as "this vehicle's rollover dynamics on Town03_Opt," not silently pooled with the
Town04_Opt braking/steering-lock/throttle-brake findings as if map were held constant across
the whole suite. Above 90 mph remains genuinely untested (this substudy's own predeclared
ceiling, not escalated further); no fresh-CARLA-launch repeat yet (same outstanding gate as
every other Workstream 2 substudy).

## Week 3 physical-limits finding: throttle and brake are NOT physically symmetric at matched [0,1] commands (added 2026-09-26)

**What was observed:** `src/test12___physical_limits_throttle_brake_symmetry.py` (the fourth
and last Workstream 2 substudy) applied 4 matched command magnitudes (0.25/0.50/0.75/1.00) --
throttle from a standing start, brake from a matched actual 35 mph entry speed (same speed as
the earlier braking finding) -- and compared normal-speed (>=5 m/s) peak response magnitudes.
At every level where both sides produced a comparable measurement, the response was
**asymmetric**: braking produced 3.3-4.9x the peak deceleration magnitude of the matched
throttle command's peak acceleration (0.50: 2.05 vs. -6.78 m/s²; 0.75: 4.52 vs. -22.33 m/s²;
1.00: 8.25 vs. -27.18 m/s²). At 0.25, throttle alone could not even reach the 5 m/s
normal-speed regime within a 5-second window (topped out at 8.1 mph) while brake at the same
0.25 command already produced -6.11 m/s² of peak deceleration -- reinforcing the same
asymmetry qualitatively even where the formal comparison is reported as inconclusive for lack
of comparable data, not forced into a number.

**Evidence status:** **CONFIRMED** (measured directly, reproduced identically across 3 live
runs during development) for this vehicle/map/friction/speed -- **NOT YET TESTED** across
other speeds, frictions, or a longer throttle window (which might let the 0.25 level reach a
comparable measurement). Not yet repeated across a fresh CARLA launch.

**Why this matters:** Directly answers the advisor's Sep 22 question ("is throttle and brake
control symmetric?") -- no. **RL-design consequence, stated by the plan itself and confirmed
here empirically:** a combined action space mapping throttle and brake onto a single signed
`[-1, 1]` axis would give the same numeric magnitude very different real physical
consequences depending on sign. Any future combined actuator action likely needs asymmetric
scaling, separate action dimensions, or different rate limits for throttle vs. brake, not a
shared raw numeric range. This completes all four Workstream 2 physical-limits substudies
(steering lock, rollover, braking, throttle/brake symmetry) at least once each; fresh-launch
repeats for all four remain the standing gate before any of these findings should be treated
as fully stable simulator behavior.

## Week 3 addition: observational other-vehicle occupancy check for candidate paths (added 2026-09-26)

**What was added:** `src/vehicle_occupancy.py` (Workstream 1.2) -- a conservative
clear/occupied/unknown check for whether another vehicle currently occupies the commanded
or swept-transition lateral corridor, wired into `lane_follow_step()` as
`commanded_path_occupancy`/`transition_path_occupancy` telemetry, alongside (not replacing)
the existing LiDAR-clearance and map-drivability signals. This is the "check whether a car
is in its way to swerve" gap the advisor raised (2026-09-08). It is purely observational
this week: it does not gate braking or steering, same as the map-drivability check before
it. No prior scenario test had ever included another vehicle actor.

**Validation status: CONFIRMED live** across all four required cases (no adjacent vehicle,
an actor clearly inside the candidate path, an actor just outside it, a deliberately
unavailable side), for both left and right scripted directions -- see
`src/test13___candidate_path_occupancy_validation.py`. A vehicle placed on a candidate path
is consistently reported occupied (575/576 observed ticks); a clearly-clear vehicle never
is; an unavailable corridor reports "unknown", never a silently-promoted "clear".

**Two bugs found and fixed during this work, one of them a real reliability issue beyond
the occupancy feature itself:**
1. The occupancy classifier's conservative circular footprint (built from the vehicle's own
   bounding box) is larger than intuition suggests -- confirmed live, the Tesla Model 3's
   occupancy radius is ~2.70m (from a 2.396m half-length, the larger of its two bounding-box
   half-extents, plus a safety margin), giving a total occupancy threshold of ~4.10m from a
   1.4m-half-width corridor. An initial test margin of 3.5m was not actually outside that
   threshold and was live-caught as a false "occupied" -- not a bug in the check itself, but
   a reminder that this conservative design produces a wider effective footprint than a car's
   visual width alone would suggest.
2. **A latent, pre-existing reliability bug in `test3___ped_intrusion_scenario.run_scenario()`**:
   its `finally` cleanup block called `print_run_summary(stats)` before `stats` was
   necessarily assigned, so any exception raised early in scenario setup (before `stats` is
   set) raised a masking `UnboundLocalError` instead, which skipped the rest of `finally` --
   including `restore_async_mode(world)`. Confirmed live: this left the CARLA world stuck in
   `synchronous_mode=True` with no client ticking it after a setup-time crash. This was
   triggered here by an other-vehicle spawn collision (fixed separately, see below), but the
   underlying `finally`-ordering bug could have masked any other early-setup failure in any
   scenario run, silently leaving the simulator in a bad state for the next script. Fixed by
   initializing `stats = None` before the `try` block and guarding the summary print.

**Secondary finding:** the specific highway spawn location used throughout this project's
tests (`SPAWN_INDEX=242`, `Town04_Opt`) has an asymmetric usable road width -- a probed
spawn point at 7.5m left of route center failed (collided with fixed scene geometry,
consistent with the guardrail already found in the Week 3 rollover substudy), while the same
7.5m to the right, and 7.0m to either side, spawned cleanly. Not yet characterized further;
relevant to any future test needing a wide lateral working area at this location.

## Week 3 addition: trajectory-visualization tooling MVP (Workstream 3, added 2026-09-27)

**What was added:** `src/trajectory_recording.py` (a `TraceTick`-based recorder that plugs
into `run_scenario()`'s existing `tick_observer` hook, no changes to control logic) and
`src/trajectory_plot.py` (pure coordinate/series-preparation functions, offline-testable
against synthetic straight/curved trajectories, plus matplotlib rendering of a world-frame
spatial overlay and a companion time-series panel). `TraceTick` (trace_schema.py) gained a
`requested_lateral_offset_m` field; `lane_follow_step()` and `run_scenario()` additively
expose pose and pedestrian position in their existing telemetry dict (reusing already-read
CARLA state, no new API calls). See `src/test14___trajectory_visualization_mvp.py` for the
assembled live tool.

**Validation status: CONFIRMED live**, per the plan's own 3.3 success criterion. On
Town04_Opt: 3 identical fixed-condition repeats came back bit-for-bit identical (deterministic
sim under synchronous mode -- zero run-to-run spread is the correct/expected result here, not
a failure to detect spread). A first positive control (pedestrian starting side only) came
back nearly indistinguishable from baseline in the spatial plot -- found by inspecting the
actual overlay image, not just the aggregate outcome label, per the standing rule to inspect
traces rather than summaries. Root cause: this scripted controller has no lateral reaction to
which side the pedestrian starts from in near-cross mode, so that variable barely perturbs
the ego's own path -- a real but weak differentiator for this particular tool test. A second
positive control using the already-validated evasive-swerve controller
(`build_evasive_offset_fn`, peak_offset_m=2.5) produced a clearly, visibly distinguishable
spatial curve (CTE mean 0.412m/max 1.735m vs. ~0.035m/0.355m baseline) and confirmed
event-marker (trigger/hazard_activation/closest_approach/termination) placement matches the
raw per-tick position/distance data by hand-checking the CSV.

**One bug found and fixed:** event-marker text labels on the spatial overlay plot rendered on
top of each other illegibly when multiple event types landed near the same position (the
common case for a short straight scenario). Fixed with a fixed per-event-type vertical
label stagger. Not fixed (cosmetic, left as follow-up): labels for the *same* event type
across several overlaid runs that end near the same position still stack.

**Known scope limitation:** route-centerline overlay is implemented and offline-tested
(`prepare_route_reference`), but `run_scenario()` does not currently expose its internal
`route_points_world` to a caller, so the live MVP runs above did not include a centerline
overlay. Not required by the plan's 3.3 success criterion; left for a future pass if this
tool sees continued use.

## Week 3 addition: a real coverage gap found via a far-crossing pedestrian test -- CARLA's collision sensor missed it too (added 2026-09-27, corrected same day)

**What prompted this:** Quentin watched test13 live and noticed the ego always just stops for
the pedestrian rather than visibly swerving around and continuing. His hypothesis was that
the LiDAR hazard corridor governing braking needs to "move with" the ego's swerve. That
specific idea is exactly the approach already tried twice (2026-09-18, see the "Two
attempts... reverted" note near the top of this document) and is an explicit Non-goal in the
Week 3 plan pending real design review -- it was NOT re-attempted here.

**First pass (superseded below):** every existing scenario script (test5/test6/test7/test8/
test13/test14) uses `ScenarioConfig(walker_cross="near")` -- the pedestrian walks to lane
center and stays there indefinitely, which correctly forces a full stop regardless of swerve
quality (the safety net working as intended). No script had ever used `walker_cross="far"`
(pedestrian actually exits the lane), so a new script, `src/test16___far_cross_swerve_and_
merge_back.py`, exercised it for the first time. A first automated run reported
`outcome=slowed_avoided`, `collision_detected=False` -- this was reported as a clean success.
**It was not.** Quentin ran the same script live and watched the ego actually hit the
pedestrian (visually confirmed: the pedestrian's geometry glitched/was shoved as the ego drove
through it), directly contradicting the automated result. Re-verifying with an independent,
offline-tested geometric contact check (`src/pedestrian_contact.py`, using the Tesla Model 3's
real bounding-box dimensions, deliberately built because CARLA's own sensor had just been
shown unreliable here) confirmed a sustained 0.5-second contact window (26 consecutive ticks),
not a single-tick glitch -- strong independent corroboration of what Quentin saw. This is
recorded as a caution against trusting a single clean-looking automated result, exactly the
"inspect traces, not aggregate labels" discipline this project has otherwise followed.

**Root cause, traced from the full per-tick record:**
1. The original-lane hazard corridor only watches a band +/-1.4m from route centerline
   (`NOODLE_HALF_WIDTH_M`). The far-crossing pedestrian ends up at +2.55m from centerline --
   outside that band.
2. Once the pedestrian crosses past 1.4m (~5s into the encounter), the corridor loses them
   completely -- it reports no LiDAR return at all for nearly the rest of the encounter, not
   just briefly.
3. The swerve's peak offset (1.5m) happened to land almost exactly on that same 1.4m boundary
   -- nowhere near the pedestrian's actual 2.55m excursion. Re-testing with a much larger
   offset (3.2m) did NOT meaningfully improve the outcome, since the corridor's blind spot is
   unrelated to how far the ego itself swerves.
4. The scripted recovery controller's clear-confirmation logic deliberately treats a missing
   LiDAR return as "unknown, not confirmed clear" (a real, intentional safety choice from an
   earlier, unrelated 2026-09-18 bug fix) -- but combined with the corridor now returning "no
   data" almost every tick, confirmed clearance (and the return-to-lane it gates) was delayed
   by roughly 3 extra seconds past when the pedestrian was actually long gone.
5. During that multi-second window, the ego cruised at full speed, still holding its offset,
   through a stretch of space nothing was watching: the original corridor structurally cannot
   see it (outside its band), and the commanded/transition corridor checks that DO cover that
   space are observational-only telemetry (see the Workstream 1.2 finding above) that, even if
   wired up, only track vehicle actors, not pedestrians.

**The actual finding:** this is a genuine, previously-undiscovered coverage gap, distinct from
the flagged braking-authority Non-goal. It is not about which corridor's reading should govern
braking -- it is that no existing corridor, watched or observational, covers a pedestrian who
has moved outside the original lane's narrow band but remains physically reachable by the
ego's own chosen escape path.

**Evidence status: CONFIRMED** (a real collision, independently verified two ways -- visual
observation and an offline-tested geometric check -- across a reproducible, deterministic
scenario). **Not done:** no fix implemented. A candidate safe direction exists (a new, purely
additive ego-centered pedestrian-proximity check that could only ever add a hazard-brake
trigger, never remove the original corridor's authority) but this is a real change to
safety-relevant control logic and awaits explicit direction before implementation, consistent
with how every other safety-relevant decision in this project has been handled.

### Week 3 experimental follow-up: ego-rooted swept-path prototype (added 2026-09-27)

An opt-in prototype now evaluates one intended swept path rooted at the ego's exact current
position. Its LiDAR query uses arc length along that path, not Euclidean roof-sensor distance,
and does not apply the older sensor-local forward-X gate. Before a maneuver is committed, the
original route retains braking authority. After five consecutive ticks where the ego is moving
in the requested direction, the ego-rooted path is drivable, and any known pedestrian is
geometrically clear of that path, authority transfers to the swept path and remains there until
the lateral maneuver request ends. Only that active path then controls braking; the original
corridor remains telemetry but cannot keep braking for an obstacle in the abandoned lane.
Occupancy is deliberately not a handoff veto: a vehicle in the new path must instead trigger
braking at its own swept-path distance. This remains opt-in and default-off.

The contact oracle was also corrected for these experiments: a new oriented-rectangle-vs-circle
check uses the Tesla's actual half-length, half-width, and yaw instead of applying its half-length
as a circular radius in every direction. This resolved the earlier ambiguity around a lateral
1.68m center separation: the old circular oracle could label such a lateral pass as contact even
though width-aware geometry gives about 0.30m edge clearance.

Focused live evidence is promising but preliminary:

- Test18's buffered positive case requested a 1.982m offset (0.60m planned margin) with a 0.5s
  smooth shift. Two consecutive live runs both avoided a full stop, completed lane recovery,
  recorded zero oriented-contact ticks, and reported 2.20m minimum pedestrian center distance.
- Test18's deliberately undersized 0.5m case never confirmed swept-path clearance, safely
  full-stopped 6.30m from the pedestrian, and recorded zero oriented-contact ticks.
- Test19 repeated test16's historically unsafe 1.5m far-cross path with the prototype enabled.
  It recorded 125 swept-path LiDAR hazard ticks, zero oriented-contact ticks, and 7.69m minimum
  center distance, ending `slowed_avoided` rather than colliding. It did not complete lane
  recovery within the five-second post-crossing tail, so recovery remains unresolved.
- Test20's nine-run matrix covered both directions, two speeds/TTCs, stationary and moving
  pedestrians, an unsafe far-cross backstop, and a parked-car-blocked escape path. It recorded
  zero oriented-contact ticks in all nine runs; the blocked path correctly retained the stop.
- The full offline suite passed 225 tests after the active-path handoff, stop-hold, and physical
  route-return tracking work below.

Quentin subsequently ran and watched test18 and test19 himself. He visually confirmed the same
behavior: test18's wide swerve cleared and continued, its undersized swerve retained braking,
and test19 continued braking while the ego was outside the original corridor. This independent
visual check is important because CARLA's collision event previously missed a real test16 hit.

Test21/test22 initially appeared to show that the ego-rooted LiDAR tube never saw a parked car
in the swerve lane. That interpretation was first corrected by separating `no_return`,
`detected_beyond_trigger`, and `hazard`: the swept tube was seeing the car, but the original-lane
pedestrian had already reduced speed and shrunk the dynamic braking threshold. A deeper control
issue remained, however: parked-car occupancy vetoed clearance of the old pedestrian, conflating
"the old obstacle is no longer on this path" with "there is a different obstacle farther along
this path." The stable authority handoff above now separates those decisions.

In the final live test22 run, authority transferred to the ego-swept path at t=4.74s. The old
corridor continued to observe the pedestrian for diagnostic purposes but supplied zero active
hazard ticks after handoff. Braking resumed at t=7.02s only when the parked car crossed the
swept-path threshold (`32.64m` measured versus `32.76m` trigger). All 149 active hazard ticks
were sourced from `ego_swept_active`; the ego stopped with 8.92m center separation from the car,
5.76m from the pedestrian, and zero oriented pedestrian-contact ticks.

Test23 removes that ambiguity entirely: no pedestrian is spawned, the ego voluntarily shifts
4.882m, and only a parked Model 3 occupies the shifted path. In the repeated live validation,
the original corridor recorded zero hazard ticks while the swept sensor recorded 609 detections,
148 threshold crossings, and owned the committed maneuver for 886 ticks. It first crossed the
threshold with the car 34.12m away at t=7.02s, matching test22's parked-car timing, and slowed
from 24.8mph to near zero with a 7.55m minimum center separation. This confirms that the swept
tube independently activates braking for the parked vehicle whether or not an irrelevant
pedestrian remains in the original lane.

A proposed lower LiDAR height cutoff (`z_min=-1.8m` instead of `-1.0m`) was tested and rejected:
it admitted road returns, caused 1,183 false hazard ticks from startup, and stopped the ego after
only 3.5m. The safe cutoff remains unchanged. Test23 also exposed low-speed mode chatter/creep
after the initial successful stop. A pre-PR fix now prevents an occupied committed path from
being treated as clear merely because the speed-dependent LiDAR trigger shrinks as the ego
slows. Below 1m/s, a still-occupied path enters `STOP_HOLD`; the validated run reached 0.0mph,
held about 9.8m center-to-center from the parked car, and recorded zero post-hazard
`CRUISE`/`RECOVER` ticks.

Recovery reporting now distinguishes the controller finishing its requested-offset schedule
from the ego physically returning to the route. A generic tracker observes the measured
route-relative lateral offset and only records a physical return after the ego has first
departed and then remains within 0.25m of route center for ten consecutive simulation ticks
while the requested offset is also centered. This matters for blocked cases such as test22:
the debug/requested path can merge back and the recovery command can finish while the stationary
parked car keeps the real ego stopped about one lane away. Such a run must report command
completion separately from `physically_returned_to_route=False`, together with its final
measured lateral offset. Live validation confirmed that distinction: blocked-path test22
completed its recovery request but remained stopped at a measured +4.88m offset and correctly
reported no physical return, while clear-path test18A completed the maneuver, returned to
+0.00m, and reported a physical return. Test18B retained the safety backstop, stopped without
contact, and correctly reported no physical return.

After the handoff change, the full nine-case test20 matrix again recorded zero oriented contacts.
It covered both directions, 25/35mph, two TTC settings, stationary/far-cross pedestrians, an
undersized far-cross path (140 swept hazard ticks), and a vehicle-blocked path that correctly
full-stopped (133 swept hazard ticks). The valid cases continued and recovered as before.

This is **not yet a general dynamic-obstacle solution**. When a scripted pedestrian is present,
the experimental handoff still uses its ground-truth position as positive evidence that the new
path clears the old obstacle. Raw LiDAR controls hazards after handoff but does not yet maintain
persistent, actor-independent tracks. Validation is also still concentrated on one map/route.
The next architectural step is temporal LiDAR clustering/tracking with uncertainty through
missing frames, followed by investigation of test19's delayed recovery and broader validation
of occupied-path hold/release behavior with moving vehicles.

## Week 4 decision: CARLA's built-in Chrono physics is not the research backend (added 2026-10-01)

**Keep CARLA's default physics as the primary research backend. Chrono is a NO-GO for the
main experiment, but a LIMITED-GO for optional robustness testing.**

**What was done** (branch `experiment/chrono-physics-feasibility`; this was a bounded feasibility check):
- Confirmed that packaged CARLA 0.9.16 ships Project Chrono 6.0.0 and runs it when launched with
  `CarlaUE4.exe --chrono`. Drove a Tesla Model 3 by hand under both backends
  (`docs/MANUAL_CARLA_DRIVING.md`). Chrono felt more inertial, with longer acceleration
  windup; that impression was not measured.
- Built `src/test24___chrono_backend_smoke.py`, a backend-selectable smoke test, on a new
  pure module `src/physics_backend.py` (offline-tested):
  - Template and nested-reference validation with SHA-256 hashes, plus a check that the
    server was launched with `--chrono`. Both run before anything spawns.
  - The recorded backend label comes from evidence: a failed enable or a collision can never
    be recorded as `chrono`. CARLA 0.9.16 has no API call that reports which backend is active.
  - A predeclared control schedule and an initial-state gate before measurement.
  - Run directories are exclusive and never overwritten.

  `TraceTick` gained additive raw-getter columns (`get_acceleration`,
  `get_angular_velocity`, `vel_z`, sim frame, wall time per tick), and `PhysicsRunManifest`
  gained a top-level `physics_backend` field (`None` = every pre-Week-4 run, all default
  physics).
- Chrono never reached the measured sequence. After enabling, the braked, parked car
  crept at ~0.16 m/s, so the initial-state gate correctly aborted. The result was bit-identical
  on repeat. `src/test25___chrono_hold_diagnostic.py` then held a parked car for 10 s under
  8 controlled variations, run twice with bit-identical results.

**Measured (Town04_Opt; single configuration; descriptive):**

| Case (10 s hold) | Net movement |
|---|---|
| Default, brake 1.0 or **no brake**, 0.62° slope or flat | 0.00 m (never moves) |
| Chrono, brake 1.0, 0.62° slope | ~2.0 m downhill, turned ~+28°, still 0.2 m/s at the end |
| Chrono, no brake, 0.62° slope | ~3.4 m straight downhill, accelerating |
| Chrono, brake 1.0, flat (0.00°) | ~12 cm, slowing (0.009 m/s at the end), not exactly zero |
| Chrono, brake 1.0 + hand brake, slope | ~3.7 m, worse than no brake (see cause below) |
| Chrono enabled on the first tick (no settle), slope | ~1.9 m over ~11 s, so enable timing is not the cause |

Every Chrono enable produced a ~3–3.7° pitch jolt. Chrono cost ~7.5–10 ms of wall time per
0.02 s tick, against ~1 ms for default physics: roughly 10x slower.

**Diagnosed cause** (from CARLA 0.9.16 and Chrono 6.0.0 source; code and templates were not
changed):
- Chrono's `ChBrakeSimple` documents that it cannot model static sticking. Chrono has
  brake locking, but it is off by default, and CARLA's bridge never enables it.
- CARLA passes `brake + hand_brake` to Chrono without clamping, so brake 1.0 plus hand brake
  sends an out-of-range 2.0 instead of a real parking-brake command.
- CARLA hard-codes terrain friction to 1, so low map friction is not the cause.
- Part of the enable jolt is a bridge artifact: Chrono initializes the body 0.25 m up, and
  the mapping back to Unreal adds a fixed 2.5° pitch.

There is no confirmed JSON- or template-only fix. A real fix needs a source-level change to
CARLA's Chrono bridge and a rebuild, which is out of scope.

**Why this decision:**
- Chrono's deterministic low-speed creep can contaminate exactly the outcomes this project
  measures: stopping, collision, and safe stop.
- Default physics has known artifacts of its own. It is unrealistically sticky at rest (an
  unbraked car does not roll down a 0.6° grade), and it has the Week 3 low-speed braking snap.
  But it is stable and reproducible (the regression run after the refactor matched the
  earlier trace exactly), which is what controller comparisons need.
- Default physics keeps continuity with last semester's and Week 3's experiments.

**Evidence status:** **CONFIRMED** for the hold behavior: measured directly, bit-identical
across repeated runs, and explained from source. The planned matched straight-braking and
constant-turn Chrono comparisons were **not run**: once Chrono failed the at-rest gate, the
decision no longer depended on them.

**Rules for any future Chrono use (LIMITED-GO):**
- Report it only as a separate sensitivity/robustness study, using matched default/Chrono
  reruns through the same script.
- Never compare Chrono numbers directly with historical default-physics numbers.
- Prefer verified flat terrain, and keep measured phases away from near-zero-speed behavior.
- Do not use CARLA's hand brake under Chrono.

CarSim is the only other vehicle-dynamics integration worth a brief look, and only if
Connecticut College already has a license and support. No simulator migration is planned.

## Week 4 baseline: scripted braking+steering beats braking alone in a measurable window (added 2026-10-02)

**What was done** (branch `experiment/steer-brake-baseline`, stacked on the Chrono branch, CARLA
default physics):
- `src/test26___steer_brake_baseline.py` runs three matched controllers on one
  stationary-pedestrian scenario (Town04_Opt, spawn-242 route):
  - **no_intervention:** hazard braking disabled.
  - **brake_only:** full brake from onset.
  - **brake_steer:** the same full brake plus test18's scripted swerve, 1.98 m to the
    route-right over 0.5 s.
- All three start reacting at the same scripted moment, the "oracle onset" (opt-in
  `run_scenario(hazard_command_fn=...)`). That moment is when the ego is a set
  time-to-collision (TTC) from the pedestrian, using ground truth rather than LiDAR
  detection timing.
- Every run is scored by the new `src/encounter_metrics.py` protocol over an identical 8 s
  window after onset. It uses oriented-footprint contact and clearance in every direction,
  stopping metrics measured from the shared onset, an explicit outcome reason, and per-tick
  footprint drivability.

**Measured** (sweep `20261002_004906`; deterministic, as repeat runs reproduced exactly):

| Onset TTC (s) | 35 mph brake_only | 35 mph brake_steer | 45 mph brake_only | 45 mph brake_steer |
|---|---|---|---|---|
| 0.6 | hit 12.3 m/s | hit 12.1 m/s | hit 15.9 m/s | hit 15.7 m/s |
| 0.8 | hit 10.2 m/s | **passed, 0.12 m** | hit 13.6 m/s | **passed, 0.30 m** |
| 1.0 | hit 8.0 m/s | **stopped, 0.32 m** | hit 10.9 m/s | **passed, 0.45 m** |
| 1.2 | stopped, 0.23 m | stopped, 0.79 m | hit 8.1 m/s | **stopped, 0.54 m** |
| 1.4–1.6 | stopped clear | stopped clear | stopped clear | stopped clear |

no_intervention hit the pedestrian at full speed in every case.

**Findings:**
- **Scripted braking+steering avoids the pedestrian where braking alone hits it: onset TTC
  0.8–1.0 s at 35 mph, widening to 0.8–1.2 s at 45 mph.** The maneuver is physically
  achievable and measurable in CARLA, which was the precondition for adding steering to the RL
  action space.
- Margins inside the window are small (0.12–0.54 m). Both strategies fail at 0.6 s.
- At full brake, brake_steer stops beside or before the pedestrian and never returns to the
  route, so route recovery is not exercised by this baseline.
- **CARLA's collision sensor reported no collision in any of the 22 contacts** (21 in the
  sweep plus 1 in the smoke run, out of 39 runs; including a 20 m/s straight-through hit).
  An earlier version of this section said 26, a miscount corrected on 2026-10-04 from the
  per-run `metrics.json`/`result.json` files. The legacy `RunResult` labelled these `slowed_avoided` or
  `full_stop`. RunResult outcomes for stationary or far-cross pedestrians therefore undercount
  collisions. Use `encounter_metrics` for any controller comparison.
- Caveat, recorded and deliberately not pursued: **steering costs no braking in CARLA's
  default vehicle model.** brake_steer decelerates exactly like brake_only while also turning,
  so combined acceleration can exceed ~1 g. Results are reported as CARLA produces them.
- Map quirk: Town04_Opt has ~2 cm gaps between some adjacent Driving lanes, where
  `get_waypoint(project_to_road=False)` returns nothing. A seam-tolerant check
  (`map_drivability.classify_point_drivability_seam_tolerant`) is used for footprint
  drivability. The swept-path corridor check still uses the strict version.

**Evidence status:** **CONFIRMED** for this scenario: one location, route-right swerve,
stationary pedestrian, full brake, deterministic reruns. It is not yet tested at other
locations, with left swerves, at partial brake levels, or with LiDAR detection in the loop.

---

## Phase 1 (Week 5): trustworthy RL environment and the v3 re-audit (added 2026-10-04)

**What was done** (branch `experiment/rl-env-trustworthy`, CARLA 0.9.16 default physics; the
server was confirmed launched without `--chrono`):
- `CarlaAEBEnv(collision_signal="legacy" | "geometric")`. `"legacy"` (default) keeps last
  semester's semantics: CARLA's sensor OR the near-cross proximity fallback. `"geometric"`
  adds oriented-footprint pedestrian contact in every direction (`src/rl_collision_signal.py`).
  Every component is reported in `info` each tick. The env also places a `"stationary"`
  pedestrian correctly; it used to silently treat it as `"far"`.
- Offline verification against all 39 Week 4 test26 runs (committed fixture): the
  geometric signal catches **all 22 recorded contacts on exactly the recorded first-contact
  tick, with zero false contacts in the 17 clear runs**. The legacy signal would have caught 0.
- **Live before/after check:** v3 through the default (legacy) env on 4 matched-grid
  configs, including two archived collisions; old code twice, new code once. **Identical on
  every tick** (pose, control, action, reward, termination, observation). The old code also
  reproduced itself exactly.
- `src/eval_policy_encounters.py`: scores a learned policy with the test26 protocol (same
  onset, same 8 s window, `encounter_metrics`), runs past the env's own termination, and
  records full provenance. `src/analyze_rl_reaudit.py` compares archived, reproduced-legacy
  and protocol labels.

**Re-audit of v3 on last semester's 200-configuration matched grid**
(`src/runs/rl_encounter_eval/20261004_192904_rl_encounter_eval_default`, git `0bcc0e6`
clean, model `sac_v3_1600k.zip` SHA-256 `ef04f2fd04aa...`):

| | Archived (`eval_sac_on_sweep.py`) | Protocol (`encounter_metrics`) |
|---|---|---|
| Collision / contact | 40 (20%) | **73 (36.5%)**; 68 (34%) at the 0.188 m walker radius |
| "Full stop" | 148 | 60 actual stops clear (all near-cross) |
| Passed the pedestrian clear | (no label) | 67 (all far-cross) |
| "Slowed, avoided" | 12 | 0 (all 12 were contacts) |
| Far-cross contacts | 2 | **33** |
| Near-cross contacts | 38 | 40 |
| Contacts in "avoidable" scenarios | 2 / 120 | **15 / 120** |

**Findings:**
- **The old evaluation pipeline reproduces exactly from current code: the reconstructed
  legacy label matches the archived label in all 200 episodes.** The difference is entirely
  in how outcomes are scored, not in the model or the simulation.
- **v3's corrected contact rate on the matched grid is 36.5% (73/200), not 20%.** The
  archived "full stop" count also falls from 148 to 60 actual stops. 67 were the car
  passing a far-cross pedestrian who had cleared the lane, and 21 were contacts.
- **Every one of the 73 contacts is a frontal hit at real speed (median 10.0 m/s, minimum
  3.7 m/s, none below 3 m/s).** CARLA's collision sensor missed 51 of them. The result is
  not driven by the radius choice: using CARLA's measured walker box (half-extent
  0.188 m) instead of the conservative 0.3 m removes only 5 contacts, all far-cross.
- **Why far-cross hits were hidden:** the scripted walker keeps moving along its line after
  being hit, reaches its endpoint, and the env ends the episode as "pedestrian crossed".
  The old evaluator then calls that a full stop (19 cases) or, after truncation,
  "slowed_avoided" (12 cases).
- **Why some near-cross hits were hidden: the legacy proximity fallback (env and test3)
  compares a 3D center distance with a 2D-calibrated 2.7 m threshold.** The walker's origin
  is 1.09 m above the vehicle's (measured live), so the fallback fires only once the
  pedestrian's center is about 7 cm inside the front bumper. In the evaluator smoke run it
  never fired, even though the 2D distance, lateral offset and speed all met its
  thresholds.
- v3 contacts cluster by scenario rather than by crossing type. Every TTC 1.8 s cell and
  the 35-45 mph TTC 2.2 s cells contain contacts, with nearly the same pattern for near and
  far crossings. By speed: 8/40 at 22 and 28 mph, 16/40 at 35 and 40 mph, 25/40 at 45 mph.
- After its first stop, v3 can release the brake and creep forward into a pedestrian
  standing in the lane (smoke run: second touch at 2.2 m/s). The old evaluator ended the
  episode at the first stop and never observed this.

**What this does NOT yet establish:** a SAC-versus-fixed-profile comparison. The archived
fixed-profile numbers (19-23%) came from `RunResult`, which shares the sensor and
3D-fallback problems and saved no traces. **They must be re-run with trace recording and
re-scored under the same protocol before any comparison.** The avoidability labels also use
nominal target speed (known problem #5).

**Evidence status:** **CONFIRMED** for v3 on this grid: deterministic policy, one pass,
labels reproduced 200/200, trace-level checks of contact geometry. It is a single model and
a single seed.

---

## Week 5: SAC v3 versus the fixed profiles under one protocol (added 2026-10-05)

**What was done** (branch `experiment/fixed-profile-rescore`, CARLA 0.9.16 default physics,
server renderless with `-RenderOffScreen`, no `--chrono`):
- `src/test27___fixed_profile_rescore.py` re-ran last semester's final matched sweep
  (`src/runs/20260417_202355`: 4 fixed profiles x 200 configs). Each run used its archived
  `config.json` unchanged and `run_scenario` exactly as `sweep.py` did. The only change is
  a longer post-crossing settle (9 s instead of 3 s) so the protocol's 8 s window is
  covered. Every run was recorded as a trace and scored with `encounter_metrics`, the
  same protocol as test26 and the v3 re-audit.
- `src/analyze_controller_comparison.py` matched the 200 scenarios across the four
  profiles and SAC v3 (from the Phase 1 re-audit) and compared contacts per scenario.
- Evidence: `src/runs/fixed_profile_rescore/20261004_203248_fixed_profile_rescore_default`
  (git `10ec7a0`, crash-recovery resume at `ec50109`; the scenario code path is identical).

**Reproduction:** **800/800 runs match the archive exactly on everything fixed at onset**
(trigger time, speed and distance at trigger, hazard engagement). Time-to-stop differs in
24 runs. 17 are because the longer tail lets the car stop after the archive had ended. In
the other 7, the car stops after hitting the pedestrian, and CARLA's car-pedestrian
contact physics is not bit-repeatable. Neither affects the contact outcome. CARLA crashed
once at run 619 (server process gone); the run resumed from there with no lost results.

**Results on the 200-scenario matched grid** (contact = oriented footprint within the
0.3 m pedestrian circle; the 0.188 m column uses CARLA's measured walker box):

| Controller | Archived collisions | **Protocol contacts** | at 0.188 m | Near / far | 22 / 28 / 35 / 40 / 45 mph | "Avoidable" scenarios |
|---|---|---|---|---|---|---|
| step_constant | 38 (19%) | **60 (30.0%)** | 58 | 36 / 24 | 0 / 8 / 12 / 16 / 24 | 10 / 120 |
| proportional_ramp | 40 (20%) | **62 (31.0%)** | 58 | 36 / 26 | 0 / 8 / 14 / 16 / 24 | 12 / 120 |
| exponential | 42 (21%) | **67 (33.5%)** | 60 | 38 / 29 | 2 / 8 / 16 / 16 / 25 | 15 / 120 |
| SAC v3 | 40 (20%) | **73 (36.5%)** | 68 | 40 / 33 | 8 / 8 / 16 / 16 / 25 | 15 / 120 |
| cautious_ramp | 46 (23%) | **76 (38.0%)** | 76 | 44 / 32 | 8 / 8 / 16 / 18 / 26 | 16 / 120 |

Paired, same scenarios (exact McNemar on one deterministic pass per controller):

| SAC v3 vs. | SAC hit, profile clear | Profile hit, SAC clear | Both hit | p |
|---|---|---|---|---|
| step_constant | 13 | 0 | 60 | 0.0002 |
| proportional_ramp | 11 | 0 | 62 | 0.001 |
| exponential | 7 | 1 | 66 | 0.07 |
| cautious_ramp | 1 | 4 | 72 | 0.38 |

**Findings:**
- **The collision undercount affected every controller, not just SAC: fixed-profile
  contact rates are 30-38%, not the archived 19-23%.**
- **Last semester's headline ("SAC's collision rate was in the same range as the fixed
  rules") does not survive.** Under one protocol, SAC v3 ranks second-worst of five.
  **It hits the pedestrian in 13 scenarios where step_constant does not, and never the
  reverse; against proportional_ramp it is 11 to 0.** Against exponential the difference
  is not clear (p = 0.07), and SAC is not distinguishable from cautious_ramp.
- Most contacts are shared: 60 scenarios are hit by every controller, mostly TTC 1.8 s
  and the higher speeds. These are close to physically unavoidable for braking alone,
  which is exactly where Phase 2's steering is meant to help.
- SAC's extra contacts are concentrated at low speed: 8 contacts at 22 mph vs. 0 for
  step_constant and proportional_ramp.
- SAC "passes clear" far more often (67, vs. 12-37 for the profiles): it tends to drive
  on after a far-cross pedestrian has cleared rather than stopping. Its median
  normal-speed jerk is the lowest (640 vs. 776-1088 m/s^3), but comfort was not the
  question here and has its own known measurement caveats.

**Evidence status:** **CONFIRMED** as a descriptive comparison on this grid: deterministic
scripted scenarios, onset-fixed values reproduced 800/800, one deterministic pass per
controller, single SAC model and seed. The McNemar p-values describe these 200 scenarios,
not a population of policies or seeds. The avoidability labels still use nominal target
speed (known problem #5).

---

## Week 5 (Phase 2, step 1): the passage checked live, and the best aim position per scenario (added 2026-10-10)

**What was done** (branch `experiment/steering-rl-env`, CARLA default physics, renderless):
- `test26` gained `--passage-u`: the `brake_passage_edge` mode (full brake plus steering
  aimed at passage coordinate `u`, held from onset) runs once per aim position. It also
  gained `progress.log`, `--resume`, `--relaunch-carla` and `--table`.
- Two sweeps on the stationary-pedestrian scenario, 35 and 45 mph, 126 runs in total:
  `20261010_104704` (onset TTC 0.6-1.6 s) and `20261010_112344` (0.3-0.5 s, added to find
  where no option works). Each scenario ran no_intervention, brake_only, brake_steer
  (last week's fixed 1.98 m swerve) and `u` = 0.25, 0.5, 0.75, 1.0.
- `u` = 1.0 aims 7.42 m to the right (two lanes over); 0.25 aims 1.85 m.

**Measured.** "passed" and "stopped" give the smallest gap between the car's body and
the pedestrian (0.3 m radius); "hit" gives the impact speed. no_intervention hit at full
speed in every scenario.

35 mph:

| Onset TTC (s) | brake_only | brake_steer | u=0.25 | u=0.5 | u=0.75 | u=1.0 | Best |
|---|---|---|---|---|---|---|---|
| 0.3 | hit 14.9 | hit 14.9 | hit 14.6 | hit 14.6 | hit 14.6 | hit 14.6 | all hit; least bad u=0.5, 14.6 m/s |
| 0.4 | hit 14.2 | hit 14.1 | hit 13.9 | hit 13.5 | hit 13.5 | hit 13.5 | all hit; least bad u=0.75, 13.5 m/s |
| 0.5 | hit 13.3 | hit 13.1 | hit 12.9 | passed 0.44 m | passed 0.49 m | passed 0.48 m | u=0.75 |
| 0.6 | hit 12.3 | hit 12.1 | hit 10.6 | passed 0.98 m | passed 1.34 m | passed 1.35 m | u=1.0 |
| 0.8 | hit 10.2 | passed 0.12 m | passed 0.20 m | stopped 1.64 m | stopped 2.83 m | stopped 3.61 m | u=1.0 |
| 1.0 | hit 8.0 | stopped 0.32 m | stopped 0.31 m | stopped 1.98 m | stopped 3.60 m | stopped 4.96 m | u=1.0 |
| 1.2 | stopped 0.23 m | stopped 0.79 m | stopped 1.29 m | stopped 3.44 m | stopped 5.03 m | stopped 6.28 m | u=1.0 |
| 1.4 | stopped 3.37 m | stopped 3.87 m | stopped 4.29 m | stopped 6.13 m | stopped 7.94 m | stopped 9.50 m | u=1.0 |
| 1.6 | stopped 6.48 m | stopped 6.93 m | stopped 7.31 m | stopped 9.39 m | stopped 10.99 m | stopped 12.50 m | u=1.0 |

45 mph:

| Onset TTC (s) | brake_only | brake_steer | u=0.25 | u=0.5 | u=0.75 | u=1.0 | Best |
|---|---|---|---|---|---|---|---|
| 0.3 | hit 19.0 | hit 18.9 | hit 18.6 | hit 18.4 | hit 18.4 | hit 18.4 | all hit; least bad u=0.75, 18.4 m/s |
| 0.4 | hit 18.1 | hit 18.0 | hit 17.7 | hit 17.2 | hit 17.1 | hit 17.1 | all hit; least bad u=0.75, 17.1 m/s |
| 0.5 | hit 17.0 | hit 16.8 | hit 16.1 | passed 0.45 m | passed 0.44 m | passed 0.32 m | u=0.5 |
| 0.6 | hit 15.9 | hit 15.7 | passed 0.16 m | passed 1.25 m | passed 1.64 m | passed 1.68 m | u=1.0 |
| 0.8 | hit 13.6 | passed 0.30 m | passed 0.36 m | passed 1.97 m | passed 3.26 m | passed 4.19 m | u=1.0 |
| 1.0 | hit 10.9 | passed 0.45 m | passed 0.45 m | stopped 2.16 m | stopped 3.77 m | stopped 5.29 m | u=1.0 |
| 1.2 | hit 8.1 | stopped 0.54 m | stopped 0.47 m | stopped 2.40 m | stopped 4.78 m | stopped 6.73 m | u=1.0 |
| 1.4 | stopped 0.95 m | stopped 1.84 m | stopped 2.41 m | stopped 5.05 m | stopped 7.62 m | stopped 9.65 m | u=1.0 |
| 1.6 | stopped 4.93 m | stopped 5.71 m | stopped 6.18 m | stopped 8.65 m | stopped 11.10 m | stopped 13.21 m | u=1.0 |

"Best" is the safe option with the most clearance or, where every option hits, the one
with the lowest impact speed. Full tables: `best_route_table.md` in each sweep folder, or
`test26 --table <sweep>`.

**Findings:**
- **The passage is right on the live server.** Its edges were -3.87 m and +7.42 m in all
  72 aimed runs, identical to the offline check, with no tick missing a passage.
- **The results are repeatable.** All 36 runs shared with last week's sweep reproduced it
  exactly.
- **Aiming into the passage avoids the pedestrian with as little as 0.5 s of warning.**
  Braking alone needs 1.2 s at 35 mph and 1.4 s at 45 mph; last week's fixed swerve needs
  0.8 s.
- **At 0.4 s or less there is no safe route: every option hits, at both speeds.** The
  larger swerves hit slowest, but only 0.3-1.0 m/s slower than braking alone.
- **A bigger swerve almost always gives more clearance, and `u` = 1.0 is usable.** It
  never left the road or spun the car in 18 runs, so the action range needs no cap. The
  car does not reach the 7.42 m aim under full braking: it gets 4.6-5.1 m over at 35 mph
  and 6.2-6.5 m at 45 mph, and stops angled 37-50 degrees to the road.
- **At the limit, more is not better.** At 0.5 s the best aim is `u` = 0.75 (35 mph) and
  0.5 (45 mph); at 45 mph `u` = 1.0 leaves 0.32 m against 0.45 m, because the sharper
  turn swings the car's side closer.
- The smallest aim that is safe is `u` = 0.5 at TTC 0.5 s and at 35 mph / 0.6 s, and
  `u` = 0.25 in every other scenario from 0.6 s up.
- Pedestrian radius: 58 contacts at 0.3 m, 57 at 0.188 m. The one difference is 35 mph /
  0.6 s / `u` = 0.25, which overlaps the 0.3 m circle by 3 cm and misses CARLA's walker
  box by 8 cm.
- Steering commitment: 0 side switches and 0 direction changes in every aimed run, as
  expected for a held aim.
- CARLA's collision sensor reported 4 of the 58 contacts.

**Caveat (known, not pursued):** steering costs no braking in CARLA's default vehicle
model. Lateral acceleration (speed x yaw rate) reached 19 m/s^2 at 35 mph and 26 m/s^2
at 45 mph in the aimed runs, about 2-2.6 g, which real tires cannot give. These
clearances are CARLA's, not real-world predictions.

**What this means for the RL environment (open, for Quentin and Prof. Izmirli):**
- In this scene a bigger swerve costs nothing, so "safest by clearance" is nearly always
  `u` = 1.0. If the policy should prefer the smallest swerve that clears, the reward has
  to say so.
- The informative warning times are about 0.3-1.2 s. Above that every option succeeds.

**Evidence status:** **CONFIRMED** for this scenario: one location, stationary pedestrian,
right swerves, full brake, held aim, one deterministic run per case.

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

**Status (2026-10-04): fixed for RL evaluation** by `eval_policy_encounters.py` (explicit `passed_clear`/`stopped_clear`/`contact` outcomes). Re-audit: 67 of v3's 148 archived "full stops" were passes and 21 were contacts.

**Severity:** Critical to reported results.

`carla_aeb_env.py` terminates an episode when the far pedestrian reaches its endpoint. `eval_sac.py` and `eval_sac_on_sweep.py` infer full stop from noncollision termination. This converts pedestrian clearance into a vehicle stop.

**Impact:** 33/148 final matched “full stops” and 16/64 generic random-eval “full stops” lack actual stop evidence.

**Fix direction:** Return/store explicit `termination_reason` and define `full_stop` only from the environment's speed-threshold flag.

**Historical sibling bug already recognized in the code:** The comment immediately above `ped_crossed` documents an earlier near-cross failure: reaching the lane-center endpoint used to terminate the episode before a collision could register, producing an `impossible + full_stop` misclassification. The current code fixed that case by allowing `ped_crossed` termination only for far crossings. The remaining defect is the same underlying semantic mistake one layer later: evaluators assume every noncollision `terminated` episode means the ego stopped. Far-cross clearance may reasonably remain a terminal event, but it must be reported as `pedestrian_cleared`, not `actual_stop`. If eventual-stop rate is a comparison metric, all controllers must additionally receive the same post-clearance observation horizon.

### 2. SAC and fixed-profile episode horizons differ

**Status (2026-10-04): fixed for RL evaluation** (same onset + 8 s window as test26, ignoring env termination). Fixed profiles still need a re-run under the same protocol.

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

**Status (2026-10-04): CONFIRMED and larger than suspected.** v3: 33 far-cross contacts vs. 2 archived. Near-cross too: the proximity fallback uses 3D distance against a 2D threshold (the walker origin is 1.09 m above the vehicle's), so it misses slow frontal contact. Fixed for RL by `collision_signal="geometric"` and the encounter evaluator; `RunResult`/test3 are unchanged.

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

## Design artifact: tightening lateral constraints (Week 3 Workstream 1.3, added 2026-09-27)

**Status: RESOLVED by Prof. Izmirli's reply (email, 2026-10-03). See
[Resolution](#resolution-advisor-answer-2026-10-03) at the end of this section.** The text
below is the original Week 3 design pass, kept as written.

**Original status (2026-09-27): deliberately NOT implemented.** Per the Week 3 plan's own gate for this item
("if the intended behavior remains ambiguous, keep this as a design artifact and do not
guess in control code"), this section is the required design pass, not a spec for code
written this week. No `lane_follow.py` changes accompany this entry.

### The advisor's idea, as given (2026-09-15)

> "think about how constraints are in play for steering; should it be unconstrained at
> first but the lateral steer constraints close in as the pedestrian/hazard gets in view
> so as to not run off the road or maybe so it can not over swerve and run off the road?"

The Week 3 plan already flagged the core ambiguity directly: **"clarify whether
constraints should tighten because time is running out, widen because urgency is
increasing, or constrain rate while preserving an already committed maneuver."** Working
through those three readings against this project's own already-measured evidence (not
speculation) shows they are not just differently worded -- they prescribe different, and
in one case directly opposed, controller behavior.

### Three candidate formalizations

All three share the same inputs the plan specifies -- speed, hazard distance/TTC, current
offset, remaining drivable space (`map_drivability.py`, merged Week 2) and remaining
unoccupied space (`vehicle_occupancy.py`, merged this week, PR #9) -- and differ only in
how the allowed lateral target range responds to shrinking TTC.

**A. Tighten as TTC shrinks** (allowed |offset| range shrinks as the hazard nears).
Rationale: less time before impact means less time to execute and recover from a large
swerve, and a late, aggressive swerve has a measured real cost -- test10 (this week, PR
#6) found full-lock steering at speed can bring the vehicle to a near-stop within 1-2
seconds purely from cornering drag, with body slip angle growing to 20-40 degrees in
sustained turns. Reading Izmirli's own words literally ("constraints... close in... so
as to not... over swerve"), this is the most direct interpretation.
**Problem:** it can remove exactly the capability needed exactly when it is needed most.
This project's own steering-plus-braking matrix (Week 2, PR #3) already found that
`steering_only` never fully avoided a pedestrian without eventually invoking a last-resort
stop -- a modest, bounded swerve was already not enough authority in several tested
configurations. Shrinking the bound further as danger increases would make that worse,
not better, in exactly the cases where a swerve is the last real option before impact.

**B. Widen as TTC shrinks** (allowed |offset| range grows as the hazard nears).
Rationale: as the situation becomes more certain and more dangerous, the controller
should be given MORE lateral authority, not less, since a modest fixed offset already
measured poorly: the Week 2 offset-magnitude sweep (`test8`, 2026-09-18) found that
increasing the requested peak offset from 1.0m to 2.5m (2.5x larger) improved real minimum
pedestrian clearance only from 3.50m to 3.90m, because total 3D clearance is dominated by
the roughly offset-independent longitudinal braking gap, with lateral offset contributing
only under a square root. Widening the bound near the hazard could let the controller draw
on more of that limited lever exactly when the longitudinal gap is smallest.
**Problem:** this is the reading Izmirli's own sentence argues against most directly
("close in", not "open up"), and a widening bound with no other constraint is exactly the
shape of rule that produces the outcome he explicitly said he wants to avoid ("run off the
road" / "over swerve") if it is not simultaneously capped by remaining drivable and
unoccupied space.

**C. Constrain the RATE of change, not the offset magnitude, especially once a lateral
target is already committed.**
Rationale: this reframes "constraint" as being about smoothness/commitment rather than a
shrinking or growing envelope on the final target. This project has already hit the
concrete failure mode this reading targets, twice, independently, this semester:
- The 2026-09-18 recovery-oscillation bug: a flickering "hazard clear" signal near a
  stationary pedestrian let the recovery controller re-arm and reverse a committed
  maneuver mid-execution (`request` bounced `+2.50 -> +0.36 -> +2.50m`), fixed by requiring
  a sustained-clear window rather than a single-tick read.
- The 2026-09-18 braking-governance-chatter regression (the hardened retry above): rapid
  tick-to-tick switching of which corridor governed braking caused real instability
  (`recovered` dropped 4/12 -> 1/12, jerk nearly doubled) despite fixing the danger it was
  built to address.
Both were fixed by adding *hysteresis to a decision*, not by changing the decision's
target value -- exactly what interpretation C proposes as a general principle: once
committed to a lateral target, changing that commitment should become harder as time runs
out, independent of whether the target's own numeric magnitude should grow or shrink.

### Cross-reference: this may be the same question as Workstream 4.1

The advisor-backlog memory already records a near-identical open question from the
Sep 22 meeting, filed under the *separate* per-tick-vs-macro-actions topic (Workstream
4.1): "Ask Izmirli what failure mode he had in mind when he questioned per-tick decisions:
unstable action switching, inability to commit early enough, poor credit assignment, or
physical irreversibility once a maneuver starts." "Unstable action switching" and
"inability to commit early enough" are, in substance, the same concern as interpretation C
above. It is plausible Izmirli's tightening-constraint idea (1.3) and his per-tick-action
skepticism (4.1) are the same underlying worry -- that a frame-by-frame controller can
change its mind too easily -- expressed twice in two different framings a week apart,
rather than two independent design questions. Worth surfacing to him as one combined
question rather than two separate ones.

### Recommendation

**Do not implement any of A, B, or C in control code this week.** They are not just
differently worded; A and B prescribe opposite bound directions, and C targets a different
mechanism (rate/commitment) entirely. Guessing which one Izmirli meant and hand-coding it
risks the same pattern already seen twice on this branch this semester: a plausible-looking
fix that passes its own targeted check but introduces a different regression the broader
matrix catches later (see the braking-authority open question above -- the parallel is
direct).

**Specific question to bring back to Izmirli** (combining this with the 4.1 question above
into one conversation): *"When you said the lateral constraint should close in as the
hazard comes into view, did you mean (a) the maximum allowed swerve distance should
shrink as time-to-collision drops, (b) it should grow, or (c) once the car has committed to
a swerve direction, it shouldn't be allowed to change its mind or reverse that commitment
as time runs out -- and is this the same concern as when you questioned whether per-tick
decisions could handle a committed maneuver?"*

**If/when disambiguated:** whichever interpretation Izmirli confirms should still go
through the same validation discipline used for every other control change this project
has made this semester -- explicit monotonic/boundary test cases written offline first,
one focused live positive and one focused live negative case, then the full validation
matrix, inspecting traces rather than trusting aggregate outcome labels alone. This design
artifact intentionally stops short of writing that specification, since committing to
concrete boundary values for an ambiguous rule would itself be guessing.

### Resolution: advisor answer (2026-10-03)

Quentin asked Prof. Izmirli which of (a), (b), (c) he meant, and whether he preferred
per-tick steering or a one-time swerve decision (left/right/none) plus braking. His answers:

- **(a) Shrink the max swerve as TTC drops: "probably not." Rejected; do not build it.**
- **(b) Grow emergency steering authority: "hopefully going to be an emergent property of
  learned steering." Do not hard-code it.**
- **(c) No reversing a committed direction: "would be good, since indecision will reduce
  the efficacy of the maneuver,"** but he "wouldn't be surprised if the per-tick actions
  automatically lead to this behavior." So measure it first, not constrain it.
- **Per-tick vs. one-time swerve: continuous, per-tick steering** ("at least that is the
  more interesting case").
- His underlying idea is a **"passage"**: the car continuously observes the allowable range
  across the road (e.g. from the left road edge to a pedestrian walking left) and adjusts
  its steering to it. This generalizes to a second obstacle, such as another car after the
  pedestrian. Aiming at the passage midpoint is too timid: it won't steer hard enough and
  doesn't anticipate where the pedestrian will be when the car reaches its walking line.
  He suggests aiming toward the extreme edge on the swerve side (leftmost for a left
  swerve, rightmost for a right swerve, center when going straight).

**What this means for Phase 2 (steering in RL; design not yet approved):**
- The action space gets continuous per-tick steering plus braking. Whether the action is a
  raw steering command or a lateral aim point (through `lane_follow_step`'s route-relative
  offset) is an open design choice. The aim-point path inherits the lateral controller's
  smoothing: in test18 it reached 1.65 m of a 1.98 m request. So measure each option's
  authority before choosing.
- The observation includes the passage. Compute it for the ego *center* (shrink by the
  half-width plus a margin), so "aim at the extreme edge" never puts part of the car off the
  road or into the obstacle. Build it from `map_drivability` and `vehicle_occupancy` so other
  obstacles can narrow it later. For crossing pedestrians, the edge should use the
  pedestrian's predicted position when the ego reaches the crossing line (not needed for the
  stationary-pedestrian baseline).
- No rule that shrinks the swerve bound. Do not cap the action range tighter than the
  passage itself.
- Evaluation adds a steering-commitment metric (e.g. sign reversals of the requested
  steering or lateral offset after onset). If learned policies waver, a commitment
  constraint (c) is an advisor-endorsed fallback, not a guess.
- His "passage-edge aim" rule is also a candidate scripted baseline to compare a learned
  policy against.

## Phase 2 design decisions (Quentin, 2026-10-04/05)

These turn the advisor's answer above into concrete choices for steering in the RL
environment. They were decided before any Phase 2 code was written. Full option analysis:
`plans/Phase-2_2026-10-04_steering-rl-design-options.md` (local).

### 1. The steering action is a position across the "passage" (option C)

**Plain-language idea.** At the pedestrian's position there is an open gap the car can
drive through: the passage. Picture a ruler laid across the road over that gap:

```
 left road edge                         pedestrian                        right road edge
 |------ safe for the car's center ------|  (blocked)  |------ safe for the car's center ------|
 u = -1                                u = 0 (the car's normal path, lane center)          u = +1
```

- Each tick the policy outputs **one number `u` between -1 and +1**: a spot on that ruler.
- `u = 0` means "keep the normal lane-center path" (no swerve).
- `u = +1` means "aim at the rightmost spot where the car's *center* can be and still keep
  its whole body on the road and clear of the obstacle." `u = -1` is the same on the left.
- Values in between are proportional; for example, `u = +0.5` is halfway from lane center
  to the right edge.
- The controller converts `u` to a sideways distance in meters and feeds it to the
  existing lane-following steering (`lane_follow_step(lateral_offset_m=...)`). That steering
  already steers smoothly toward a point ahead that is offset from the route.

**Why this design:**
- It is Prof. Izmirli's idea made learnable: aiming at the extreme edge (`u = +1` or `-1`) is
  exactly his suggested rule. The policy can learn when to use the extreme and when
  something gentler is better.
- The edges are the safe limits by construction, so every action keeps the car on the
  road and clear of the obstacle, provided the passage is computed correctly. A raw
  steering-wheel action can easily put the car off the road.
- It generalizes: a second obstacle (e.g. another car after the pedestrian) just moves an
  edge of the ruler inward, and `u = +1` still means "as far right as is safe."
- It reuses steering code that is already validated (test5/test18/test26).

**Things to remember:**
- If the passage is computed wrong, the action is wrong. The passage module gets its own
  offline tests first, including the Town04 lane-seam gaps.
- The edges are for the car's *center*: the gap shrunk by the car's half-width (1.08 m)
  plus a margin.
- One sub-choice will be measured in the scripted baseline before training: does the
  steering aim at a point 6 m ahead (the current lookahead) or at the pedestrian's line?

### 2. The hazard starts with an "oracle onset" (for training and evaluation)

**What it means.** A real car must first *notice* the pedestrian (LiDAR detection) and then
*react*. With an oracle onset, the simulator itself announces "hazard now" to the
controller at a scripted moment: when the car is a chosen time-to-collision (TTC, e.g.
0.6-1.6 s) from the pedestrian, computed from true positions. From that tick on, the
controller acts.

**Why:**
- **Fair comparison.** Every controller (the three test26 scripted baselines and the
  learned policy) gets exactly the same warning time. Any difference in outcome is then due
  to the maneuver itself, not to when each one happened to detect the pedestrian.
- **It isolates the research question.** If detection and maneuvering are tested together,
  a failure could be a late detection or a bad maneuver, and you can't tell which.
- **It's needed for a stationary pedestrian.** LiDAR sees a pedestrian standing in the lane
  from far away, so braking alone succeeds trivially. That is why test26 used oracle
  onset.
- Warning time becomes an experimental knob: sweeping TTC maps exactly where steering
  starts to beat braking.

**Limitation (deliberate, recorded):** it assumes perfect, instant perception. LiDAR in the
loop comes back in Phase 4 (robustness), where the same policy is tested with real
detection timing.

### 3. Observations and reward: add the proposed terms

Approved. Observation additions: passage left/right edge, distance to the pedestrian's
line, route-lateral offset, heading error, lateral velocity, previous steering action.
Reward: geometric collision (Phase 1 signal; the env default flips to `"geometric"` in
Phase 2), an off-road penalty from footprint drivability, and the existing comfort terms.
**No steering-reversal penalty at first**: reversals are measured by a new evaluation
metric, as the advisor suggested.

### 4. Swerve right only at first; open both sides if results are promising

The first policy may only swerve right (`u` limited to [0, +1]). If it learns well, the
range opens to [-1, +1] so the policy chooses a side.

**Assessment: a good, simple start.**
- It matches test26, whose brake_steer baseline swerves right, so the first comparison is
  like-for-like.
- It halves what the policy must learn.
- This location has more room on the right (Week 3).
- The cost is that it can't yet show *choosing* a side. That is the planned second step.
- The code will support [-1, +1] from the start with a configurable limit, so opening up
  later is a setting change, not a rewrite.

### 5. Pedestrian contact radius: 0.3 m primary, 0.188 m reported alongside

**What "contact" means in the protocol.** The pedestrian is modeled as a circle of radius
0.3 m around its center. "Contact" means the car's rectangular footprint overlaps that
circle. CARLA's own walker collision box is smaller (half-width 0.188 m, measured
2026-10-04). So under the 0.3 m rule, a car whose body passes within about 11 cm of
CARLA's walker box *without touching it* is still counted as contact.

**Decision: keep 0.3 m as the primary radius, and always report the 0.188 m count next
to it.** Reasons:
- **It's safety-conservative.** A pass within ~11 cm of a person is a failure in any
  safety sense. CARLA's box is a simplified body without arms, swing or bags.
- **It can only over-count, never miss.** That matters given this project's main lesson:
  every earlier check undercounted.
- **It's consistent.** Every Week 4/5 number (test26, the v3 re-audit) used 0.3 m, so
  results stay comparable.
- **It barely changes conclusions so far.** v3: 73 contacts at 0.3 m vs. 68 at 0.188 m. No
  test26 outcome changes.
- Reports should say "contact (including passes within ~0.1 m)" where the distinction
  matters.

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
