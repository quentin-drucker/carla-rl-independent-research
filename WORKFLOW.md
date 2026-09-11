# COM496 CARLA Project — Workflow & File Reference

Quentin Druck | Junior Spring Research 2026  
CARLA 0.9.16 | Python 3.12 | stable-baselines3 SAC

---

## Quick Reference — Commands by Goal

| What you want to do | Command |
|---|---|
| Start CARLA (headless, fastest) | See section below |
| Train a new SAC model | `python train_sac.py` |
| Continue training existing model | `python continue_sac.py` |
| Evaluate a SAC model (100 random episodes) | `python eval_sac.py sac_aeb_rand_1600k` |
| Plot SAC eval results | `python plot_eval.py` |
| Run profile sweep (800 runs) | `python sweep.py` |
| Resume a crashed sweep | `python resume_sweep.py runs/<timestamp>` |
| Run SAC on same configs as sweep | `python eval_sac_on_sweep.py sac_aeb_rand_1600k` |
| Unified SAC vs profiles comparison | `python plot_unified_comparison.py` |
| Comfort/safety/radar comparison | `python plot_comfort_safety.py` |
| Speed×TTC 5-panel comparison | `python plot_speed_ttc_comparison.py` |
| Stopping distance reference plot | `python plot_stopping_distance.py` |
| Inspect SAC brake curves live | `python inspect_brake_curves.py` |

---

## 1. Running CARLA

Always start CARLA before running any script that connects to the simulator.

**Headless (recommended for training/sweeps — much faster, no rendering):**
```
"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe" -RenderOffScreen
```

**Windowed (use this if you want to watch what's happening):**
```
"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe"
```

Wait ~15 seconds for CARLA to fully load before running any Python script. CARLA listens on `localhost:2000` by default.

**Scripts that require CARLA running:**
`train_sac.py`, `continue_sac.py`, `eval_sac.py`, `eval_sac_on_sweep.py`,
`sweep.py`, `resume_sweep.py`, `inspect_brake_curves.py`, `brake_test.py`,
`brake_calibration.py`, `test3___ped_intrusion_scenario.py`

**Scripts that do NOT require CARLA (offline, just read CSVs):**
`plot_eval.py`, `plot_unified_comparison.py`, `plot_comfort_safety.py`,
`plot_speed_ttc_comparison.py`, `plot_stopping_distance.py`

---

## 2. SAC Reinforcement Learning

### Training a new model from scratch
```
python train_sac.py
```
- Trains for 800,000 steps by default
- Saves to `sac_aeb_rand_800k.zip`
- Checkpoints saved every 20k steps to `checkpoints/`
- Randomizes episode configs (speed 22–50 mph, TTC 1.5–4.0s) each episode
- **Requires CARLA running**

**Key file:** `train_sac.py`  
**Key file:** `carla_aeb_env.py` — the Gymnasium environment the SAC agent trains in  
**Key file:** `scenario_config.py` — dataclass defining all scenario parameters  

### Continuing training from a checkpoint
```
python continue_sac.py                              # 800k → 1600k (default)
python continue_sac.py sac_aeb_rand_800k 800000     # explicit: load, add 800k steps
python continue_sac.py sac_aeb_rand_1600k 400000    # load 1600k, train 400k more
```
- Loads `.zip` model file, trains additional steps, saves to a new `.zip`
- **Never overwrites the source model** — always writes a new file
- Step counter preserved (`reset_num_timesteps=False`)
- Output name is auto-generated: e.g. loading `sac_aeb_rand_1600k` + 400k → `sac_aeb_rand_2000k.zip`
- Safe to Ctrl+C — saves interrupted checkpoint automatically

**Key file:** `continue_sac.py`

### Evaluating a SAC model
```
python eval_sac.py                          # evaluates sac_aeb_rand_800k (default)
python eval_sac.py sac_aeb_rand_1600k       # evaluate specific model
```
- Runs 100 fixed episodes (seed=42, same scenarios every time — comparable across models)
- Saves results to `eval_results.csv` in `src/`
- Prints per-episode table + collision breakdown by physics label
- **Requires CARLA running**

Output columns in `eval_results.csv`:
`episode, reward, collision, full_stop, speed_mph, ttc_s, walker_cross,`
`physics_label, margin_m, max_jerk_mps3, mean_jerk_mps3, min_ped_dist_m,`
`min_ttc_obs_s, time_to_stop_s`

**Key file:** `eval_sac.py`

### Plotting SAC eval results
```
python plot_eval.py                   # reads eval_results.csv
python plot_eval.py my_results.csv    # or specify a file
```
Produces 4 figures (saved to `src/`):
- `speed_ttc_scatter.png` — every episode on speed×TTC space; shape=near/far cross, color=physics label, marker type=outcome. The most information-dense plot.
- `outcome_by_label.png` — stacked bar: collision vs safe per physics label
- `reward_by_label.png` — two-panel reward distribution (safe episodes left, collision right)
- `success_by_speed.png` — success rate on physically avoidable scenarios by speed band

**Key file:** `plot_eval.py`

### Physics labels (avoidability classifier)
Every episode is labeled based on whether stopping was physically possible:
- **avoidable** — margin ≥ 8m. Collision is entirely the agent's fault.
- **borderline-avoidable** — margin 0–8m. Technically possible but very tight.
- **borderline-impossible** — margin −8–0m. Technically impossible but close.
- **impossible** — margin < −8m. No policy could stop in time.

Margin = `v₀ × TTC − v₀² / (2 × 3.5)` where 3.5 m/s² is the ramp-limited effective deceleration.

**Key file:** `avoidability.py`

---

## 3. Hand-Designed Braking Profile Sweep

### Running the sweep
```
python sweep.py
```
- Runs a grid of scenario configs × 4 braking profiles
- **Current grid:** 5 speeds × 5 TTC values × 2 walker modes × 2 encounter distances × 2 detection ranges × 4 profiles = **800 runs**
- Speeds: 22, 28, 35, 40, 45 mph
- TTC values: 1.8, 2.2, 2.6, 3.0, 3.5 s
- Walker cross: near (stops in center lane) or far (crosses full road)
- Detection range (brake_headway_s): 2.5s baseline vs 5.0s early-detection
- Output saved to `runs/<timestamp>/` with one subfolder per run + `sweep_summary.csv`
- **Requires CARLA running** (headless recommended — takes several hours)

The 4 braking profiles evaluated:
- **proportional_ramp** — brake proportional to how close the ped is (smooth)
- **step_constant** — full brake immediately on trigger (aggressive)
- **cautious_ramp** — ramp up slowly, max 50% brake (gentle, misses tight scenarios)
- **exponential** — brake increases quadratically with proximity (soft then hard)

**Key file:** `sweep.py` — edit `SWEEP_CONFIGS` at the bottom to change the grid  
**Key file:** `test3___ped_intrusion_scenario.py` — the actual scenario run per config  
**Key file:** `run_result.py` — dataclass for per-run outcome metrics  

### Resuming a crashed sweep
If CARLA crashes or the computer shuts off mid-sweep:
```
python resume_sweep.py                          # auto-finds latest sweep folder
python resume_sweep.py runs/20260417_202355     # specify folder explicitly
```
- Reads `sweep_summary.csv`, finds all rows with `error=CRASHED`
- Re-runs only those configs, patches results back into the existing CSV
- Safe to run multiple times if it crashes again

**Key file:** `resume_sweep.py`

---

## 4. Head-to-Head Comparison (SAC vs Profiles)

This is the main comparison pipeline. Run these in order after both the sweep and SAC eval are done:

### Step 1 — Run SAC on the exact same configs as the sweep
```
python eval_sac_on_sweep.py sac_aeb_rand_1600k
python eval_sac_on_sweep.py sac_aeb_rand_1600k runs/20260417_202355/sweep_summary.csv
```
- Loads the unique scenario configs from `sweep_summary.csv` (200 configs from proportional_ramp rows)
- Runs SAC deterministically on each one
- Saves `sac_on_sweep_results.csv` in the same folder as the sweep
- **Requires CARLA running**

**Key file:** `eval_sac_on_sweep.py`

### Step 2 — Generate unified comparison figures
```
python plot_unified_comparison.py
python plot_unified_comparison.py runs/20260417_202355/sweep_summary.csv
```
Produces 2 figures saved into the sweep folder:
- `unified_outcomes.png` — LEFT: stacked outcome bars (full stop / slowed / collision) per controller. RIGHT: collision rate by physics label for each controller side by side.
- `unified_fullstop_by_speed.png` — full-stop rate per speed group (22/28/35/40/45 mph) per controller.

Auto-detects the SAC model label from the CSV so no edits needed when switching models.

**Key file:** `plot_unified_comparison.py`

### Step 3 — Comfort/safety/risk comparison
```
python plot_comfort_safety.py
python plot_comfort_safety.py runs/20260417_202355/sweep_summary.csv
```
Produces 2 figures saved into the sweep folder:
- `comfort_safety_distributions.png` — violin+strip plots for mean jerk, min pedestrian distance, and time-to-stop across all 5 controllers.
- `comfort_safety_radar.png` — radar/spider chart normalizing 5 metrics per controller: overall safety, avoidable-zone performance, pedestrian margin, braking comfort, response speed. Larger polygon = better overall.

Also prints a summary table to the terminal.

**Key file:** `plot_comfort_safety.py`

---

## 5. Additional Informational Plots

### Speed×TTC 5-panel comparison (offline)
```
python plot_speed_ttc_comparison.py
```
- Reads `sweep_summary.csv` and `sac_on_sweep_results.csv`
- Produces `speed_ttc_5panel.png`: one panel per controller showing outcomes on the speed×TTC grid with physics boundary overlaid
- No CARLA needed

**Key file:** `plot_speed_ttc_comparison.py`

### Stopping distance reference (offline)
```
python plot_stopping_distance.py
```
- Reads `brake_calibration_results.csv`
- Shows raw vehicle stopping distance vs effective system stopping distance (ramp limiter overhead)
- Explains why ramp_up_per_s=4.0 roughly doubles stopping distance vs raw braking
- No CARLA needed

**Key file:** `plot_stopping_distance.py`

### Inspect SAC brake curves live
```
python inspect_brake_curves.py          # 5 episodes default
python inspect_brake_curves.py 10       # 10 episodes
```
- Loads the SAC model, runs N live episodes, logs per-tick brake commands
- Plots SAC's actual brake output vs what each hand-designed profile would have done at the same state
- Reveals behaviors like phantom pre-braking and oscillation
- Outputs `brake_curves_Nepisodes.png` and `brake_curves_data.csv`
- **Requires CARLA running**

**Key file:** `inspect_brake_curves.py`

---

## 6. Output File Map

### Model files (in `src/`)
| File | Description |
|---|---|
| `sac_aeb_rand_800k.zip` | SAC trained 800k steps |
| `sac_aeb_rand_1600k.zip` | SAC continued to 1.6M steps |
| `checkpoints/` | Intermediate checkpoints saved every 20k steps during training |

### CSV data files (in `src/`)
| File | Produced by | Contains |
|---|---|---|
| `eval_results.csv` | `eval_sac.py` | 100-episode SAC eval on random scenarios |
| `brake_calibration_results.csv` | `brake_calibration.py` | Raw vehicle deceleration measurements |
| `brake_curves_data.csv` | `inspect_brake_curves.py` | Per-tick brake trace data |

### Sweep output (in `src/runs/<timestamp>/`)
| File | Produced by | Contains |
|---|---|---|
| `sweep_summary.csv` | `sweep.py` | One row per run: all config params + outcome metrics for all 4 profiles |
| `sac_on_sweep_results.csv` | `eval_sac_on_sweep.py` | SAC results on the same 200 unique configs |
| `run_XXXX/config.json` | `sweep.py` | Per-run scenario config |
| `run_XXXX/result.json` | `sweep.py` | Per-run outcome metrics |

### Plot files
| File | Produced by | Location |
|---|---|---|
| `speed_ttc_scatter.png` | `plot_eval.py` | `src/` |
| `outcome_by_label.png` | `plot_eval.py` | `src/` |
| `reward_by_label.png` | `plot_eval.py` | `src/` |
| `success_by_speed.png` | `plot_eval.py` | `src/` |
| `unified_outcomes.png` | `plot_unified_comparison.py` | `src/runs/<timestamp>/` |
| `unified_fullstop_by_speed.png` | `plot_unified_comparison.py` | `src/runs/<timestamp>/` |
| `comfort_safety_distributions.png` | `plot_comfort_safety.py` | `src/runs/<timestamp>/` |
| `comfort_safety_radar.png` | `plot_comfort_safety.py` | `src/runs/<timestamp>/` |
| `speed_ttc_5panel.png` | `plot_speed_ttc_comparison.py` | `src/` |
| `stopping_distance_reference.png` | `plot_stopping_distance.py` | `src/` |
| `brake_curves_Nepisodes.png` | `inspect_brake_curves.py` | `src/` |

---

## 7. Infrastructure Files (don't need to edit these)

| File | What it does |
|---|---|
| `carla_aeb_env.py` | Gymnasium environment wrapper — manages the CARLA world, vehicle, walker, sensors, reward, and observation for RL training |
| `test3___ped_intrusion_scenario.py` | Single-run pedestrian intrusion scenario used by the profile sweep |
| `avoidability.py` | Computes physics-based avoidability label and stopping margin for a scenario config |
| `scenario_config.py` | Dataclass for all scenario parameters (speed, TTC, walker mode, etc.) |
| `run_result.py` | Dataclass for per-run outcome metrics |
| `lane_follow.py` | Lane-following and braking controller used inside the scenario |
| `carla_session.py` | CARLA connection + world loading helpers |
| `spawning.py` | Vehicle and walker spawn helpers |
| `spectator.py` | Camera follow controller for visualization |
| `math_utils.py` | Speed/distance/jerk utility functions |
| `walker_utils.py` | Pedestrian movement control |
| `loop_utils.py` | Simulation tick loop helpers |
| `train_sac.py` | SAC training setup + `sample_config()` function (defines training scenario distribution) |

---

## 8. Typical End-to-End Workflow

```
# 1. Start CARLA headless
"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe" -RenderOffScreen

# 2. Train SAC (overnight)
python train_sac.py                         # → sac_aeb_rand_800k.zip

# 3. Continue training if desired (overnight)
python continue_sac.py                      # → sac_aeb_rand_1600k.zip

# 4. Evaluate the model (20 min)
python eval_sac.py sac_aeb_rand_1600k       # → eval_results.csv

# 5. Plot SAC-only eval (offline, instant)
python plot_eval.py

# 6. Run profile sweep (several hours)
python sweep.py                             # → runs/<timestamp>/sweep_summary.csv
# If it crashes:
python resume_sweep.py runs/<timestamp>

# 7. Run SAC on matched configs (30 min)
python eval_sac_on_sweep.py sac_aeb_rand_1600k

# 8. Generate all comparison plots (offline, instant)
python plot_unified_comparison.py
python plot_comfort_safety.py
python plot_speed_ttc_comparison.py
python plot_stopping_distance.py
```
