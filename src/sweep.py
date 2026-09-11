"""
sweep.py

Runs the test3 pedestrian intrusion scenario many times, varying
parameters across runs and saving results to disk.

"Sweep" means: loop over a range of parameter values, run the scenario
once per combination, and collect the outcomes into a dataset. This is
how you go from "one interesting run" to "50 runs covering different
conditions" without manually editing and re-running the script.

Output structure (created automatically):
    runs/
      <sweep_timestamp>/
        run_0000/
          config.json      <- the ScenarioConfig used for that run
          result.json      <- the RunResult (outcome metrics)
        run_0001/
          config.json
          result.json
        ...
        sweep_summary.csv  <- one row per run, all params + outcomes

Usage:
    python sweep.py

Edit SWEEP_CONFIGS at the bottom of this file to define your parameter
combinations. Two helper functions are provided:

    make_grid_sweep(...)   -> all combinations of the given lists
    make_random_sweep(...) -> N randomly sampled configs from ranges

For a first test, the default at the bottom runs a small 12-run grid.
"""

import csv
import os
import sys
import traceback
from datetime import datetime
from itertools import product
import random as _random
from typing import List, Optional

# CARLA path setup must happen before importing scenario modules.
CARLA_ROOT = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16"
_carla_pythonapi_root  = os.path.join(CARLA_ROOT, "PythonAPI")
_carla_pythonapi_carla = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if _carla_pythonapi_root not in sys.path:
    sys.path.append(_carla_pythonapi_root)
if _carla_pythonapi_carla not in sys.path:
    sys.path.append(_carla_pythonapi_carla)

from scenario_config import ScenarioConfig
from run_result import RunResult
from test3___ped_intrusion_scenario import run_scenario


# =====================================================================
# Sweep builders
# =====================================================================

def make_grid_sweep(
    target_mph:           List[float]          = [25.0, 35.0, 45.0],
    encounter_distance_m: List[float]          = [60.0, 120.0],
    walker_side:          List[str]            = ["left"], #["left", "right"],
    walker_cross:         List[str]            = ["near"],
    walker_speed_mps:     List[float]          = [1.8],
    weather_preset:       List[str]            = ["ClearSunset"],
    trigger_ttc_s:        List[Optional[float]]= [2.8],
    brake_profile:        List[str]            = ["proportional_ramp"],
    brake_headway_s:      List[float]          = [2.5],
    **fixed_kwargs,
) -> List[ScenarioConfig]:
    """
    Return all combinations of the provided parameter lists (cartesian product).

    Any parameter not listed here stays at its ScenarioConfig default.
    Use **fixed_kwargs to pin additional parameters to a single value,
    e.g. make_grid_sweep(sim_seconds=25.0, trigger_ttc_s=[2.5])

    trigger_ttc_s: seconds-before-crossing when the pedestrian trigger fires.
      [None]       -> use legacy formula (walker_startup_s + cross_time)
      [2.8]        -> pin all runs to 2.8s TTC
      [2.0, 2.8, 3.5] -> sweep over three tightness levels

    brake_profile: which braking strategy to evaluate.
      ["proportional_ramp"]                            -> baseline only
      ["proportional_ramp", "step_constant",
       "cautious_ramp", "exponential"]                 -> all four profiles

    brake_headway_s: seconds of speed-scaled lookahead added to base detection distance.
      trigger_distance_m = 5.0 + speed_mps * brake_headway_s
      [2.5]        -> baseline detection range (e.g. ~44m at 35mph)
      [2.5, 5.0]   -> sweep baseline vs early-detection (e.g. ~44m vs ~83m at 35mph)

    Example:
        3 speeds x 2 encounter distances x 4 profiles x 2 headways = 48 configs
    """
    combos = list(product(
        target_mph,
        encounter_distance_m,
        walker_side,
        walker_cross,
        walker_speed_mps,
        weather_preset,
        trigger_ttc_s,
        brake_profile,
        brake_headway_s,
    ))
    configs = []
    for (mph, enc, side, cross, wspeed, weather, ttc, profile, headway) in combos:
        cfg = ScenarioConfig(
            target_mph=mph,
            encounter_distance_m=enc,
            walker_side=side,
            walker_cross=cross,
            walker_speed_mps=wspeed,
            weather_preset=weather,
            trigger_ttc_s=ttc,
            brake_profile=profile,
            brake_headway_s=headway,
            **fixed_kwargs,
        )
        configs.append(cfg)
    return configs


def make_random_sweep(
    n: int,
    seed: int = 0,
    target_mph_range:           tuple = (20.0, 50.0),
    encounter_distance_m_range: tuple = (60.0, 150.0),
    walker_speed_mps_range:     tuple = (1.0, 2.5),
    trigger_ttc_s_range:        tuple = (2.0, 4.0),
    walker_sides:               List[str] = ["left", "right"],
    walker_crosses:             List[str] = ["near", "far"],
    weather_presets:            List[str] = ["ClearNoon", "CloudyNoon",
                                             "HardRainNoon", "WetNoon", "ClearSunset"],
    **fixed_kwargs,
) -> List[ScenarioConfig]:
    """
    Return N randomly sampled configs drawn from the specified ranges.

    The seed makes the sweep reproducible: same seed -> same N configs every time.
    Each config records the seed it was generated from (config.random_seed).

    Example:
        make_random_sweep(n=50, seed=42)
        -> 50 configs with randomly varied parameters, reproducible with seed=42
    """
    rng = _random.Random(seed)
    configs = []
    for i in range(n):
        cfg = ScenarioConfig(
            target_mph=round(rng.uniform(*target_mph_range), 1),
            encounter_distance_m=round(rng.uniform(*encounter_distance_m_range), 1),
            walker_speed_mps=round(rng.uniform(*walker_speed_mps_range), 2),
            trigger_ttc_s=round(rng.uniform(*trigger_ttc_s_range), 2),
            walker_side=rng.choice(walker_sides),
            walker_cross=rng.choice(walker_crosses),
            weather_preset=rng.choice(weather_presets),
            random_seed=seed,
            **fixed_kwargs,
        )
        configs.append(cfg)
    return configs


# =====================================================================
# Sweep runner
# =====================================================================

def run_sweep(
    configs: List[ScenarioConfig],
    output_dir: Optional[str] = None,
) -> List[RunResult]:
    """
    Run one scenario per config, save config.json + result.json per run,
    and write a sweep_summary.csv at the end.

    Args:
        configs:    List of ScenarioConfig objects to run (in order).
        output_dir: Directory to write results into. Defaults to
                    runs/<timestamp> in the current working directory.

    Returns:
        List of RunResult objects (one per config, None if a run crashed).
    """
    if output_dir is None:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("runs", timestamp)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[sweep] Output directory: {os.path.abspath(output_dir)}")
    print(f"[sweep] {len(configs)} run(s) queued\n")

    results: List[Optional[RunResult]] = []

    for i, cfg in enumerate(configs):
        run_id = f"run_{i:04d}"
        cfg.run_id = run_id
        run_dir = os.path.join(output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Save config before the run so it exists even if the run crashes.
        cfg.to_json(os.path.join(run_dir, "config.json"))

        print(f"\n[sweep] Starting {run_id} ({i+1}/{len(configs)})  "
              f"target={cfg.target_mph}mph  encounter={cfg.encounter_distance_m}m  "
              f"walker={cfg.walker_side}/{cfg.walker_cross}  weather={cfg.weather_preset}")

        try:
            result = run_scenario(cfg, plot_after=False)
            result.to_json(os.path.join(run_dir, "result.json"))
            results.append(result)
            print(f"[sweep] {run_id} complete | "
                  f"hazard={result.hazard_triggered} | "
                  f"outcome={result.outcome} | "
                  f"min_ped_dist={result.min_ped_distance_m:.2f}m | "
                  f"ego_at_trigger={result.ego_speed_at_trigger_mph:.1f}mph")
        except Exception:
            print(f"[sweep] ERROR in {run_id} -- skipping, continuing sweep.")
            traceback.print_exc()
            results.append(None)

    # Write summary CSV
    _write_summary_csv(output_dir, configs, results)
    _print_sweep_summary(results, configs=configs)

    return [r for r in results if r is not None]


def _write_summary_csv(output_dir: str, configs: List[ScenarioConfig], results: list):
    """Write one CSV row per run combining config params and outcome metrics."""
    csv_path = os.path.join(output_dir, "sweep_summary.csv")

    # Column order: run identity, then config params, then outcome metrics.
    fieldnames = [
        "run_id",
        # Config
        "target_mph", "encounter_distance_m",
        "walker_speed_mps", "walker_side", "walker_cross",
        "walker_startup_s", "trigger_ttc_s", "trigger_delay_s",
        "braking_ramp_up_per_s", "brake_headway_s", "brake_profile",
        "weather_preset", "sun_altitude_deg", "cloudiness",
        "sim_seconds", "random_seed",
        # Outcome
        "hazard_triggered", "collision_detected", "outcome",
        "min_ped_distance_m", "min_ttc_s",
        "max_jerk_mps3", "mean_jerk_mps3",
        "ego_speed_at_trigger_mph", "ego_speed_at_trigger_mps",
        "ego_dist_at_trigger_m", "trigger_time_s", "time_to_stop_s",
        "error",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for cfg, result in zip(configs, results):
            row = {
                "run_id": cfg.run_id,
                "target_mph": cfg.target_mph,
                "encounter_distance_m": cfg.encounter_distance_m,
                "walker_speed_mps": cfg.walker_speed_mps,
                "walker_side": cfg.walker_side,
                "walker_cross": cfg.walker_cross,
                "walker_startup_s": cfg.walker_startup_s,
                "trigger_ttc_s": f"{cfg.trigger_ttc_s:.2f}" if cfg.trigger_ttc_s is not None else "",
                "trigger_delay_s": cfg.trigger_delay_s,
                "braking_ramp_up_per_s": cfg.braking_ramp_up_per_s,
                "brake_headway_s": cfg.brake_headway_s,
                "brake_profile": cfg.brake_profile,
                "weather_preset": cfg.weather_preset,
                "sun_altitude_deg": cfg.sun_altitude_deg,
                "cloudiness": cfg.cloudiness,
                "sim_seconds": cfg.sim_seconds,
                "random_seed": cfg.random_seed,
            }
            if result is not None:
                row.update({
                    "hazard_triggered": result.hazard_triggered,
                    "collision_detected": result.collision_detected,
                    "outcome": result.outcome,
                    "min_ped_distance_m": f"{result.min_ped_distance_m:.3f}",
                    "min_ttc_s": f"{result.min_ttc_s:.3f}" if result.min_ttc_s >= 0 else "",
                    "max_jerk_mps3": f"{result.max_jerk_mps3:.2f}",
                    "mean_jerk_mps3": f"{result.mean_jerk_mps3:.2f}",
                    "ego_speed_at_trigger_mph": f"{result.ego_speed_at_trigger_mph:.2f}",
                    "ego_speed_at_trigger_mps": f"{result.ego_speed_at_trigger_mps:.3f}",
                    "ego_dist_at_trigger_m": f"{result.ego_dist_at_trigger_m:.1f}",
                    "trigger_time_s": f"{result.trigger_time_s:.2f}" if result.trigger_time_s is not None else "",
                    "time_to_stop_s": f"{result.time_to_stop_s:.2f}" if result.time_to_stop_s is not None else "",
                    "error": "",
                })
            else:
                row["error"] = "CRASHED"
            writer.writerow(row)

    print(f"\n[sweep] Summary CSV -> {csv_path}")


def _print_sweep_summary(results: list, configs: Optional[List] = None):
    """Print a brief aggregate summary after all runs complete.

    If configs is provided and contains brake_headway_s values, the summary
    is split into one section per unique headway so the effect is immediately
    visible in the terminal without opening the CSV.
    """
    total   = len(results)
    crashed = sum(1 for r in results if r is None)
    ran     = total - crashed
    if ran == 0:
        print("[sweep] All runs crashed.")
        return

    def _section(label, pairs):
        valid  = [r for _, r in pairs if r is not None]
        if not valid:
            return
        n      = len(valid)
        haz    = sum(1 for r in valid if r.hazard_triggered)
        stops  = sum(1 for r in valid if r.outcome == "full_stop")
        slowed = sum(1 for r in valid if r.outcome == "slowed_avoided")
        colls  = sum(1 for r in valid if r.outcome == "collision")
        dists  = [r.min_ped_distance_m for r in valid if r.min_ped_distance_m >= 0]
        print(f"\n{'='*55}")
        print(label)
        print(f"  Hazard brake triggered : {haz}/{n}")
        print(f"  Full stop before ped   : {stops}/{n}")
        print(f"  Slowed + avoided       : {slowed}/{n}")
        print(f"  Collisions             : {colls}/{n}  ({100*colls/n:.0f}%)")
        if dists:
            print(f"  Min ped dist (best)    : {min(dists):.2f} m")
            print(f"  Min ped dist (mean)    : {sum(dists)/len(dists):.2f} m")
        print(f"{'='*55}")

    all_pairs = list(zip(configs or [None]*len(results), results))

    # Overall
    _section(f"SWEEP COMPLETE  ({ran}/{total} runs succeeded, {crashed} crashed)",
             all_pairs)

    # Split by headway if configs are available and have the attribute
    if configs:
        headways = sorted(
            hw for hw in set(
                getattr(cfg, "brake_headway_s", None) for cfg in configs
            )
            if hw is not None
        )
        if len(headways) > 1:
            print("\n--- Breakdown by detection range (brake_headway_s) ---")
            for hw in headways:
                pairs = [
                    (cfg, r) for cfg, r in all_pairs
                    if getattr(cfg, "brake_headway_s", None) == hw
                ]
                _section(f"  headway={hw}s  "
                         f"(trigger ~{5.0 + 15.27*hw:.0f}m at 35mph, "
                         f"~{5.0 + 19.76*hw:.0f}m at 45mph)",
                         pairs)
    print()


# =====================================================================
# Entry point -- edit SWEEP_CONFIGS to define your sweep
# =====================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Option A: Grid sweep
    # Cartesian product of all listed values.
    # This example: 3 speeds x 2 encounter distances x 2 sides = 12 runs
    # ------------------------------------------------------------------
    SWEEP_CONFIGS = make_grid_sweep(
        target_mph=[22.0, 28.0, 35.0, 40.0, 45.0],
        encounter_distance_m=[60.0, 120.0],
        walker_side=["left"],
        walker_cross=["near", "far"],
        walker_speed_mps=[1.8],
        weather_preset=["ClearSunset"],
        trigger_ttc_s=[1.8, 2.2, 2.6, 3.0, 3.5],
        brake_profile=[
            "proportional_ramp",
            "step_constant",
            "cautious_ramp",
            "exponential",
        ],
        # Detection range sweep: baseline (2.5s) vs early-detection (5.0s).
        # trigger_distance_m = 5.0 + speed_mps * brake_headway_s
        #   At 35mph: 43.8m  vs  82.5m
        #   At 45mph: 54.5m  vs 104.0m
        brake_headway_s=[2.5, 5.0],
        sim_seconds=30.0,  # hard ceiling only — runs end dynamically when ped finishes crossing + 3s settle
        # Total: 5 speeds x 2 distances x 2 walker_cross x 5 ttc x 4 profiles x 2 headways = 800 runs
        # Unique scenario configs (for SAC eval): 5 x 2 x 2 x 5 x 2 = 200
    )

    # ------------------------------------------------------------------
    # Option B: Random sweep (comment out Option A and uncomment this)
    # Randomly samples N configs. Same seed = same configs every time.
    # ------------------------------------------------------------------
    # SWEEP_CONFIGS = make_random_sweep(
    #     n=20,
    #     seed=42,
    #     target_mph_range=(20.0, 50.0),
    #     encounter_distance_m_range=(60.0, 150.0),
    #     walker_speed_mps_range=(1.0, 2.2),
    #     weather_presets=["ClearNoon", "HardRainNoon", "WetNoon", "ClearSunset"],
    #     sim_seconds=20.0,
    # )

    run_sweep(SWEEP_CONFIGS)
