"""
eval_sac.py

Evaluates a trained SAC model and saves per-episode results to a CSV for plotting.

Usage:
    python eval_sac.py                          <- evaluates sac_aeb_rand_800k
    python eval_sac.py sac_aeb_rand_800k        <- same, explicit
    python eval_sac.py checkpoints/sac_aeb_rand_50000_steps.zip

Output:
    - Per-episode table printed to console (with physics label and margin)
    - eval_results.csv saved to src/ for use with plot_eval.py

Physics labels:
    avoidable             — agent could have stopped; collision is agent's fault
    borderline-avoidable  — technically avoidable but within 8m uncertainty band
    borderline-impossible — technically impossible but within 8m uncertainty band
    impossible            — no policy could stop in time; not agent's fault

EVAL_SEED fixes the random scenario sequence so every run of this script
evaluates the exact same 100 scenarios in the same order. Change the seed
to generate a different fixed set (e.g. for a second validation batch).
"""

import sys
import csv
import random
import numpy as np
import carla as _carla

from stable_baselines3 import SAC
from carla_aeb_env import CarlaAEBEnv
from spectator import SpectatorController
from train_sac import sample_config
from avoidability import compute_avoidability

EVAL_SEED = 42   # fixed seed → same 100 scenarios every run
random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)

model_path = sys.argv[1] if len(sys.argv) > 1 else "sac_aeb_rand_800k"

print(f"\n{'='*55}")
print(f"  EVALUATING: {model_path}")
print(f"{'='*55}\n")

env   = CarlaAEBEnv(config_fn=sample_config)
model = SAC.load(model_path, env=env)

N_EPISODES = 100
rewards    = []
collisions = 0
stops      = 0
rows       = []   # collected for CSV

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    avoid  = compute_avoidability(env.cfg)
    done         = False
    total_r      = 0.0
    last_info    = {}

    # Hard-teleport spectator to vehicle before anything else
    assert env._world is not None and env._vehicle is not None
    _ego_tf  = env._vehicle.get_transform()
    _ego_loc = _ego_tf.location
    _fwd     = _ego_tf.get_forward_vector()
    env._world.get_spectator().set_transform(_carla.Transform(
        _carla.Location(
            x=_ego_loc.x - _fwd.x * 12.0,
            y=_ego_loc.y - _fwd.y * 12.0,
            z=_ego_loc.z + 5.0,
        ),
        _carla.Rotation(pitch=-15.0, yaw=_ego_tf.rotation.yaw, roll=0.0),
    ))
    env._world.tick()
    spec = SpectatorController(env._world, env._vehicle, debug_draw=False)

    # Per-episode metric accumulators
    # Skip first 10 ticks to avoid LP-filter init spike (not real jerk)
    JERK_WARMUP_TICKS = 10
    jerk_ticks        = []
    _tick_count       = 0
    min_ped_dist_m    = float("inf")
    min_ttc_s         = float("inf")
    hazard_start_s    = None
    stop_time_s       = None
    last_terminated   = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, last_info = env.step(action)
        total_r += r
        done = terminated or truncated
        last_terminated = terminated
        spec.tick(fixed_dt=0.02)

        # Accumulate per-tick metrics
        _tick_count += 1
        if _tick_count > JERK_WARMUP_TICKS:
            jerk_ticks.append(last_info.get("jerk_mps3", 0.0))

        tick_ttc = last_info.get("ttc_s", float("inf"))
        if tick_ttc < min_ttc_s:
            min_ttc_s = tick_ttc

        if env._walker is not None and env._vehicle is not None:
            try:
                d = env._vehicle.get_location().distance(env._walker.get_location())
                if d < min_ped_dist_m:
                    min_ped_dist_m = d
            except Exception:
                pass

        sim_t = last_info.get("sim_time_s", 0.0)
        mode  = last_info.get("drive_mode", "CRUISE")
        if hazard_start_s is None and mode in ("HAZARD_BRAKE", "STOP_HOLD"):
            hazard_start_s = sim_t
        if (hazard_start_s is not None and stop_time_s is None
                and env._vehicle is not None):
            try:
                from math_utils import get_speed_mps
                if get_speed_mps(env._vehicle) < 0.3:
                    stop_time_s = sim_t
            except Exception:
                pass

    spec.close()

    rewards.append(total_r)
    hit       = last_info.get("collision", False)
    full_stop = last_terminated and not hit
    label     = avoid["label"]
    margin    = avoid["margin_m"]
    margin_str = f"{margin:+.1f}m" if margin != float("inf") else "  inf"

    max_jerk   = max(jerk_ticks) if jerk_ticks else 0.0
    mean_jerk  = sum(jerk_ticks) / len(jerk_ticks) if jerk_ticks else 0.0
    time_to_stop = (round(stop_time_s - hazard_start_s, 2)
                    if (stop_time_s is not None and hazard_start_s is not None)
                    else "")

    if hit:
        collisions += 1
    if full_stop:
        stops += 1

    rows.append({
        "episode":        ep + 1,
        "reward":         round(total_r, 2),
        "collision":      int(hit),
        "full_stop":      int(full_stop),
        "speed_mph":      round(env.cfg.target_mph, 1),
        "ttc_s":          round(env.cfg.trigger_ttc_s, 2) if env.cfg.trigger_ttc_s else "",
        "walker_cross":   env.cfg.walker_cross,
        "physics_label":  label,
        "margin_m":       round(margin, 1) if margin != float("inf") else "inf",
        "max_jerk_mps3":  round(max_jerk, 2),
        "mean_jerk_mps3": round(mean_jerk, 2),
        "min_ped_dist_m": round(min_ped_dist_m, 2) if min_ped_dist_m != float("inf") else "",
        "min_ttc_obs_s":  round(min_ttc_s, 3) if min_ttc_s != float("inf") else "",
        "time_to_stop_s": time_to_stop,
    })

    print(f"  ep {ep+1:3d}: reward={total_r:7.2f}  "
          f"collision={'YES' if hit else 'no ':3}  "
          f"full_stop={'YES' if full_stop else 'no '}  "
          f"physics={label:<22}  margin={margin_str}  "
          f"[{env.cfg.walker_cross} {env.cfg.target_mph:.0f}mph ttc={env.cfg.trigger_ttc_s:.1f}s]")

env.close()

# ── Save CSV ────────────────────────────────────────────────────────────────
import os
model_stem = os.path.splitext(os.path.basename(model_path))[0]
csv_path = f"eval_results_{model_stem}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\n  Results saved -> {csv_path}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  SUMMARY ({N_EPISODES} episodes)")
print(f"  Mean reward  : {np.mean(rewards):.2f}")
print(f"  Collisions   : {collisions}/{N_EPISODES}")
print(f"  Full stops   : {stops}/{N_EPISODES}")

# Break collisions down by physics label
from collections import Counter
label_counts    = Counter(r["physics_label"] for r in rows)
col_by_label    = Counter(r["physics_label"] for r in rows if r["collision"])
print(f"\n  Collision breakdown:")
for lbl, total in sorted(label_counts.items()):
    ncol = col_by_label.get(lbl, 0)
    print(f"    {lbl:<26} {ncol} collisions / {total} episodes")
print(f"{'='*55}\n")
