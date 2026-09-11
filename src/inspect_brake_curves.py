"""
inspect_brake_curves.py
Quentin | Junior Spring Research, 2026.

Per-tick brake curve inspection: runs the SAC agent on a small set of
representative scenarios and visualizes what brake command it outputs
over time, compared to what each fixed profile would have done in the
same state.

For each episode this script records every tick:
  - SAC raw action (rl_brake_override before ramp limiter)
  - SAC applied brake_cmd (after ramp limiter)
  - Profile brake_targets analytically computed from the same tick state:
      proportional_ramp  = penetration
      step_constant      = 1.0 (whenever hazard active)
      cautious_ramp      = min(penetration, 0.5)
      exponential        = penetration²
    where penetration = clamp((trigger_dist - d_min) / (trigger_dist - PANIC_DIST), 0, 1)

Output:
  brake_curves_<N>episodes.png   — one row per episode, brake command over time
  brake_curves_data.csv          — raw per-tick table for all episodes

Usage:
    python inspect_brake_curves.py                          <- 5 random episodes (sac_v2_1600k)
    python inspect_brake_curves.py sac_v2_1600k 8          <- 8 episodes
    python inspect_brake_curves.py sac_v2_1600k 5 avoidable  <- filter label

Note: requires a 6-dim observation model (sac_v2_*). Old 5-dim models (sac_aeb_rand_*)
are incompatible with the current CarlaAEBEnv and cannot be loaded.

Prerequisite: CARLA must be running (CarlaUE4.exe).
"""

import os
import sys
import csv
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stable_baselines3 import SAC
from carla_aeb_env import CarlaAEBEnv
from train_sac import sample_config
from avoidability import compute_avoidability

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

MODEL_PATH   = sys.argv[1] if len(sys.argv) > 1 else "sac_v2_1600k"
N_EPISODES   = int(sys.argv[2]) if len(sys.argv) > 2 else 5
LABEL_FILTER = sys.argv[3] if len(sys.argv) > 3 else None  # e.g. "avoidable"

random.seed(99)
np.random.seed(99)

# ---------------------------------------------------------------------------
# Physics constants from carla_aeb_env.py (must match)
# ---------------------------------------------------------------------------

PANIC_DISTANCE_M  = 5.0
BRAKE_BASE_DIST_M = 5.0
BRAKE_HEADWAY_S   = 2.5    # ScenarioConfig default


def _profile_targets(d_min, trigger_dist, hazard_active):
    """
    Analytically compute what each fixed profile would output at this tick state.
    Returns dict: profile_name -> brake_target (float in [0,1]).
    Note: this is the DESIRED target before the ramp limiter applies.
    """
    if not hazard_active or d_min is None or d_min >= trigger_dist:
        return {p: 0.0 for p in ("proportional_ramp", "step_constant",
                                  "cautious_ramp", "exponential")}
    denom       = max(1e-3, trigger_dist - max(0.0, PANIC_DISTANCE_M))
    penetration = max(0.0, min(1.0, (trigger_dist - d_min) / denom))

    return {
        "proportional_ramp": penetration,
        "step_constant":     1.0,
        "cautious_ramp":     min(penetration, 0.5),
        "exponential":       penetration * penetration,
    }


# ---------------------------------------------------------------------------
# Run episodes and record per-tick traces
# ---------------------------------------------------------------------------

env   = CarlaAEBEnv(config_fn=sample_config)
model = SAC.load(MODEL_PATH, env=env)

all_traces  = []   # list of episode trace dicts
all_ticks   = []   # flat list of per-tick dicts for CSV

collected = 0
attempts  = 0

print(f"\nCollecting {N_EPISODES} episodes"
      + (f" (label filter: {LABEL_FILTER})" if LABEL_FILTER else "") + "...")

while collected < N_EPISODES:
    attempts += 1
    if attempts > N_EPISODES * 5:
        print(f"  [warn] gave up after {attempts} attempts; only got {collected} episodes")
        break

    obs, _ = env.reset()
    avoid  = compute_avoidability(env.cfg)
    if LABEL_FILTER and avoid["label"] != LABEL_FILTER:
        continue

    tick_data = []
    done      = False
    collision = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        sac_raw   = float(np.clip(np.asarray(action).flat[0], 0.0, 1.0))

        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        tel         = env._last_telemetry
        d_min       = tel.get("d_min_ahead_m")
        trig_dist   = tel.get("trigger_distance_m", BRAKE_BASE_DIST_M + 15.0 * BRAKE_HEADWAY_S)
        hazard_flag = bool(tel.get("hazard_brake_cmd", 0))
        brake_cmd   = tel.get("brake_cmd", 0.0)   # after ramp limiter

        profiles    = _profile_targets(d_min, trig_dist, hazard_flag)

        tick_data.append({
            "sim_time_s":        info.get("sim_time_s", 0.0),
            "ego_speed_mps":     info.get("ttc_s", 0.0),   # not speed — fixed below
            "sac_action":        sac_raw,
            "sac_brake_cmd":     brake_cmd,
            "hazard_active":     int(hazard_flag),
            "d_min_m":           d_min if d_min is not None else -1.0,
            "trigger_dist_m":    trig_dist,
            "ttc_s":             info.get("ttc_s", float("inf")),
            "proportional_ramp": profiles["proportional_ramp"],
            "step_constant":     profiles["step_constant"],
            "cautious_ramp":     profiles["cautious_ramp"],
            "exponential":       profiles["exponential"],
        })
        if info.get("collision", False):
            collision = True

    # Trim to only show from trigger fire to end (where the interesting behavior is)
    trigger_idx = next(
        (i for i, t in enumerate(tick_data) if t["hazard_active"]),
        0
    )
    trace_window = tick_data[max(0, trigger_idx - 10):]  # show 10 pre-trigger ticks

    all_traces.append({
        "episode":       collected + 1,
        "label":         avoid["label"],
        "margin_m":      avoid["margin_m"],
        "collision":     collision,
        "speed_mph":     env.cfg.target_mph,
        "ttc_s":         env.cfg.trigger_ttc_s,
        "walker_cross":  env.cfg.walker_cross,
        "ticks":         trace_window,
    })
    for tick in tick_data:
        all_ticks.append({"episode": collected + 1, **tick})

    outcome = "COLLISION" if collision else "safe"
    print(f"  ep {collected+1}: {env.cfg.target_mph:.0f}mph "
          f"ttc={env.cfg.trigger_ttc_s}  physics={avoid['label']:<24}  {outcome}")
    collected += 1

env.close()

# ---------------------------------------------------------------------------
# Save raw CSV
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_out    = os.path.join(script_dir, "brake_curves_data.csv")
if all_ticks:
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_ticks[0].keys())
        writer.writeheader()
        writer.writerows(all_ticks)
    print(f"\nSaved per-tick data -> {csv_out}")

# ---------------------------------------------------------------------------
# Plot brake curves
# ---------------------------------------------------------------------------

PROFILE_STYLES = {
    "proportional_ramp": dict(color="#4C72B0", lw=1.1, ls="--",  label="Proportional Ramp"),
    "step_constant":     dict(color="#DD8452", lw=1.1, ls=":",   label="Step Constant"),
    "cautious_ramp":     dict(color="#55A868", lw=1.1, ls="-.",  label="Cautious Ramp"),
    "exponential":       dict(color="#C44E52", lw=1.1, ls=(0,(3,1,1,1)), label="Exponential"),
}
SAC_COLOR = "#1a1a2e"

n_ep  = len(all_traces)
fig, axes = plt.subplots(n_ep, 1, figsize=(12, 3.2 * n_ep), squeeze=False)

for row_idx, trace in enumerate(all_traces):
    ax   = axes[row_idx][0]
    tks  = trace["ticks"]
    if not tks:
        ax.set_visible(False)
        continue

    t_arr = np.array([tk["sim_time_s"] for tk in tks])

    # Shade hazard-active region
    hazard_flags = [tk["hazard_active"] for tk in tks]
    for i, (h, t) in enumerate(zip(hazard_flags, t_arr)):
        if h:
            ax.axvspan(t, t + 0.02, color="lightyellow", zorder=0, alpha=0.6)

    # Profile brake targets (dashed/dotted — what each would have aimed for)
    for profile, style in PROFILE_STYLES.items():
        vals = np.array([tk[profile] for tk in tks])
        ax.plot(t_arr, vals, zorder=2, **style)

    # SAC action (raw target before ramp) — thick blue solid
    sac_vals = np.array([tk["sac_action"] for tk in tks])
    ax.plot(t_arr, sac_vals, color=SAC_COLOR, lw=2.2, ls="-",
            label="SAC (raw action)", zorder=4)

    # SAC applied command (after ramp) — thin translucent
    sac_cmd = np.array([tk["sac_brake_cmd"] for tk in tks])
    ax.plot(t_arr, sac_cmd, color=SAC_COLOR, lw=1.0, ls="-", alpha=0.4,
            label="SAC (after ramp limiter)", zorder=3)

    # Trigger line
    trigger_t = next((tk["sim_time_s"] for tk in tks if tk["hazard_active"]), None)
    if trigger_t is not None:
        ax.axvline(trigger_t, color="gray", lw=0.9, ls="--")
        ax.text(trigger_t + 0.05, 0.92, "trigger", fontsize=7, color="gray",
                transform=ax.get_xaxis_transform())

    col_str = "COLLISION" if trace["collision"] else "safe stop"
    margin  = trace["margin_m"]
    margin_str = f"{margin:+.1f}m" if margin != float("inf") else "inf"
    ax.set_title(
        f"Episode {trace['episode']}: {trace['speed_mph']:.0f} mph  "
        f"TTC={trace['ttc_s']}s  "
        f"{trace['walker_cross']}  |  "
        f"physics: {trace['label']} (margin {margin_str})  |  "
        f"outcome: {col_str}",
        fontsize=9, loc="left",
    )
    ax.set_ylabel("Brake command", fontsize=8)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(t_arr[0] - 0.1, t_arr[-1] + 0.1)
    ax.grid(alpha=0.25)
    if row_idx == 0:
        ax.legend(fontsize=7.5, loc="upper right", ncol=3, framealpha=0.88)
    if row_idx == n_ep - 1:
        ax.set_xlabel("Simulation time (s)", fontsize=9)

    # Outcome-colored border
    border_col = "#e74c3c" if trace["collision"] else "#2ecc71"
    for spine in ax.spines.values():
        spine.set_edgecolor(border_col)
        spine.set_linewidth(1.8)

fig.suptitle(
    f"Brake Command Over Time: {MODEL_PATH.replace('_',' ')} vs Fixed Profiles\n"
    f"Solid = SAC action.  Dashed/dotted = what each fixed profile would have targeted at the same state.\n"
    f"Yellow band = hazard active.  Vertical dashed = trigger fire.  "
    f"Red border = collision, green = safe.",
    fontsize=10,
)
fig.tight_layout()

png_out = os.path.join(script_dir, f"brake_curves_{n_ep}episodes.png")
fig.savefig(png_out, dpi=150)
print(f"Saved brake curve figure -> {png_out}")
