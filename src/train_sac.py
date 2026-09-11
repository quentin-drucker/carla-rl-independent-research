"""
train_sac.py

SAC training script for the CARLA AEB environment.
Saves a checkpoint every 10,000 steps so you never lose more than
~10 minutes of training if you stop early.

Usage:
    python train_sac.py

To stop early: press Ctrl+C — the most recent checkpoint is kept.
Final model saved as: sac_v2_1600k.zip
Checkpoints saved as: checkpoints/sac_v2_XXXXX_steps.zip

=============================================================================
IMPORTANT — RUN CARLA IN HEADLESS (NO WINDOW) MODE FOR OVERNIGHT TRAINING
=============================================================================
Rendering the CARLA window wastes GPU and roughly halves training speed.
For overnight runs, close CARLA and relaunch it with:

    "C:\\Users\\qdruc\\OneDrive\\Desktop\\CARLA_0.9.16\\CarlaUE4.exe" -RenderOffScreen

The window will not appear — CARLA runs silently in the background.
Then run this script normally: python train_sac.py

Normal (windowed) launch is fine for quick smoke tests where you want to
watch what the agent is doing, but not for long training runs.
=============================================================================

RANDOMIZED CONFIG TRAINING:
Each episode samples a fresh ScenarioConfig drawn from the same parameter
space as the braking-profile sweep. This forces the agent to learn a general
braking policy rather than memorizing one fixed scenario.

Ranges sampled each episode (continuous uniform unless noted):
  target_mph          : 22 – 50 mph
  trigger_ttc_s       : 1.5 – 4.0 s
  walker_speed_mps    : 1.0 – 2.5 m/s
  encounter_distance_m: 50 – 130 m
  walker_cross        : "near" (67%) or "far" (33%)  <- discrete, weighted
  walker_side         : "left" or "right"            <- discrete, uniform

Fixed (not randomized):
  braking_ramp_up_per_s: 4.0  (controller knob, not a scenario param)
  brake_headway_s      : 2.5  (controller knob)
  road_friction        : None (CARLA default dry asphalt)
  weather_preset       : ClearNoon (no effect on numeric state vector)

Near-cross is weighted 2:1 over far-cross because far-cross episodes almost
never produce a collision regardless of braking strategy — the agent must face
enough near-cross near-miss situations to learn that braking is required.
"""

import os
import random
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from carla_aeb_env import CarlaAEBEnv
from scenario_config import ScenarioConfig


# ---------------------------------------------------------------------------
# Config sampler — called once per episode reset
# ---------------------------------------------------------------------------

def sample_config() -> ScenarioConfig:
    """
    Return a randomly sampled ScenarioConfig drawn from continuous ranges.

    Sampling strategy (v3):
      50% — fully random TTC (baseline coverage across all scenario types)
      50% — borderline-targeted: TTC chosen to land in the borderline-avoidable
             zone for the sampled speed (margin 0–8m). Data showed 69% collision
             rate at TTC=1.8s borderline-avoidable, and 50–58% at 22–28 mph —
             these short-window cases were underrepresented in earlier training.

    High-speed performance (SAC advantage at 40–45 mph) is preserved because:
      - 50% of episodes still sample TTC randomly (includes all scenario types)
      - The borderline-targeted 50% spans the full speed range including high speed
    """
    A_EFF = 3.5  # effective deceleration (ramp-limited), matches avoidability.py

    # Scenario parameters the agent must generalize across
    target_mph       = random.uniform(22.0, 50.0)
    walker_speed_mps = random.uniform(1.0, 2.5)

    # 50% of episodes: target the borderline-avoidable TTC range for this speed.
    # Borderline-avoidable: 0 <= v0*TTC - v0²/(2*A_EFF) < 8m
    # Solving for TTC: v0/(2*A_EFF) <= TTC < v0/(2*A_EFF) + 8/v0
    v0 = target_mph * 0.44704
    ttc_ba_min = v0 / (2.0 * A_EFF)
    ttc_ba_max = ttc_ba_min + 8.0 / v0
    ttc_ba_min = max(1.5, ttc_ba_min)
    ttc_ba_max = min(3.5, ttc_ba_max)

    if random.random() < 0.5 and ttc_ba_min < ttc_ba_max:
        trigger_ttc_s = random.uniform(ttc_ba_min, ttc_ba_max)
    else:
        trigger_ttc_s = random.uniform(1.5, 3.5)
    walker_speed_mps = random.uniform(1.0, 2.5)

    # Encounter distance must be large enough that the trigger doesn't fire at t=0.
    # If encounter_distance_m < speed * trigger_ttc_s, trigger fires immediately at
    # sim start before ego has built up speed, making TTC misleading.
    # Fix: set encounter_distance_m = lead_dist + random buffer of 30-80m.
    lead_dist_m          = (target_mph * 0.44704) * trigger_ttc_s
    encounter_distance_m = lead_dist_m + random.uniform(30.0, 80.0)
    encounter_distance_m = min(encounter_distance_m, 160.0)  # cap for route length

    # Discrete choices
    walker_cross = random.choices(["near", "far"], weights=[2, 1])[0]
    walker_side  = random.choice(["left", "right"])
    brake_headway_s = random.choice([2.5, 5.0])

    # sim_seconds: enough for a 50 mph car to stop from furthest encounter with margin
    sim_seconds = 30.0

    return ScenarioConfig(
        target_mph=target_mph,
        trigger_ttc_s=trigger_ttc_s,
        walker_speed_mps=walker_speed_mps,
        encounter_distance_m=encounter_distance_m,
        walker_cross=walker_cross,
        walker_side=walker_side,
        # Fixed: controller infrastructure, not scenario variation
        braking_ramp_up_per_s=4.0,
        brake_headway_s=brake_headway_s,
        road_friction=None,
        weather_preset="ClearNoon",
        sim_seconds=sim_seconds,
    )

if __name__ == "__main__":
    # --- Config ---
    TOTAL_STEPS      = 1_600_000
    CHECKPOINT_EVERY = 20_000
    CHECKPOINT_DIR   = "checkpoints"
    FINAL_PATH       = "sac_v3_1600k"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=CHECKPOINT_DIR,
        name_prefix="sac_v3",
        verbose=1,
    )

    print(f"\n{'='*55}")
    print(f"  SAC v3 TRAINING (6-dim state, borderline-targeted + headway randomized)")
    print(f"  {TOTAL_STEPS:,} steps — 50% random / 50% borderline TTC, headway=[2.5,5.0]")
    print(f"  Checkpoint every {CHECKPOINT_EVERY:,} steps -> {CHECKPOINT_DIR}/")
    print(f"  Final model     -> {FINAL_PATH}.zip")
    print(f"  Stop early with Ctrl+C — latest checkpoint is safe.")
    print(f"{'='*55}\n")

    env = CarlaAEBEnv(config_fn=sample_config)

    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            buffer_size=100_000,   # was 10k — too small, agent forgot early experiences
            batch_size=512,        # was 256 — larger batches = more stable updates
            learning_rate=3e-4,
            tau=0.005,
            ent_coef="auto",
        )

        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=checkpoint_callback,
        )

        model.save(FINAL_PATH)
        print(f"\n{'='*55}")
        print(f"  TRAINING COMPLETE")
        print(f"  Final model saved -> {FINAL_PATH}.zip")
        print(f"{'='*55}\n")

    except KeyboardInterrupt:
        interrupted_path = f"sac_v3_interrupted_{model.num_timesteps}steps"
        model.save(interrupted_path)
        print(f"\n{'='*55}")
        print(f"  Training interrupted at step {model.num_timesteps:,}")
        print(f"  Model saved -> {interrupted_path}.zip")
        print(f"  (Checkpoints also saved in {CHECKPOINT_DIR}/)")
        print(f"{'='*55}\n")

    finally:
        env.close()
