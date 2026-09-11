"""
continue_sac.py
Quentin | Junior Spring Research, 2026.

Continues training an existing SAC model for additional steps.
Loads sac_aeb_rand_800k and trains for another 800k steps -> 1.6M total.

Usage:
    python continue_sac.py                          <- 800k -> 1.6M (default)
    python continue_sac.py sac_aeb_rand_800k 800000 <- explicit args

=============================================================================
IMPORTANT — RUN CARLA HEADLESS FOR OVERNIGHT TRAINING
=============================================================================
    "C:\\Users\\qdruc\\OneDrive\\Desktop\\CARLA_0.9.16\\CarlaUE4.exe" -RenderOffScreen

Then run this script normally.
=============================================================================

NOTE on replay buffer:
  SB3 does not save the replay buffer in the .zip checkpoint by default.
  Continuing training starts with an empty replay buffer, which is normal.
  The policy network weights are fully restored. The first ~10k steps will
  collect fresh experience; training quality is unaffected after that.

NOTE on step counter:
  reset_num_timesteps=False preserves the step counter from 800k onward.
  TensorBoard logs and checkpoint filenames will reflect the continued count.
"""

import os
import sys
import math
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from carla_aeb_env import CarlaAEBEnv
from train_sac import sample_config

# Reset entropy to this value after loading. By the end of a long run,
# ent_coef decays near zero — continuing with it that low means the agent
# barely explores the new training distribution and policy degrades.
# 0.15 is enough to re-explore without forgetting the learned policy.
RESET_ENT_COEF = 0.15

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

LOAD_PATH    = sys.argv[1] if len(sys.argv) > 1 else "sac_aeb_rand_800k"
EXTRA_STEPS  = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000

# Output naming: preserve model prefix, increment step count
import re
_m = re.search(r"^(.+?)_(\d+)k$", LOAD_PATH)
if _m:
    _prefix, _base_k = _m.group(1), int(_m.group(2))
else:
    _prefix, _base_k = "sac_continued", 0
_total_k = _base_k + EXTRA_STEPS // 1000
FINAL_PATH       = f"{_prefix}_{_total_k}k"
CHECKPOINT_DIR   = "checkpoints"
CHECKPOINT_EVERY = 20_000

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\n{'='*55}")
print(f"  SAC CONTINUED TRAINING")
print(f"  Load from   : {LOAD_PATH}.zip")
print(f"  Extra steps : {EXTRA_STEPS:,}")
print(f"  Final model : {FINAL_PATH}.zip")
print(f"  Checkpoint  : every {CHECKPOINT_EVERY:,} steps -> {CHECKPOINT_DIR}/")
print(f"  Stop early with Ctrl+C — latest checkpoint is safe.")
print(f"{'='*55}\n")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

env = CarlaAEBEnv(config_fn=sample_config)

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_EVERY,
    save_path=CHECKPOINT_DIR,
    name_prefix=FINAL_PATH,
    verbose=1,
)

try:
    model = SAC.load(LOAD_PATH, env=env)

    print(f"  Loaded model from {LOAD_PATH}.zip")
    print(f"  Continuing from step {model.num_timesteps:,}")

    # Reset entropy coefficient so the agent explores the new distribution.
    # After a long run ent_coef decays near zero; continuing with it that low
    # causes policy degradation rather than improvement.
    old_ent = float(th.exp(model.log_ent_coef).item())
    with th.no_grad():
        model.log_ent_coef.fill_(math.log(RESET_ENT_COEF))
    print(f"  Entropy reset: {old_ent:.4f} -> {RESET_ENT_COEF} (will re-decay over training)\n")

    model.learn(
        total_timesteps=EXTRA_STEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=False,   # preserve step counter from 800k
    )

    model.save(FINAL_PATH)
    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total steps : {model.num_timesteps:,}")
    print(f"  Saved -> {FINAL_PATH}.zip")
    print(f"{'='*55}\n")

except KeyboardInterrupt:
    interrupted_path = f"{FINAL_PATH}_interrupted_{model.num_timesteps}steps"
    model.save(interrupted_path)
    print(f"\n  Interrupted at step {model.num_timesteps:,}")
    print(f"  Saved -> {interrupted_path}.zip")

finally:
    env.close()
