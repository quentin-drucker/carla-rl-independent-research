"""
rl_reward_design.py
Quentin | Junior Spring Research, 2026.

=============================================================================
REINFORCEMENT LEARNING DESIGN — AEB Braking Controller
=============================================================================

PURPOSE OF THIS FILE
--------------------
This file designs (but does not yet train) an RL-based braking controller
to replace the fixed braking profiles (proportional_ramp, step_constant, etc.)
in lane_follow.py.

It contains:
  1. A plain-English explanation of every RL concept as it applies HERE.
  2. The state vector definition — what the agent "sees" each tick.
  3. The action space — what the agent can "do" each tick.
  4. The reward function — how we score the agent's behavior each tick.
  5. Implementation notes — how this would plug into the existing simulation.

You do NOT need to understand neural networks to understand this file.
The reward function is just math — the neural network part is handled
by a library (stable-baselines3) that you call like any other Python package.

=============================================================================
SECTION 1 — KEY CONCEPTS
=============================================================================

POLICY
------
The "policy" is the brain of the agent. It is a function:

    brake_force = policy(state)

At every tick, the policy looks at the current state and outputs a brake
force between 0.0 (no braking) and 1.0 (full braking).

Before training: the policy is random — it outputs garbage.
After training:  the policy has learned, from thousands of simulated runs,
                 which brake force tends to lead to the best outcome.

EPISODE
-------
One episode = one full scenario run (equivalent to one call to run_scenario()).
The agent experiences the episode tick by tick, collecting rewards.
After the episode ends (collision, full stop, or sim_seconds reached),
the environment resets and a new episode begins.

TRAINING
--------
Training = running thousands of episodes and using the reward signal to
nudge the policy toward better decisions after each episode.

We will train TWO algorithms and compare them:

  PPO (Proximal Policy Optimization) — on-policy, stable, lower sample efficiency
  SAC (Soft Actor-Critic)            — off-policy, replay buffer, higher sample efficiency

See Section 5 for a full comparison and the implementation code for both.
Both are available in stable-baselines3 and both support continuous actions.

CONTINUOUS vs DISCRETE ACTION
------------------------------
Discrete: the agent picks from a fixed menu of brake values: {0, 0.25, 0.5, 0.75, 1.0}
Continuous: the agent outputs any value in [0.0, 1.0]

THIS DESIGN USES CONTINUOUS ACTIONS.
The agent outputs a single float brake_force ∈ [0.0, 1.0] every tick.
There is no menu — the agent can output 0.412 or 0.871 or any value.
This matches Delavari et al. (2025) Table 3: "Longitudinal only, continuous [0,1],
direct mapping." SAC requires continuous actions. PPO supports both but works
equally well with continuous.

=============================================================================
SECTION 2 — STATE VECTOR (what the agent observes each tick)
=============================================================================

The state must contain enough information for the agent to make a good decision.
Think of it as "what a skilled human driver would look at before deciding
how hard to press the brake."

Key principle: inputs should be NORMALIZED to roughly [0, 1] or [-1, 1].
Neural networks train much better on normalized inputs.
"""

import math


# ---------------------------------------------------------------------------
# State vector specification
# ---------------------------------------------------------------------------
# Each entry: (variable_name, description, how_to_compute, normalization)
#
# The final state fed to the RL policy is a 1D numpy array of these values,
# in this order. Changing the order or length requires retraining.

STATE_SPEC = [
    # 1. How fast the car is going, normalized to a 0-60 mph range.
    #    Why: the right brake force depends heavily on current speed.
    #    Faster = need to brake harder/earlier for the same stopping distance.
    {
        "name": "ego_speed_norm",
        "description": "Current ego vehicle speed, normalized 0-1",
        "raw_source": "get_speed_mps(vehicle)",
        "normalization": "speed_mps / 26.82",   # 26.82 m/s = 60 mph
    },

    # 2. How close the nearest obstacle is, normalized to the trigger zone.
    #    Why: this is the primary input — how much danger is there right now?
    #    0.0 = nothing detected / far away. 1.0 = at panic distance (right on top).
    {
        "name": "obstacle_proximity_norm",
        "description": "Normalized obstacle closeness: 0=far/clear, 1=at panic distance",
        "raw_source": "d_min_ahead from LiDAR noodle",
        "normalization": "clamp((trigger_distance_m - d_min_ahead) / trigger_distance_m, 0, 1)"
                         " — 0 when no hazard, rises toward 1 as obstacle closes in",
    },

    # 3. Time-to-collision (TTC), clipped and normalized.
    #    Why: distance alone doesn't capture urgency — 5m at 5mph is fine,
    #         5m at 45mph is a crisis. TTC combines both into one number.
    #    0.0 = no hazard detected. 1.0 = TTC ≤ 0.5s (imminent collision).
    {
        "name": "ttc_urgency_norm",
        "description": "TTC urgency: 0=safe/no hazard, 1=collision imminent (TTC ≤ 0.5s)",
        "raw_source": "d_min_ahead / ego_speed_mps",
        "normalization": "clamp(1.0 - (ttc / TTC_MAX_S), 0, 1)  where TTC_MAX_S = 4.0",
    },

    # 4. The previous brake command the agent applied.
    #    Why: the agent should know what it just did so it can smooth its own
    #         commands (avoid jerk from large tick-to-tick swings).
    {
        "name": "prev_brake_cmd",
        "description": "Brake command applied last tick [0, 1]. Already normalized.",
        "raw_source": "speed_state['brake_prev']",
        "normalization": "already in [0, 1]",
    },

    # 5. Whether a hazard is currently active (binary flag).
    #    Why: gives the agent a clear "hazard zone entered" signal, separate
    #         from the continuous proximity measure.
    {
        "name": "hazard_active",
        "description": "1.0 if LiDAR detects obstacle inside trigger zone, else 0.0",
        "raw_source": "1.0 if hazard_active else 0.0",
        "normalization": "binary, already in {0, 1}",
    },
]

# State vector length (must match number of entries above)
STATE_DIM = len(STATE_SPEC)   # = 5


def build_state_vector(
    ego_speed_mps: float,
    d_min_ahead,          # float or None
    trigger_distance_m: float,
    prev_brake_cmd: float,
    hazard_active: bool,
    ttc: float,           # pre-computed as d_min / speed, or inf if no hazard
    *,
    speed_max_mps: float = 26.82,   # 60 mph
    ttc_max_s: float = 4.0,
) -> list:
    """
    Build the normalized state vector from raw simulation values.
    Returns a list of 5 floats, ready to be passed to the RL policy.

    This function would be called inside the main simulation loop,
    replacing the brake_target calculation in lane_follow_step().
    """
    import numpy as np

    # 1. Speed
    speed_norm = min(ego_speed_mps / speed_max_mps, 1.0)

    # 2. Obstacle proximity
    if d_min_ahead is None or d_min_ahead >= trigger_distance_m:
        proximity_norm = 0.0
    else:
        proximity_norm = max(0.0, min(1.0,
            (trigger_distance_m - d_min_ahead) / max(trigger_distance_m, 1e-3)
        ))

    # 3. TTC urgency
    if ttc == float("inf") or ttc >= ttc_max_s:
        ttc_urgency = 0.0
    else:
        ttc_urgency = max(0.0, min(1.0, 1.0 - (ttc / ttc_max_s)))

    # 4. Previous brake (already normalized)
    brake_prev_norm = max(0.0, min(1.0, prev_brake_cmd))

    # 5. Hazard flag
    hazard_flag = 1.0 if hazard_active else 0.0

    return [speed_norm, proximity_norm, ttc_urgency, brake_prev_norm, hazard_flag]


# =============================================================================
# SECTION 3 — ACTION SPACE
# =============================================================================
#
# The agent outputs ONE number each tick: the desired brake force.
#
# Action: brake_force ∈ [0.0, 1.0]
#   0.0 = no braking (full cruise)
#   1.0 = maximum braking (panic stop)
#
# This replaces the brake_target calculation inside lane_follow_step().
# Everything else (the ramp limiter, STOP_HOLD mode, RECOVER mode) stays
# exactly the same — the RL agent only replaces how brake_target is set.
#
# The ramp limiter (braking_ramp_up_per_s) still applies on top of the
# agent's output, so the agent doesn't need to learn smoothing — it's
# already built into the existing infrastructure.
#
# Action space type: Box(low=0.0, high=1.0, shape=(1,), dtype=float32)
# (This is stable-baselines3 / gymnasium notation)

ACTION_DIM = 1   # single continuous value


# =============================================================================
# SECTION 4 — REWARD FUNCTION
# =============================================================================
#
# This is the most critical design decision in the whole RL setup.
# The reward function defines what "good driving" means numerically.
#
# We compute a reward R every tick. At the end of an episode, the agent
# has accumulated a total reward = sum of all per-tick rewards.
# The agent's goal is to learn the policy that maximizes this total.
#
# Our reward has three components:
#
#   R_total = R_safety + R_comfort + R_efficiency
#
# Each component is explained below with motivation and tuning notes.
#
# IMPORTANT: the weights (W_*) are hyperparameters. The right values
# depend on which objective you care most about. Suggested starting
# values are given, but these will need tuning based on training results.
#
# A good sanity check: for a "perfect" run (smooth stop, no collision,
# comfortable ride), R_total per tick should be small and positive.
# For a collision tick, R_total should be a large negative number.

# ---------------------------------------------------------------------------
# Reward weights — tune these to change the safety/comfort/efficiency tradeoff
# ---------------------------------------------------------------------------

W_COLLISION     = 200.0   # penalty for hitting the pedestrian (applied once at collision tick)
                           # Large because collision is the worst possible outcome.

W_TTC_DANGER    = 2.0     # per-tick penalty when TTC drops below danger threshold
                           # Penalizes getting close even without a collision.

W_JERK          = 0.05    # per-tick penalty for jerk (uncomfortable braking)
                           # Small per tick but accumulates over many ticks of rough braking.

W_SMOOTH_ACTION = 0.5     # per-tick penalty for large tick-to-tick changes in brake command
                           # Encourages the agent to output smooth, gradual commands.

W_CRUISE_BONUS  = 0.1     # per-tick reward for maintaining cruise speed when no hazard
                           # Discourages unnecessary braking / phantom braking.

W_STOP_BONUS    = 5.0     # one-time reward for achieving a clean full stop before the ped
                           # Rewards the ideal outcome explicitly.

# Danger thresholds
TTC_DANGER_THRESHOLD_S  = 1.5   # TTC below this starts incurring penalty
TTC_CRITICAL_THRESHOLD_S = 0.5  # TTC below this incurs maximum penalty


def compute_reward(
    *,
    collision_this_tick: bool,
    ttc_s: float,                    # float("inf") if no hazard
    jerk_mps3: float,                # absolute jerk this tick
    brake_cmd_this_tick: float,      # brake command applied this tick [0, 1]
    brake_cmd_prev_tick: float,      # brake command last tick [0, 1]
    ego_speed_mps: float,            # current speed
    target_speed_mps: float,         # cruise target speed
    hazard_active: bool,
    full_stop_achieved: bool,        # True on the tick ego first reaches < 0.3 m/s after hazard
    fixed_dt: float = 0.02,          # simulation timestep (50 Hz)
) -> tuple:
    """
    Compute the per-tick reward signal for the RL agent.

    Returns (total_reward, reward_breakdown_dict) so you can log each
    component separately and understand what the agent is learning from.

    SIGN CONVENTION: positive = good, negative = bad.
    The agent maximizes total reward, so penalties are negative values.
    """

    # ------------------------------------------------------------------
    # R_safety: penalize dangerous proximity and collisions
    # ------------------------------------------------------------------

    r_safety = 0.0

    # Hard collision penalty — applied once at the tick the collision occurs.
    # This is the dominant signal: the agent must learn that collision = very bad.
    if collision_this_tick:
        r_safety -= W_COLLISION

    # Soft TTC penalty — continuous signal that gets stronger as TTC drops.
    # This teaches the agent to start worrying BEFORE the collision, not only at it.
    # Without this, the agent might learn to wait until the last moment.
    if ttc_s < TTC_DANGER_THRESHOLD_S and ttc_s > 0:
        # Normalized danger level: 0.0 at threshold, 1.0 at critical threshold
        danger = max(0.0, min(1.0,
            (TTC_DANGER_THRESHOLD_S - ttc_s) /
            (TTC_DANGER_THRESHOLD_S - TTC_CRITICAL_THRESHOLD_S + 1e-6)
        ))
        r_safety -= W_TTC_DANGER * danger * fixed_dt
        # Note: multiply by fixed_dt so the per-second penalty is consistent
        # regardless of simulation timestep.

    # ------------------------------------------------------------------
    # R_comfort: penalize jerky / uncomfortable braking
    # ------------------------------------------------------------------

    r_comfort = 0.0

    # Jerk penalty — proportional to jerk magnitude.
    # Only applies when a hazard is active (pre-trigger jerk is irrelevant,
    # same logic as in the existing simulation loop).
    if hazard_active:
        r_comfort -= W_JERK * jerk_mps3 * fixed_dt

    # Action smoothness penalty — penalizes large jumps in brake command
    # between consecutive ticks. This directly discourages the slam-brake
    # behavior of "step_constant" and rewards gradual, smooth control.
    brake_delta = abs(brake_cmd_this_tick - brake_cmd_prev_tick)
    r_comfort -= W_SMOOTH_ACTION * brake_delta * fixed_dt

    # ------------------------------------------------------------------
    # R_efficiency: reward good driving behavior
    # ------------------------------------------------------------------

    r_efficiency = 0.0

    # Cruise bonus — small reward each tick the car is near target speed
    # with no hazard active. Discourages phantom braking (braking for no reason).
    if not hazard_active:
        speed_closeness = max(0.0, 1.0 - abs(ego_speed_mps - target_speed_mps) / target_speed_mps)
        r_efficiency += W_CRUISE_BONUS * speed_closeness * fixed_dt

    # Full stop bonus — one-time reward when the car achieves a clean stop.
    # This is the "gold star" for the ideal outcome.
    if full_stop_achieved:
        r_efficiency += W_STOP_BONUS

    # ------------------------------------------------------------------
    # Total
    # ------------------------------------------------------------------
    r_total = r_safety + r_comfort + r_efficiency

    breakdown = {
        "r_total":      r_total,
        "r_safety":     r_safety,
        "r_comfort":    r_comfort,
        "r_efficiency": r_efficiency,
        "collision":    collision_this_tick,
        "ttc_s":        ttc_s,
        "jerk_mps3":    jerk_mps3,
        "brake_delta":  brake_delta,
    }

    return r_total, breakdown


# =============================================================================
# SECTION 5 — HOW THIS CONNECTS TO THE EXISTING SIMULATION
# =============================================================================
#
# The existing simulation loop in test3___ped_intrusion_scenario.py calls
# lane_follow_step() every tick, which internally computes brake_target
# based on the selected brake_profile.
#
# To use an RL agent instead, the change is surgical:
#
#   BEFORE (fixed profile):
#       brake_target = <computed by profile logic in lane_follow_step()>
#
#   AFTER (RL agent):
#       state = build_state_vector(...)          # defined above
#       brake_target, _ = rl_policy.predict(state, deterministic=True)
#       reward, breakdown = compute_reward(...)  # score the decision
#       # store (state, action, reward, next_state) for training
#
# Everything else — the ramp limiter, STOP_HOLD, RECOVER, telemetry,
# the sweep pipeline — stays exactly the same.
#
# =============================================================================
# ALGORITHM COMPARISON: PPO vs SAC
# =============================================================================
#
# We train BOTH algorithms and compare them against each other and against
# the four fixed braking profiles. This gives three comparison tiers:
#
#   Tier 1: Fixed rules (proportional_ramp, step_constant, cautious_ramp, exponential)
#   Tier 2: RL-PPO trained policy
#   Tier 3: RL-SAC trained policy
#
# All three tiers are evaluated on the identical 48-scenario sweep CSV,
# so the comparison is apples-to-apples across identical conditions.
#
# ---- PPO (Proximal Policy Optimization) ----
# Type: On-policy (discards experience after each update)
# Strengths:
#   - Most stable training — lowest variance, least likely to collapse
#   - Simplest to tune (clip_range is the main knob)
#   - Good baseline: if PPO can't solve it, something is wrong with the env
#   - Medium article CarRacing benchmark: PPO achieved 867 ± 147 (consistent)
# Weaknesses:
#   - Low sample efficiency — throws away data after every update
#   - CARLA episodes are ~30-40 seconds; each episode is expensive to run
#   - Needs more total timesteps than SAC to reach same performance
# Best for: establishing a reliable baseline, stable first training run
#
# ---- SAC (Soft Actor-Critic) ----
# Type: Off-policy (stores all past experience in a replay buffer, reuses it)
# Strengths:
#   - Much higher sample efficiency — learns from past CARLA episodes repeatedly
#   - Entropy regularization: explores braking nuances automatically
#     (the "temperature" alpha auto-tunes to balance explore vs exploit)
#   - Medium article CarRacing benchmark: SAC achieved 906 ± 22 (best mean)
#   - Delavari et al. (2025): SAC and PPO together dominate CARLA AV literature
#   - CONTINUOUS ACTIONS ONLY — perfectly matched to our [0,1] brake space
# Weaknesses:
#   - Slightly more hyperparameters (tau, buffer_size, ent_coef)
#   - Off-policy can be unstable early in training; needs a warm-up period
# Best for: final production-quality policy; critical when episodes are slow
#
# RECOMMENDATION: Train SAC first (better suited to slow CARLA environment).
# Train PPO second as a stability check. Compare both on the 48-run sweep.
# Source: Delavari et al. (2025), "A Survey on Reinforcement Learning for
# Autonomous Driving," Section 3.1; Zai & Brown (2020) algorithm comparisons.
#
# =============================================================================
# TRAINING SETUP
# =============================================================================
#
# Step 1: Install dependencies
#       pip install stable-baselines3[extra] gymnasium
#
# Step 2: Wrap run_scenario() as a Gymnasium environment
#       - observation_space = Box(shape=(STATE_DIM,), low=0, high=1, dtype=float32)
#       - action_space      = Box(shape=(1,), low=0.0, high=1.0, dtype=float32)
#       - step()  → calls one tick, returns (obs, reward, terminated, truncated, info)
#       - reset() → reinitializes scenario, returns (obs, info)
#
# Step 3a: Train with SAC (recommended first)
#
#       from stable_baselines3 import SAC
#       model = SAC(
#           "MlpPolicy", env,
#           verbose=1,
#           learning_rate=3e-4,     # Adam LR — standard starting point
#           buffer_size=10_000,     # replay buffer; keep small for memory (CARLA is heavy)
#           batch_size=256,         # mini-batch per gradient update
#           tau=0.005,              # soft target network update rate
#           gamma=0.99,             # discount factor — care about future rewards
#           ent_coef="auto",        # auto-tune entropy temperature (no manual alpha tuning)
#           train_freq=1,           # update after every env step
#           gradient_steps=1,
#       )
#       model.learn(total_timesteps=200_000)
#       model.save("aeb_sac_policy")
#
# Step 3b: Train with PPO (comparison baseline)
#
#       from stable_baselines3 import PPO
#       model = PPO(
#           "MlpPolicy", env,
#           verbose=1,
#           n_steps=2048,           # steps collected per update (one ~50-tick episode = ~2048)
#           batch_size=64,          # mini-batch size for surrogate loss
#           n_epochs=10,            # passes over collected data per update
#           learning_rate=2.5e-4,
#           gamma=0.99,
#           gae_lambda=0.95,        # generalized advantage estimation smoothing
#           clip_range=0.2,         # trust-region clip — keeps updates conservative
#       )
#       model.learn(total_timesteps=500_000)   # PPO needs ~2.5x more steps than SAC
#       model.save("aeb_ppo_policy")
#
# Step 4: Evaluate both policies
#       Run each saved policy through run_sweep() with the same 48 configs.
#       Compare the resulting sweep_summary.csv files side by side.
#       Key columns to compare: outcome, min_ped_distance_m, max_jerk_mps3,
#       mean_jerk_mps3, time_to_stop_s.
#
# Step 5: Log reward breakdown during training
#       The compute_reward() function returns a breakdown dict with r_safety,
#       r_comfort, r_efficiency separately. Log all three via:
#           info["r_safety"] = breakdown["r_safety"]
#       in your Gymnasium env's step() method, then monitor in TensorBoard.
#       Delavari et al. (2025) Section 10.4 warns that aggregating multiple
#       objectives into one scalar hides trade-offs — logging separately
#       directly addresses this.


# =============================================================================
# SECTION 6 — REWARD FUNCTION SANITY CHECKS
# =============================================================================
#
# Run this file directly to see what the reward function outputs for
# representative "good", "okay", and "bad" tick scenarios.

if __name__ == "__main__":

    print("=" * 60)
    print("REWARD FUNCTION SANITY CHECKS")
    print("=" * 60)

    scenarios = [
        {
            "label": "GOOD: smooth cruise, no hazard",
            "kwargs": dict(
                collision_this_tick=False, ttc_s=float("inf"),
                jerk_mps3=5.0, brake_cmd_this_tick=0.0,
                brake_cmd_prev_tick=0.0, ego_speed_mps=15.64,
                target_speed_mps=15.64, hazard_active=False,
                full_stop_achieved=False,
            ),
        },
        {
            "label": "OKAY: gentle ramp braking, TTC=2.0s",
            "kwargs": dict(
                collision_this_tick=False, ttc_s=2.0,
                jerk_mps3=30.0, brake_cmd_this_tick=0.3,
                brake_cmd_prev_tick=0.25, ego_speed_mps=12.0,
                target_speed_mps=15.64, hazard_active=True,
                full_stop_achieved=False,
            ),
        },
        {
            "label": "GOOD: full stop achieved cleanly",
            "kwargs": dict(
                collision_this_tick=False, ttc_s=0.8,
                jerk_mps3=20.0, brake_cmd_this_tick=0.5,
                brake_cmd_prev_tick=0.48, ego_speed_mps=0.1,
                target_speed_mps=15.64, hazard_active=True,
                full_stop_achieved=True,
            ),
        },
        {
            "label": "BAD: step-constant slam brake (high jerk + high delta)",
            "kwargs": dict(
                collision_this_tick=False, ttc_s=1.2,
                jerk_mps3=300.0, brake_cmd_this_tick=1.0,
                brake_cmd_prev_tick=0.0, ego_speed_mps=10.0,
                target_speed_mps=15.64, hazard_active=True,
                full_stop_achieved=False,
            ),
        },
        {
            "label": "WORST: collision tick",
            "kwargs": dict(
                collision_this_tick=True, ttc_s=0.1,
                jerk_mps3=350.0, brake_cmd_this_tick=1.0,
                brake_cmd_prev_tick=0.9, ego_speed_mps=8.0,
                target_speed_mps=15.64, hazard_active=True,
                full_stop_achieved=False,
            ),
        },
    ]

    for s in scenarios:
        r_total, breakdown = compute_reward(**s["kwargs"])
        print(f"\n  {s['label']}")
        print(f"    R_total={r_total:+.4f}  "
              f"(safety={breakdown['r_safety']:+.4f}, "
              f"comfort={breakdown['r_comfort']:+.4f}, "
              f"efficiency={breakdown['r_efficiency']:+.4f})")

    print("\n" + "=" * 60)
    print("WEIGHT SUMMARY")
    print("=" * 60)
    print(f"  W_COLLISION     = {W_COLLISION}   (collision penalty, one-time)")
    print(f"  W_TTC_DANGER    = {W_TTC_DANGER}    (per-second TTC danger penalty)")
    print(f"  W_JERK          = {W_JERK}   (per-second jerk penalty)")
    print(f"  W_SMOOTH_ACTION = {W_SMOOTH_ACTION}   (per-second action-delta penalty)")
    print(f"  W_CRUISE_BONUS  = {W_CRUISE_BONUS}   (per-second cruise reward)")
    print(f"  W_STOP_BONUS    = {W_STOP_BONUS}    (one-time full-stop reward)")
    print()
    print("Tune these weights to shift the safety/comfort/efficiency tradeoff.")
    print("Higher W_COLLISION = agent prioritizes collision avoidance above all.")
    print("Higher W_JERK = agent prioritizes passenger comfort.")
    print("Higher W_CRUISE_BONUS = agent is less trigger-happy with phantom braking.")