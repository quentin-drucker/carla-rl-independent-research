"""
carla_aeb_env.py
Quentin | Junior Spring Research, 2026.

=============================================================================
GYMNASIUM ENVIRONMENT WRAPPER — AEB Braking Controller
=============================================================================

PURPOSE
-------
This file wraps the existing CARLA pedestrian-intrusion scenario as a
standard Gymnasium environment so that stable-baselines3 can train RL
agents (SAC, PPO) against it.

What this file does NOT change:
  - The scenario physics, pedestrian logic, LiDAR braking infrastructure
  - lane_follow_step() — only one new parameter (rl_brake_override) was added
  - The sweep pipeline, comparison plots, ScenarioConfig

What this file adds:
  - reset()  → spawns a fresh episode, returns the initial 5-number state
  - step()   → applies one brake action, ticks CARLA, returns reward + next state
  - close()  → tears down CARLA actors cleanly

HOW TO USE
----------
    from carla_aeb_env import CarlaAEBEnv
    from scenario_config import ScenarioConfig
    from stable_baselines3 import SAC

    env = CarlaAEBEnv()                     # connects to CARLA
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save("aeb_sac_policy")
    env.close()

PREREQUISITE: CARLA must already be running (CarlaUE4.exe).
"""

import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Gymnasium import (pip install gymnasium)
# ---------------------------------------------------------------------------
import gymnasium
from gymnasium.spaces import Box

# ---------------------------------------------------------------------------
# CARLA PythonAPI path setup (mirrors test3 script)
# ---------------------------------------------------------------------------
CARLA_ROOT = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16"
for _p in [
    os.path.join(CARLA_ROOT, "PythonAPI"),
    os.path.join(CARLA_ROOT, "PythonAPI", "carla"),
]:
    if _p not in sys.path:
        sys.path.append(_p)

import carla  # pyright: reportMissingImports=false
from agents.navigation.global_route_planner import GlobalRoutePlanner

# ---------------------------------------------------------------------------
# Project module imports
# ---------------------------------------------------------------------------
from math_utils import get_speed_mps, mps_to_mph
from spawning import prepare_spawn_context, spawn_ego_vehicle
from lidar_sensor import attach_lidar_sensor
from loop_utils import get_latest_lidar_frame
from lane_follow import lane_follow_step
from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from scenario_config import ScenarioConfig, crossing_lateral_offsets_m
from rl_collision_signal import (
    COLLISION_SIGNAL_LEGACY,
    combine_collision_signals,
    geometric_clearance_m,
    legacy_proximity_contact,
    validate_collision_signal_mode,
)
from walker_utils import (
    spawn_scripted_walker,
    init_scripted_crossing_state,
    trigger_scripted_crossing,
    update_scripted_crossing,
    destroy_scripted_walker,
)
from rl_reward_design import build_state_vector, compute_reward, STATE_DIM
# Steering mode only (Phase 2; opt-in via steer_commit=...):
from map_drivability import footprint_drivable
from passage import Obstacle, passage_from_extent, predicted_lateral_at_arrival
from passage_carla import passage_at_location
from walker_script import WalkerScript
from merge_back import DRIVE_ON_SPEED_MPS, MergeBack, hazard_behind_ego, path_clear_of_hazard
from pedestrian_contact import DEFAULT_PEDESTRIAN_RADIUS_M
from rl_steering import (
    contact_is_pedestrians,
    STEERING_STATE_DIM,
    SteeringCommitment,
    SteeringRewardConfig,
    build_steering_observation,
    compute_steering_reward,
    frame_coordinates_m,
    heading_error_rad,
    split_action,
)


# ---------------------------------------------------------------------------
# Scenario constants (mirrors test3___ped_intrusion_scenario.py)
# ---------------------------------------------------------------------------
TARGET_MAP            = "Town04_Opt"
EGO_SPAWN_INDEX       = 242
END_MARKER_SPAWN_INDEX = 168

FIXED_DT              = 0.02    # 50 Hz

LOOKAHEAD_M           = 6
STEER_GAIN            = 1.5

CONE_HALF_ANGLE_DEG   = 13.0
MAX_CONE_YAW_DEG      = 45.0
CONE_DRAW_LENGTH_M    = 80.0
LANE_WIDTH_M          = 3.0
LANE_SIDE_MARGIN_M    = 0.4
CONE_MAX_LATERAL_M    = (LANE_WIDTH_M * 0.5) + LANE_SIDE_MARGIN_M

BRAKE_BASE_DISTANCE_M = 5.0
PANIC_DISTANCE_M      = 5.0
RAMP_UP_PER_S         = 8.0
RAMP_DOWN_PER_S       = 1.0

NOODLE_ENABLE         = True
NOODLE_STEP_M         = 1.25
NOODLE_HALF_WIDTH_M   = 1.4
NOODLE_MAX_DIST_M     = CONE_DRAW_LENGTH_M
NOODLE_X_MIN_M        = 2.5

# Debug markers are refreshed every tick with this lifetime so they expire
# automatically when the episode ends (no bleed-over into the next episode).
# Must be > FIXED_DT (0.02s). Two ticks of headroom is enough.
WALKER_DEBUG_LIFE_S   = 0.05

# Maximum time allowed after the hazard trigger before the episode is force-ended.
# Covers the case where the ped gets nudged off course and never reaches its
# destination (crossing_state["done"] never fires). Normal crossings complete
# well within this window: slowest far-cross = 7m / 1.0 m/s = 7s.
POST_TRIGGER_TIMEOUT_S = 12.0

# Jerk low-pass filter alpha (same value as test3)
JERK_LP_ALPHA         = 0.25


# ===========================================================================
# Gymnasium environment
# ===========================================================================

class CarlaAEBEnv(gymnasium.Env):
    """
    Gymnasium-compatible environment for training an RL braking controller
    in CARLA's pedestrian-intrusion scenario.

    Observation space: Box(6,) — normalized state vector (see rl_reward_design.py)
    Action space:      Box(1,) — brake force in [0.0, 1.0]

    The agent replaces the brake_profile logic in lane_follow_step().
    Everything else (steering, cruise control, ramp limiter, STOP_HOLD, etc.)
    is unchanged from the existing simulation.
    """

    metadata = {"render_modes": []}

    # -----------------------------------------------------------------------
    def __init__(self, cfg: ScenarioConfig = None, config_fn=None,
                 collision_signal: str = COLLISION_SIGNAL_LEGACY,
                 steer_commit: str = None, u_bounds=(0.0, 1.0), lock_delay_s: float = 0.0,
                 steering_reward: SteeringRewardConfig = None, road_right_limit_m: float = None,
                 merge_back: bool = False, map_name: str = TARGET_MAP, route_builder=None,
                 same_direction_passage: bool = False):
        """
        Args:
            cfg: ScenarioConfig instance. Defaults to ScenarioConfig() if None.
                 Used as the fixed config when config_fn is None.
                 brake_profile is ignored (the RL agent controls braking instead).
            config_fn: Optional callable () -> ScenarioConfig. When provided,
                 called at the start of every reset() to sample a fresh config.
                 This enables episode-to-episode randomization for generalization
                 training. When set, the cfg argument is only used as a fallback
                 for the first episode before config_fn is first called.
            collision_signal: what counts as a collision for reward and
                 termination (rl_collision_signal.py). "legacy" (default) =
                 CARLA's sensor OR the near-cross proximity fallback, exactly
                 the semantics v3 was trained and evaluated with. "geometric"
                 adds oriented-footprint pedestrian contact in every direction
                 (catches the stationary/far-cross hits the sensor misses).
                 Every component is reported in info either way.
            steer_commit: None (default) = last semester's braking-only
                 environment, unchanged. "per_tick" or "lock_at_onset" turns
                 on the Phase 2 STEERING MODE (rl_steering.py):
                   - action = [brake, u]; observation = 14 values;
                   - the hazard starts by script at the trigger ("oracle
                     onset", as in test26), not by LiDAR detection, and the
                     policy has no effect before it;
                   - contact is the oriented-footprint check; the reward is
                     compute_steering_reward;
                   - the episode runs the full scoring window after onset
                     unless there is contact (no ending at a full stop).
                 "lock_at_onset" freezes the route (passage and lateral
                 target) at onset; braking stays per tick.
            u_bounds: (u_min, u_max) for the steering action. (0, 1) = right
                 swerves only; (-1, 1) opens both sides.
            lock_delay_s: lock_at_onset locks this long after onset
                 instead of at onset (0 = at onset). For a walking
                 pedestrian, who has no speed yet on the onset tick.
            steering_reward: SteeringRewardConfig; None = its defaults.
            road_right_limit_m: caps the passage's right road edge at this
                 route-relative lateral (e.g. 4.5 = the adjacent lane's far
                 edge, so u = 1 aims at the adjacent lane, not three lanes
                 over). None (default) = the whole drivable road.
            merge_back: Option A merge-back (merge_back.py): once the
                 hazard is confirmed clear behind the car, hazard mode ends
                 (cruise resumes) and the lateral target returns smoothly to
                 the original lane; the policy has no effect from then on.
                 False (default) = swerve out and hold, as before.
            map_name / route_builder / same_direction_passage: city circuit
                 (run_circuit.py). route_builder(world) -> (ego spawn
                 Transform, route waypoints); None = the Town04 test route.
                 same_direction_passage: the passage never extends into
                 oncoming lanes. next_encounter() starts the next hazard
                 further along the same route without respawning the car.
        """
        self._map_name, self._route_builder = map_name, route_builder
        self._same_direction_passage = same_direction_passage
        super().__init__()

        self._collision_signal = validate_collision_signal_mode(collision_signal)
        self._config_fn = config_fn
        self.cfg = cfg or ScenarioConfig()
        self._target_speed_mps = self.cfg.target_mph * 0.44704
        self._total_ticks = int(self.cfg.sim_seconds / FIXED_DT)

        # Gymnasium spaces --------------------------------------------------
        # Observation: 5 normalized floats (see STATE_SPEC in rl_reward_design.py)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        # Action: single continuous brake force
        self.action_space = Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Steering mode (opt-in) replaces both spaces ------------------------
        self._steer_commit = steer_commit
        if steer_commit is not None:
            self._u_min, self._u_max = u_bounds
            self._lock_delay_ticks = int(round(lock_delay_s / FIXED_DT))
            self._road_right_limit_m = road_right_limit_m
            self._merge_back_enabled = merge_back
            self._new_commitment()  # fail fast on a bad mode or delay
            self._steering_reward = steering_reward or SteeringRewardConfig()
            self.observation_space = Box(low=-1.0, high=1.0, shape=(STEERING_STATE_DIM,), dtype=np.float32)
            self.action_space = Box(low=np.array([0.0, self._u_min], dtype=np.float32),
                                    high=np.array([1.0, self._u_max], dtype=np.float32), dtype=np.float32)

        # CARLA handles (established once, kept alive across episodes) -------
        self._client = None
        self._world  = None

        # Episode actors (re-spawned each reset) ----------------------------
        self._vehicle          = None
        self._lidar            = None
        self._lidar_queue      = None
        self._collision_sensor = None
        self._walker           = None
        self._collision_flag   = {"hit": False, "other_actor": None}

        # Per-tick state (initialized in _setup_episode) --------------------
        self._speed_state      = {}
        self._crossing_state   = None
        self._walker_script    = None
        self._tick             = 0

        # LiDAR frame cache
        self._last_lidar_frame = None
        self._last_lidar_tick  = -1

        # Telemetry from previous tick (needed for reward computation)
        self._prev_telemetry   = None
        self._prev_brake_cmd   = 0.0

        # Distance / trigger tracking
        self._dist_m           = 0.0
        self._prev_loc         = None
        self._triggered        = False
        self._trigger_time_s   = None
        self._crossing_motion_started = False
        self._walker_start_loc = None
        self._walker_end_loc   = None

        # Jerk computation state (low-pass filtered, same as test3)
        self._speed_lp         = 0.0
        self._prev_speed_mps   = 0.0
        self._prev_accel_mps2  = 0.0
        self._jerk             = 0.0

        # Full-stop tracking (for stop bonus reward)
        self._full_stop_achieved = False
        self._hazard_engaged_once = False

        # Last raw telemetry dict from lane_follow_step (for inspection scripts)
        self._last_telemetry: dict = {}

        # Connect to CARLA once at construction time ------------------------
        self._connect()

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Start a new episode.
        Destroys actors from the previous episode, spawns fresh ones, and
        returns the initial observation.

        Returns:
            obs  (np.ndarray, shape=(6,)): initial state vector
            info (dict): empty — required by Gymnasium API
        """
        super().reset(seed=seed)

        # Refresh config from sampler if one was provided
        if self._config_fn is not None:
            self.cfg = self._config_fn()

        self._cleanup_actors()
        self._setup_episode()

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """
        Advance the simulation by one tick (0.02s at 50 Hz).

        Args:
            action: np.ndarray shape (1,) or scalar — brake force in [0, 1].
                    Values outside range are clipped.

        Returns:
            obs        (np.ndarray, shape=(6,)): new state after this tick
            reward     (float): per-tick reward from compute_reward()
            terminated (bool): True if collision or full stop achieved
            truncated  (bool): True if max_ticks reached (timeout)
            info       (dict): reward breakdown + sim diagnostics
        """
        # --- Unpack action -------------------------------------------------
        if self._steer_commit is None:
            brake_override = float(np.clip(np.asarray(action).flat[0], 0.0, 1.0))
        else:
            brake_override, u_action = split_action(np.asarray(action).ravel(), u_min=self._u_min, u_max=self._u_max)

        # --- Advance simulation --------------------------------------------
        self._world.tick()
        self._tick += 1
        sim_time_s = self._tick * FIXED_DT

        # --- Distance tracking ---------------------------------------------
        loc_now = self._vehicle.get_transform().location
        if self._prev_loc is not None:
            self._dist_m += loc_now.distance(self._prev_loc)
        self._prev_loc = loc_now

        ego_speed_mps = get_speed_mps(self._vehicle)

        # --- Jerk (low-pass filtered, same math as test3) ------------------
        self._speed_lp   = JERK_LP_ALPHA * ego_speed_mps + (1.0 - JERK_LP_ALPHA) * self._speed_lp
        _accel           = (self._speed_lp - self._prev_speed_mps) / FIXED_DT
        self._jerk       = abs((_accel - self._prev_accel_mps2) / FIXED_DT)
        self._prev_speed_mps  = self._speed_lp
        self._prev_accel_mps2 = _accel

        # --- Pedestrian trigger logic (mirrors test3) ----------------------
        cfg = self.cfg
        if cfg.trigger_ttc_s is not None:
            lead_time_s = cfg.trigger_ttc_s
        else:
            cross_dist_m = self._walker_start_loc.distance(self._walker_end_loc)
            lead_time_s  = cfg.walker_startup_s + (cross_dist_m / max(cfg.walker_speed_mps, 0.1))

        lead_dist_m    = ego_speed_mps * lead_time_s
        trigger_dist_m = max(0.0, cfg.encounter_distance_m - lead_dist_m)

        if (not self._triggered) and (self._dist_m >= trigger_dist_m) and (sim_time_s >= cfg.trigger_delay_s):
            self._triggered      = True
            self._trigger_time_s = sim_time_s
            trigger_scripted_crossing(self._world, self._crossing_state, debug_life_s=WALKER_DEBUG_LIFE_S)

        if self._triggered and not self._crossing_motion_started:
            if self._trigger_time_s is not None and (sim_time_s - self._trigger_time_s) >= cfg.walker_post_trigger_delay_s:
                self._crossing_motion_started = True

        if self._crossing_motion_started and self._walker_script is not None:
            self._step_walker_script(sim_time_s - self._trigger_time_s)
        elif self._crossing_motion_started:
            update_scripted_crossing(self._walker, self._crossing_state, fixed_dt=FIXED_DT)

        # Refresh debug markers each tick so they expire the moment this episode
        # ends. life_time=WALKER_DEBUG_LIFE_S (0.05s) means they vanish within
        # ~2 ticks if not refreshed — no stale markers from previous episodes.
        if self._triggered and self._crossing_state is not None:
            _s = self._crossing_state["start_loc"]
            _e = self._crossing_state["end_loc"]
            self._world.debug.draw_point(
                carla.Location(_s.x, _s.y, _s.z + 0.7),
                size=0.18, color=carla.Color(255, 0, 0), life_time=WALKER_DEBUG_LIFE_S)
            self._world.debug.draw_point(
                carla.Location(_e.x, _e.y, _e.z + 0.7),
                size=0.18, color=carla.Color(0, 255, 0), life_time=WALKER_DEBUG_LIFE_S)
            self._world.debug.draw_line(
                carla.Location(_s.x, _s.y, _s.z + 0.7),
                carla.Location(_e.x, _e.y, _e.z + 0.7),
                thickness=0.06, color=carla.Color(255, 255, 0), life_time=WALKER_DEBUG_LIFE_S)

        # --- LiDAR frame ---------------------------------------------------
        lidar_frame, self._last_lidar_frame, self._last_lidar_tick = get_latest_lidar_frame(
            self._lidar_queue,
            last_lidar_frame=self._last_lidar_frame,
            last_lidar_tick=self._last_lidar_tick,
            t=self._tick,
        )

        # --- Steering mode: route target + scripted hazard start ------------
        steer_kwargs = {}
        if self._steer_commit is not None:
            passage_now, steer_obs = self._steering_measurements(ego_speed_mps)
            target_m = self._commitment.target_offset_m(u_action, passage_now, triggered=self._triggered)
            self._hazard_on = self._triggered
            if self._merge_back is not None and self._triggered:
                ped = self._walker.get_location()
                _, ped_lateral = frame_coordinates_m(self._frame_origin, self._frame_right, (ped.x, ped.y))
                def clear_to(aim_m):  # the strip from where the car is to aim_m
                    return path_clear_of_hazard(
                        path_lo_m=steer_obs["ego_lateral_m"], path_hi_m=aim_m, hazard_lateral_m=ped_lateral,
                        hazard_lateral_speed_mps=steer_obs["pedestrian_lateral_speed_mps"],
                        hazard_radius_m=DEFAULT_PEDESTRIAN_RADIUS_M)
                target_m, self._hazard_on = self._merge_back.update(
                    sim_time_s - self._trigger_time_s, hazard_behind=self._hazard_behind_ego(), target_m=target_m,
                    path_clear=clear_to(self._merge_back.aim_m(target_m)), lane_clear=clear_to(0.0),
                    ego_stopped=ego_speed_mps <= DRIVE_ON_SPEED_MPS, ego_lateral_m=steer_obs["ego_lateral_m"])
                if self._merge_back.state == "holding_after_contact":
                    brake_override = 1.0  # after a contact: stop and hold
                elif not self._hazard_on and self._merge_back_hazard_was_on:
                    # Merge-back took over this tick: release the brake NOW. Without this, lane_follow
                    # debounces the clear for 0.2 s and ramps the brake down at 1/s, which stopped a
                    # car still rolling past the hazard (Quentin's demo, clip 3, 2026-10-23).
                    self._speed_state["brake_prev"] = 0.0
                    self._speed_state["hazard_clear_ticks"] = 10 ** 6  # skip the debounce
                self._merge_back_hazard_was_on = self._hazard_on
            steer_kwargs = dict(lateral_offset_m=target_m, scripted_hazard_active=self._hazard_on)
            if not self._hazard_on:
                brake_override = None  # before onset / after merge-back starts the policy has no effect

        # --- Controller step with RL brake override ------------------------
        telemetry = lane_follow_step(
            self._world, self._vehicle,
            lookahead_m=LOOKAHEAD_M,
            steer_gain=STEER_GAIN,
            target_speed_mps=self._target_speed_mps,
            fixed_dt=FIXED_DT,
            speed_state=self._speed_state,
            lidar_frame=lidar_frame,
            FIXED_DT=FIXED_DT,
            base_distance_m=BRAKE_BASE_DISTANCE_M,
            headway_seconds=cfg.brake_headway_s,
            panic_distance_m=PANIC_DISTANCE_M,
            ramp_up_per_s=cfg.braking_ramp_up_per_s,
            ramp_down_per_s=RAMP_DOWN_PER_S,
            brake_profile="proportional_ramp",   # ignored — rl_brake_override takes over
            rl_brake_override=brake_override,
            **steer_kwargs,
        )

        # Store raw telemetry for external inspection (inspect_brake_curves.py etc.)
        self._last_telemetry = telemetry or {}

        # --- Extract values from telemetry ---------------------------------
        d_min_ahead      = telemetry["d_min_ahead_m"]    if telemetry else None
        hazard_active    = bool(telemetry["hazard_brake_cmd"]) if telemetry else False
        trigger_dist_now = telemetry["trigger_distance_m"] if telemetry else 20.0
        brake_cmd_now    = telemetry["brake_cmd"]          if telemetry else 0.0
        drive_mode       = telemetry["drive_mode"]         if telemetry else "CRUISE"

        # --- TTC and pedestrian distance ------------------------------------
        if self._walker is not None:
            ped_dist = self._vehicle.get_location().distance(self._walker.get_location())
        else:
            ped_dist = None

        if hazard_active and ego_speed_mps > 0.1 and ped_dist is not None:
            ttc_s = ped_dist / ego_speed_mps
        else:
            ttc_s = float("inf")

        # --- Hazard / stop tracking ----------------------------------------
        if drive_mode in ("HAZARD_BRAKE", "STOP_HOLD"):
            self._hazard_engaged_once = True

        full_stop_this_tick = False
        if self._hazard_engaged_once and ego_speed_mps < 0.3 and not self._full_stop_achieved:
            self._full_stop_achieved = True
            full_stop_this_tick      = True

        # --- Collision check (rl_collision_signal.py) -----------------------
        # Sensor: CARLA physics sensor. Missed all 22 pedestrian contacts in the
        # Week 4 test26 baseline, so it cannot be the only detector.
        sensor_hit = self._collision_flag["hit"]
        self._collision_flag["hit"] = False

        # Legacy fallback (near-cross only, pedestrian ahead) and oriented-
        # footprint clearance, both from the same ground-truth poses. Which of
        # them counts toward collision_this_tick depends on collision_signal.
        legacy_proximity = False
        ped_clearance_m  = None
        ped_xy           = (None, None)
        if self._walker is not None:
            _ego_tf   = self._vehicle.get_transform()
            _ego_fwd  = _ego_tf.get_forward_vector()
            _ped_loc  = self._walker.get_location()
            ped_xy    = (_ped_loc.x, _ped_loc.y)
            legacy_proximity = legacy_proximity_contact(
                triggered=self._triggered,
                walker_cross=self.cfg.walker_cross,
                to_ped_x=_ped_loc.x - _ego_tf.location.x,
                to_ped_y=_ped_loc.y - _ego_tf.location.y,
                ego_forward_x=_ego_fwd.x,
                ego_forward_y=_ego_fwd.y,
                ped_dist_m=_ego_tf.location.distance(_ped_loc),
                ego_speed_mps=ego_speed_mps,
            )
            ped_clearance_m = geometric_clearance_m(
                ego_x_m=_ego_tf.location.x, ego_y_m=_ego_tf.location.y,
                ego_yaw_deg=_ego_tf.rotation.yaw,
                pedestrian_x_m=_ped_loc.x, pedestrian_y_m=_ped_loc.y,
            )
        collision_signals = combine_collision_signals(
            self._collision_signal,
            sensor=sensor_hit,
            legacy_proximity=legacy_proximity,
            clearance_m=ped_clearance_m,
        )
        collision_this_tick = collision_signals.collision

        if self._steer_commit is not None:
            return self._finish_steering_step(
                sim_time_s=sim_time_s, ego_speed_mps=ego_speed_mps, d_min_ahead=d_min_ahead,
                trigger_dist_now=trigger_dist_now, hazard_active=hazard_active, ttc_s=ttc_s,
                brake_cmd_now=brake_cmd_now, drive_mode=drive_mode, collision_signals=collision_signals,
                ped_xy=ped_xy, passage_now=passage_now, steer_obs=steer_obs, target_m=target_m, u_action=u_action)

        # --- Reward --------------------------------------------------------
        reward, breakdown = compute_reward(
            collision_this_tick=collision_this_tick,
            ttc_s=ttc_s,
            jerk_mps3=self._jerk if self._triggered else 0.0,
            brake_cmd_this_tick=brake_cmd_now,
            brake_cmd_prev_tick=self._prev_brake_cmd,
            ego_speed_mps=ego_speed_mps,
            target_speed_mps=self._target_speed_mps,
            hazard_active=hazard_active,
            full_stop_achieved=full_stop_this_tick,
            fixed_dt=FIXED_DT,
            current_ped_dist_m=ped_dist,
        )
        self._prev_brake_cmd = brake_cmd_now

        # --- Termination ---------------------------------------------------
        # terminated = episode ended with a meaningful outcome
        # truncated  = hit the time limit with no definitive outcome

        # Ped reached destination — only terminate on far-cross.
        #
        # For far-cross, end_loc is on the far side of the road. When done fires,
        # the ped has fully cleared ego's lane → safe, episode over.
        #
        # For near-cross, end_loc is the lane CENTER (ped stops there in front
        # of the car). done firing means the ped just got into position — that
        # is NOT a safe outcome. Terminating there causes "impossible+full_stop"
        # misclassification because the episode ends before the collision lands.
        # Near-cross episodes must end via collision or full_stop_achieved only.
        ped_crossed = (
            self._crossing_state is not None
            and self._crossing_state.get("done", False)
            and self.cfg.walker_cross.lower() == "far"
        )

        # Post-trigger timeout: if ped got nudged off course it may never reach
        # end_loc (crossing_state["done"] never fires). Truncate after 12s.
        post_trigger_timeout = (
            self._triggered
            and self._trigger_time_s is not None
            and (sim_time_s - self._trigger_time_s) > POST_TRIGGER_TIMEOUT_S
        )

        terminated = collision_this_tick or self._full_stop_achieved or ped_crossed
        truncated  = (self._tick >= self._total_ticks) or post_trigger_timeout

        # --- Observation ---------------------------------------------------
        obs = self._build_obs(ego_speed_mps, d_min_ahead, trigger_dist_now, hazard_active, ttc_s)

        # --- Info dict (for TensorBoard / diagnostics) --------------------
        info = {
            "r_safety":     breakdown["r_safety"],
            "r_comfort":    breakdown["r_comfort"],
            "r_efficiency": breakdown["r_efficiency"],
            "collision":    collision_this_tick,
            "ttc_s":        ttc_s,
            "jerk_mps3":    self._jerk,
            "drive_mode":   drive_mode,
            "tick":         self._tick,
            "sim_time_s":   sim_time_s,
            # Phase 1 additions (read-only; never fed back into control):
            # every collision component, plus ground truth for trace recording.
            **collision_signals.to_info(),
            "ego_speed_mps":   ego_speed_mps,
            "pedestrian_x_m":  ped_xy[0],
            "pedestrian_y_m":  ped_xy[1],
            "triggered":       self._triggered,
            "trigger_time_s":  self._trigger_time_s,
            "full_stop_achieved": self._full_stop_achieved,
            "ped_crossed":     ped_crossed,
            "post_trigger_timeout": post_trigger_timeout,
        }

        return obs, float(reward), terminated, truncated, info

    # -----------------------------------------------------------------------
    # Steering mode (Phase 2, opt-in). Nothing below runs in the default mode.
    # -----------------------------------------------------------------------

    def _new_commitment(self):
        return SteeringCommitment(self._steer_commit, u_min=self._u_min, u_max=self._u_max,
                                  lock_delay_ticks=self._lock_delay_ticks)

    def _setup_steering(self, carla_map, encounter_loc):
        """Per-episode steering state. The road's width at the pedestrian's
        line is scanned once here; the passage is rebuilt from it every tick
        as the pedestrian's predicted position changes."""
        self._map = carla_map
        road, frame = passage_at_location(carla_map, encounter_loc,  # no obstacles: just the road extent
                                          same_direction_only=self._same_direction_passage)
        right = road.road_right_m
        if right is not None and self._road_right_limit_m is not None:
            right = min(right, self._road_right_limit_m)
        self._road_extent = (road.road_left_m, right)
        self._frame_origin, self._frame_right = frame["origin_xyz"][:2], frame["right_xy"]
        self._commitment = self._new_commitment()
        self._merge_back = MergeBack() if self._merge_back_enabled else None
        self._merge_back_hazard_was_on = True
        self._hazard_on = False
        self._min_clearance_m = None
        self._went_off_road = False
        self._prev_u_applied = 0.0

    def _steering_measurements(self, ego_speed_mps):
        """(passage, observation inputs) for this tick, all in the passage frame
        anchored on the route at the pedestrian's line.
        ponytail: one straight frame for the whole approach. Exact near the
        line, where the policy acts; on a curved approach the far-away ego
        lateral and heading error are approximate. Use route-relative values
        if a curved location is ever trained on."""
        tf = self._vehicle.get_transform()
        ego_along, ego_lateral = frame_coordinates_m(self._frame_origin, self._frame_right,
                                                     (tf.location.x, tf.location.y))
        ped_loc, ped_vel = self._walker.get_location(), self._walker.get_velocity()
        _, ped_lateral = frame_coordinates_m(self._frame_origin, self._frame_right, (ped_loc.x, ped_loc.y))
        ped_lateral_speed = ped_vel.x * self._frame_right[0] + ped_vel.y * self._frame_right[1]
        distance = -ego_along
        predicted = predicted_lateral_at_arrival(ped_lateral, ped_lateral_speed, distance, ego_speed_mps)
        passage = passage_from_extent(*self._road_extent, [Obstacle(predicted)])
        return (None if passage.road_left_m is None else passage), dict(
            pedestrian_lateral_m=predicted, pedestrian_lateral_speed_mps=ped_lateral_speed,
            distance_to_pedestrian_line_m=distance, ego_lateral_m=ego_lateral,
            heading_error_rad=heading_error_rad(tf.rotation.yaw, self._frame_right))

    @property
    def merge_back_summary(self):
        """This episode's merge-back states and times (merge_back.py), or None when off."""
        return None if getattr(self, "_merge_back", None) is None else self._merge_back.summary()

    def _hazard_behind_ego(self):
        tf, ped = self._vehicle.get_transform(), self._walker.get_location()
        return hazard_behind_ego(ego_x_m=tf.location.x, ego_y_m=tf.location.y, ego_yaw_deg=tf.rotation.yaw,
                                 hazard_x_m=ped.x, hazard_y_m=ped.y, hazard_half_length_m=DEFAULT_PEDESTRIAN_RADIUS_M)

    def _steering_obs(self, base_obs, passage_now, steer_obs):
        return np.array(build_steering_observation(
            [float(v) for v in base_obs], passage=self._commitment.observed_passage(passage_now),
            previous_u=self._commitment.u_applied, **steer_obs), dtype=np.float32)

    def _finish_steering_step(self, *, sim_time_s, ego_speed_mps, d_min_ahead, trigger_dist_now, hazard_active,
                              ttc_s, brake_cmd_now, drive_mode, collision_signals, ped_xy, passage_now, steer_obs,
                              target_m, u_action):
        """Reward, termination, observation and info for one steering-mode tick."""
        contact_any = collision_signals.geometric_contact
        if contact_any and self._merge_back is not None:
            self._merge_back.note_contact()
        v = self._walker.get_velocity()
        pedestrian_caused = contact_any and contact_is_pedestrians(
            ego_speed_mps=ego_speed_mps, pedestrian_speed_mps=math.hypot(v.x, v.y))
        contact = contact_any and not (pedestrian_caused and self._steering_reward.pedestrian_contact_free)
        tf = self._vehicle.get_transform()
        on_road, _ = footprint_drivable(self._map, tf.location.x, tf.location.y, tf.rotation.yaw, tf.location.z)
        off_road = self._triggered and on_road is False
        self._went_off_road = self._went_off_road or off_road
        clearance = collision_signals.clearance_m
        if self._triggered and clearance is not None and (
                self._min_clearance_m is None or clearance < self._min_clearance_m):
            self._min_clearance_m = clearance
        window_ended = (self._triggered and not contact
                        and sim_time_s >= self._trigger_time_s + self._steering_reward.window_s - 1e-9)
        lateral_accel = ego_speed_mps * math.radians(self._vehicle.get_angular_velocity().z)

        reward, breakdown = compute_steering_reward(
            contact_this_tick=contact, ego_speed_mps=ego_speed_mps, off_road_this_tick=off_road,
            window_ended=window_ended, min_clearance_m=self._min_clearance_m, went_off_road=self._went_off_road,
            jerk_mps3=self._jerk, brake_cmd_this_tick=brake_cmd_now, brake_cmd_prev_tick=self._prev_brake_cmd,
            lateral_accel_mps2=lateral_accel, hazard_active=self._hazard_on, fixed_dt=FIXED_DT,
            u_this_tick=self._commitment.u_applied, u_prev_tick=self._prev_u_applied,
            config=self._steering_reward,
        )
        self._prev_brake_cmd = brake_cmd_now
        self._prev_u_applied = self._commitment.u_applied

        terminated = contact
        truncated = window_ended or (self._tick >= self._total_ticks)
        obs = self._steering_obs(
            self._build_obs(ego_speed_mps, d_min_ahead, trigger_dist_now, hazard_active, ttc_s), passage_now, steer_obs)
        shown = self._commitment.observed_passage(passage_now)
        info = {
            "r_safety": breakdown["r_safety"], "r_comfort": breakdown["r_comfort"], "r_efficiency": 0.0,
            "r_contact": breakdown["r_contact"], "r_off_road": breakdown["r_off_road"],
            "r_clearance": breakdown["r_clearance"],
            "ttc_s": ttc_s, "jerk_mps3": self._jerk, "drive_mode": drive_mode,
            "tick": self._tick, "sim_time_s": sim_time_s,
            **collision_signals.to_info(),
            "collision": contact,  # steering mode: the oriented-footprint check decides (the car's contacts)
            "contact_any": contact_any, "contact_pedestrian_caused": pedestrian_caused,
            "ego_speed_mps": ego_speed_mps, "pedestrian_x_m": ped_xy[0], "pedestrian_y_m": ped_xy[1],
            "triggered": self._triggered, "trigger_time_s": self._trigger_time_s,
            "full_stop_achieved": self._full_stop_achieved, "ped_crossed": False, "post_trigger_timeout": False,
            "steer_commit": self._steer_commit, "u_action": u_action, "u_applied": self._commitment.u_applied,
            "route_locked": self._commitment.locked, "requested_lateral_offset_m": target_m,
            "passage_left_edge_m": None if shown is None else shown.left_edge_m,
            "passage_right_edge_m": None if shown is None else shown.right_edge_m,
            "missing_passage_ticks": self._commitment.missing_passage_ticks,
            "off_road": off_road, "went_off_road": self._went_off_road, "footprint_on_road": on_road,
            "min_clearance_m": self._min_clearance_m, "lateral_accel_mps2": lateral_accel,
            "window_ended": window_ended, **steer_obs,
            "walker_phase": None if self._walker_script is None else self._walker_script.index,
            "merge_back_state": None if self._merge_back is None else self._merge_back.state,
            "merge_back_return_start_s": None if self._merge_back is None else self._merge_back.return_start_s,
            "merge_back_drive_on_start_s": None if self._merge_back is None else self._merge_back.drive_on_start_s,
        }
        return obs, float(reward), terminated, truncated, info

    @property
    def walker_transitions(self):
        """The scripted walker's phase changes this episode (walker_script.py), or None."""
        return None if self._walker_script is None else list(self._walker_script.transitions)

    def _step_walker_script(self, t_since_onset_s):
        (ox, oy), (rx, ry) = self._walker_line
        loc = self._walker.get_location()
        direction, speed = self._walker_script.step(t_since_onset_s, (loc.x - ox) * rx + (loc.y - oy) * ry)
        self._walker.apply_control(carla.WalkerControl(  # type: ignore[attr-defined]
            direction=carla.Vector3D(x=rx * direction, y=ry * direction, z=0.0), speed=speed, jump=False))

    def close(self):
        """Tear down all CARLA actors and restore async mode."""
        self._cleanup_actors()
        if self._world is not None:
            try:
                restore_async_mode(self._world)
            except Exception:
                pass
        print("[CarlaAEBEnv] closed.")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _connect(self):
        """Connect to CARLA server and load Town04_Opt. Called once at __init__."""
        print("[CarlaAEBEnv] connecting to CARLA...")
        self._client, self._world = connect_and_load_world(
            host="localhost",
            port=2000,
            timeout_s=10.0,
            target_map=self._map_name,
        )
        enable_sync_mode(self._world, fixed_dt=FIXED_DT)
        print("[CarlaAEBEnv] connected. Sync mode enabled.")

    def _setup_episode(self):
        """
        Spawn all actors and initialize all per-episode state.
        Mirrors the setup section of run_scenario() in test3.
        """
        cfg = self.cfg
        self._target_speed_mps = cfg.target_mph * 0.44704
        self._total_ticks = int(cfg.sim_seconds / FIXED_DT)

        # Apply weather
        import carla as _carla
        preset = getattr(_carla.WeatherParameters, cfg.weather_preset, _carla.WeatherParameters.ClearNoon)
        if cfg.sun_altitude_deg is not None:
            preset.sun_altitude_angle = cfg.sun_altitude_deg
        if cfg.cloudiness is not None:
            preset.cloudiness = cfg.cloudiness
        self._world.set_weather(preset)

        # Spawn ego + sensors
        if self._route_builder is None:
            spawn_points, ego_tf, bp_lib = prepare_spawn_context(self._world, SPAWN_INDEX=EGO_SPAWN_INDEX)
            route_wps = None
        else:
            ego_tf, route_wps = self._route_builder(self._world)
            bp_lib = self._world.get_blueprint_library()
        self._vehicle = spawn_ego_vehicle(self._world, bp_lib=bp_lib, spawn_tf=ego_tf)

        if cfg.road_friction is not None:
            phys = self._vehicle.get_physics_control()
            for w in phys.wheels:
                w.tire_friction = cfg.road_friction
            phys.wheels = phys.wheels
            self._vehicle.apply_physics_control(phys)

        self._lidar, self._lidar_queue = attach_lidar_sensor(self._world, bp_lib, self._vehicle)
        self._collision_sensor, self._collision_flag = self._attach_collision_sensor(bp_lib)

        # Build route
        carla_map = self._world.get_map()
        if route_wps is None:
            end_loc = spawn_points[END_MARKER_SPAWN_INDEX].location
            grp = GlobalRoutePlanner(carla_map, 2.0)
            route_wps = [wp for (wp, _) in grp.trace_route(ego_tf.location, end_loc)]
        self._route_wps = route_wps
        route_points_world = [wp.transform.location for wp in route_wps]

        # Controller state dict (mirrors test3's speed_state)
        self._speed_state = {
            "i_term": 0.0,
            "kp": 0.50,
            "ki": 0.10,
            "brake_prev": 0.0,
            "max_cone_yaw_deg": MAX_CONE_YAW_DEG,
            "cone_half_angle_deg": CONE_HALF_ANGLE_DEG,
            "cone_draw_length_m": CONE_DRAW_LENGTH_M,
            "drive_mode": "CRUISE",
            "hazard_clear_ticks": 0,
            "cone_max_lateral_m": CONE_MAX_LATERAL_M,
            "throttle_prev": 0.0,
            "brake_prev_cruise": 0.0,
            "lidar_actor": self._lidar,
            "route_points_world": route_points_world,
            "noodle_enable": NOODLE_ENABLE,
            "noodle_step_m": NOODLE_STEP_M,
            "noodle_half_width_m": NOODLE_HALF_WIDTH_M,
            "noodle_max_dist_m": NOODLE_MAX_DIST_M,
            "noodle_x_min_m": NOODLE_X_MIN_M,
        }

        # Reset all per-tick state
        self._tick                    = 0
        self._dist_m                  = 0.0
        self._prev_loc                = None
        self._last_lidar_frame        = None
        self._last_lidar_tick         = -1
        self._prev_telemetry          = None
        self._speed_lp                = 0.0
        self._prev_speed_mps          = 0.0
        self._prev_accel_mps2         = 0.0
        self._jerk                    = 0.0

        self._setup_encounter()

        # Tick once to let CARLA settle the newly spawned actors
        self._world.tick()

    def next_encounter(self, cfg):
        """Circuit mode: remove the current hazard and set up the next one, at
        cfg.encounter_distance_m along the SAME route (distance driven since the
        episode started), without respawning the car."""
        destroy_scripted_walker(self._walker)
        self._walker = None
        self.cfg = cfg
        self._setup_encounter()

    def _setup_encounter(self):
        """The pedestrian and the per-encounter state for self.cfg."""
        cfg = self.cfg
        carla_map = self._world.get_map()

        # Place pedestrian
        encounter_wp = self._get_route_waypoint_at_distance(self._route_wps, cfg.encounter_distance_m)
        road_extent = None
        if cfg.walker_cross.lower() == "far_edge":  # walks across the whole road: scan its extent
            road, _ = passage_at_location(carla_map, encounter_wp.transform.location)
            road_extent = (road.road_left_m, road.road_right_m)
        self._walker_start_loc, self._walker_end_loc = self._compute_crossing_endpoints(
            encounter_wp, side=cfg.walker_side, cross=cfg.walker_cross, road_extent_m=road_extent
        )

        spawn_yaw_deg = math.degrees(math.atan2(
            self._walker_end_loc.y - self._walker_start_loc.y,
            self._walker_end_loc.x - self._walker_start_loc.x,
        ))
        walker_tf = carla.Transform(self._walker_start_loc, carla.Rotation(yaw=spawn_yaw_deg))
        self._walker = spawn_scripted_walker(self._world, start_tf=walker_tf)
        if self._walker is None:
            raise RuntimeError("[CarlaAEBEnv] Failed to spawn walker.")

        self._crossing_state = init_scripted_crossing_state(
            start_loc=self._walker_start_loc,
            end_loc=self._walker_end_loc,
            speed_mps=cfg.walker_speed_mps,
        )
        self._walker_script = None
        if cfg.walker_script is not None:  # Week 7: behavior that changes after onset
            start_lat, _ = crossing_lateral_offsets_m(side=cfg.walker_side, cross=cfg.walker_cross,
                                                      lane_width_m=LANE_WIDTH_M)
            self._walker_script = WalkerScript(cfg.walker_script, start_lateral_m=start_lat)
            center, right = encounter_wp.transform.location, encounter_wp.transform.get_right_vector()
            self._walker_line = ((center.x, center.y), (right.x, right.y))

        self._triggered               = False
        self._trigger_time_s          = None
        self._crossing_motion_started = False
        self._prev_brake_cmd          = 0.0
        self._full_stop_achieved      = False
        self._hazard_engaged_once     = False

        if self._steer_commit is not None:
            self._setup_steering(carla_map, encounter_wp.transform.location)

    def _get_obs(self):
        """Return the current observation as a float32 numpy array."""
        ego_speed_mps = get_speed_mps(self._vehicle) if self._vehicle else 0.0
        base_obs = self._build_obs(
            ego_speed_mps=ego_speed_mps,
            d_min_ahead=None,
            trigger_distance_m=20.0,
            hazard_active=False,
            ttc_s=float("inf"),
        )
        if self._steer_commit is not None:
            return self._steering_obs(base_obs, *self._steering_measurements(ego_speed_mps))
        return base_obs

    def _build_obs(self, ego_speed_mps, d_min_ahead, trigger_distance_m, hazard_active, ttc_s):
        """Build and return the normalized 6-float state vector."""
        state = build_state_vector(
            ego_speed_mps=ego_speed_mps,
            d_min_ahead=d_min_ahead,
            trigger_distance_m=trigger_distance_m,
            prev_brake_cmd=self._prev_brake_cmd,
            hazard_active=hazard_active,
            ttc=ttc_s,
        )
        return np.array(state, dtype=np.float32)

    def _cleanup_actors(self):
        """Destroy all episode actors. Safe to call even if actors are None."""
        if self._collision_sensor is not None:
            try:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            except Exception:
                pass
            self._collision_sensor = None

        destroy_scripted_walker(self._walker)
        self._walker = None

        if self._lidar is not None:
            try:
                self._lidar.destroy()
            except Exception:
                pass
            self._lidar = None
            self._lidar_queue = None

        if self._vehicle is not None:
            try:
                self._vehicle.destroy()
            except Exception:
                pass
            self._vehicle = None

    def _attach_collision_sensor(self, bp_lib):
        """Attach a collision sensor to the ego vehicle. Returns (sensor, flag_dict)."""
        collision_bp = bp_lib.find("sensor.other.collision")
        sensor = self._world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self._vehicle,
        )
        flag = {"hit": False, "other_actor": None}

        def _on_collision(event):
            flag["hit"] = True
            flag["other_actor"] = event.other_actor.type_id if event.other_actor else "unknown"

        sensor.listen(_on_collision)
        return sensor, flag

    @staticmethod
    def _get_route_waypoint_at_distance(route_wps, distance_m):
        """Return the waypoint approximately distance_m along the route polyline."""
        if not route_wps:
            return route_wps[0]
        if distance_m <= 0.0:
            return route_wps[0]
        traveled = 0.0
        prev_loc = route_wps[0].transform.location
        for i in range(1, len(route_wps)):
            curr_loc = route_wps[i].transform.location
            seg_len  = prev_loc.distance(curr_loc)
            if traveled + seg_len >= distance_m:
                return route_wps[i]
            traveled += seg_len
            prev_loc  = curr_loc
        return route_wps[-1]

    @staticmethod
    def _compute_crossing_endpoints(encounter_wp, *, side, cross, road_extent_m=None):
        """Build walker start/end locations relative to an encounter waypoint."""
        lane_center = encounter_wp.transform.location
        right       = encounter_wp.transform.get_right_vector()
        # Same rule as test3, including "stationary" (stands at lane center).
        # Unknown side/cross values now raise instead of silently meaning
        # "right"/"far"; near/far placement is unchanged.
        start_lat, end_lat = crossing_lateral_offsets_m(side=side, cross=cross, lane_width_m=LANE_WIDTH_M,
                                                        road_extent_m=road_extent_m)

        start_loc = carla.Location(
            x=lane_center.x + right.x * start_lat,
            y=lane_center.y + right.y * start_lat,
            z=lane_center.z + 0.40,
        )
        end_loc = carla.Location(
            x=lane_center.x + right.x * end_lat,
            y=lane_center.y + right.y * end_lat,
            z=lane_center.z + 0.40,
        )
        return start_loc, end_loc
