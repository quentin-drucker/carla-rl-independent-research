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
from scenario_config import ScenarioConfig
from walker_utils import (
    spawn_scripted_walker,
    init_scripted_crossing_state,
    trigger_scripted_crossing,
    update_scripted_crossing,
    destroy_scripted_walker,
)
from rl_reward_design import build_state_vector, compute_reward, STATE_DIM


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

WALKER_DEBUG_LIFE_S   = 12.0

# Jerk low-pass filter alpha (same value as test3)
JERK_LP_ALPHA         = 0.25


# ===========================================================================
# Gymnasium environment
# ===========================================================================

class CarlaAEBEnv(gymnasium.Env):
    """
    Gymnasium-compatible environment for training an RL braking controller
    in CARLA's pedestrian-intrusion scenario.

    Observation space: Box(5,) — normalized state vector (see rl_reward_design.py)
    Action space:      Box(1,) — brake force in [0.0, 1.0]

    The agent replaces the brake_profile logic in lane_follow_step().
    Everything else (steering, cruise control, ramp limiter, STOP_HOLD, etc.)
    is unchanged from the existing simulation.
    """

    metadata = {"render_modes": []}

    # -----------------------------------------------------------------------
    def __init__(self, cfg: ScenarioConfig = None):
        """
        Args:
            cfg: ScenarioConfig instance. Defaults to ScenarioConfig() if None.
                 Walker, ego speed, weather etc. are all taken from this config.
                 brake_profile is ignored (the RL agent controls braking instead).
        """
        super().__init__()

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
            obs  (np.ndarray, shape=(5,)): initial state vector
            info (dict): empty — required by Gymnasium API
        """
        super().reset(seed=seed)

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
            obs        (np.ndarray, shape=(5,)): new state after this tick
            reward     (float): per-tick reward from compute_reward()
            terminated (bool): True if collision or full stop achieved
            truncated  (bool): True if max_ticks reached (timeout)
            info       (dict): reward breakdown + sim diagnostics
        """
        # --- Unpack action -------------------------------------------------
        brake_override = float(np.clip(action, 0.0, 1.0))

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

        if self._crossing_motion_started:
            update_scripted_crossing(self._walker, self._crossing_state, fixed_dt=FIXED_DT)

        # --- LiDAR frame ---------------------------------------------------
        lidar_frame, self._last_lidar_frame, self._last_lidar_tick = get_latest_lidar_frame(
            self._lidar_queue,
            last_lidar_frame=self._last_lidar_frame,
            last_lidar_tick=self._last_lidar_tick,
            t=self._tick,
        )

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
        )

        # --- Extract values from telemetry ---------------------------------
        d_min_ahead      = telemetry["d_min_ahead_m"]    if telemetry else None
        hazard_active    = bool(telemetry["hazard_brake_cmd"]) if telemetry else False
        trigger_dist_now = telemetry["trigger_distance_m"] if telemetry else 20.0
        brake_cmd_now    = telemetry["brake_cmd"]          if telemetry else 0.0
        drive_mode       = telemetry["drive_mode"]         if telemetry else "CRUISE"

        # --- TTC -----------------------------------------------------------
        if hazard_active and ego_speed_mps > 0.1 and self._walker is not None:
            ped_dist = self._vehicle.get_location().distance(self._walker.get_location())
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

        # --- Collision check -----------------------------------------------
        collision_this_tick = self._collision_flag["hit"]
        # Reset flag so each tick only reports its own collision
        self._collision_flag["hit"] = False

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
        )
        self._prev_brake_cmd = brake_cmd_now

        # --- Termination ---------------------------------------------------
        # terminated = episode ended with a meaningful outcome
        # truncated  = hit the time limit with no definitive outcome
        terminated = collision_this_tick or self._full_stop_achieved
        truncated  = (self._tick >= self._total_ticks)

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
        }

        return obs, float(reward), terminated, truncated, info

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
            target_map=TARGET_MAP,
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
        spawn_points, ego_tf, bp_lib = prepare_spawn_context(self._world, SPAWN_INDEX=EGO_SPAWN_INDEX)
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
        end_loc = spawn_points[END_MARKER_SPAWN_INDEX].location
        carla_map = self._world.get_map()
        grp = GlobalRoutePlanner(carla_map, 2.0)
        route = grp.trace_route(ego_tf.location, end_loc)
        route_wps = [wp for (wp, _) in route]
        route_points_world = [wp.transform.location for wp in route_wps]

        # Place pedestrian
        encounter_wp = self._get_route_waypoint_at_distance(route_wps, cfg.encounter_distance_m)
        self._walker_start_loc, self._walker_end_loc = self._compute_crossing_endpoints(
            encounter_wp, side=cfg.walker_side, cross=cfg.walker_cross
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
        self._triggered               = False
        self._trigger_time_s          = None
        self._crossing_motion_started = False
        self._last_lidar_frame        = None
        self._last_lidar_tick         = -1
        self._prev_telemetry          = None
        self._prev_brake_cmd          = 0.0
        self._speed_lp                = 0.0
        self._prev_speed_mps          = 0.0
        self._prev_accel_mps2         = 0.0
        self._jerk                    = 0.0
        self._full_stop_achieved      = False
        self._hazard_engaged_once     = False

        # Tick once to let CARLA settle the newly spawned actors
        self._world.tick()

    def _get_obs(self):
        """Return the current observation as a float32 numpy array."""
        ego_speed_mps = get_speed_mps(self._vehicle) if self._vehicle else 0.0
        return self._build_obs(
            ego_speed_mps=ego_speed_mps,
            d_min_ahead=None,
            trigger_distance_m=20.0,
            hazard_active=False,
            ttc_s=float("inf"),
        )

    def _build_obs(self, ego_speed_mps, d_min_ahead, trigger_distance_m, hazard_active, ttc_s):
        """Build and return the normalized 5-float state vector."""
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
    def _compute_crossing_endpoints(encounter_wp, *, side, cross):
        """Build walker start/end locations relative to an encounter waypoint."""
        lane_center = encounter_wp.transform.location
        right       = encounter_wp.transform.get_right_vector()
        start_lat   = LANE_WIDTH_M * 0.85
        if side.lower() == "left":
            start_lat = -start_lat
        end_lat = 0.0 if cross.lower() == "near" else -start_lat

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
