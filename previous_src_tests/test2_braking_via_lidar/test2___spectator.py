"""
test2___spectator.py

Spectator camera controller with 3 modes:

FOLLOW:
- Hard locks camera behind ego (fixed offset), yaw locked to ego.

FREE:
- Script stops touching spectator transform.
- CARLA's built-in spectator controls (WASD/QE + mouse) work normally.

GLIDE (NEW):
- Each tick, integrate spectator position using:
    spectator_pos += (ego_velocity + manual_velocity) * dt
- Rotation is NOT locked to ego. You can look anywhere.
- Manual WASD/QE movement is additive (in spectator-local axes).

Hotkeys (global, via pynput):
- F: toggle FOLLOW <-> FREE
- G: toggle GLIDE on/off (returns to previous mode)
- Hold SHIFT: speed boost for manual glide movement
"""

import carla
# pyright: reportMissingImports=false

from pynput import keyboard
import threading
import time
from typing import Optional, Set


def update_spectator_follow(spectator, vehicle):
    """Initial follow behavior"""
    if spectator is None or vehicle is None:
        return

    ego_tf = vehicle.get_transform()
    ego_loc = ego_tf.location

    cam_distance = 12.0
    cam_height = 5.0

    forward = ego_tf.get_forward_vector()
    cam_loc = carla.Location(
        x=ego_loc.x - forward.x * cam_distance,
        y=ego_loc.y - forward.y * cam_distance,
        z=ego_loc.z + cam_height
    )

    cam_rot = carla.Rotation(
        pitch=-15.0,
        yaw=ego_tf.rotation.yaw,
        roll=0.0
    )

    spectator.set_transform(carla.Transform(cam_loc, cam_rot))


class SpectatorController:
    MODE_FOLLOW = "FOLLOW"
    MODE_FREE = "FREE"
    MODE_GLIDE = "GLIDE"

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Actor,
        *,
        debug_draw: bool = True,
        glide_speed_mps: float = 12.0,     # manual movement speed in GLIDE
        glide_boost_mult: float = 2.5,     # SHIFT multiplier
        debounce_s: float = 0.25,
        fixed_dt_default: float = 0.02,    # fallback if world settings don't expose fixed dt
    ):
        self.world = world
        self.vehicle = vehicle
        self.spectator = world.get_spectator()

        self.debug_draw = debug_draw
        self.debounce_s = debounce_s

        # 3-mode state
        self.mode = self.MODE_FOLLOW
        self._mode_before_glide = self.MODE_FOLLOW

        # GLIDE integration state
        self._glide_loc: Optional[carla.Location] = None

        # Manual input state (for GLIDE)
        self._pressed: Set[str] = set()
        self._shift_down = False
        self.glide_speed_mps = float(glide_speed_mps)
        self.glide_boost_mult = float(glide_boost_mult)

        # Timing
        self._fixed_dt_default = float(fixed_dt_default)

        # Listener + locks
        self._listener = None
        self._lock = threading.Lock()
        self._last_toggle_time = 0.0

        print(
            "[spectator] pynput global listener starting. Hotkeys: F (FOLLOW/FREE), G (GLIDE toggle).",
            flush=True
        )

        self._listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self._listener.daemon = True
        self._listener.start()

        self._announce_mode()

    # -----------------------------
    # Helpers
    # -----------------------------
    def _get_dt(self, fixed_dt: Optional[float]) -> float:
        if fixed_dt is not None:
            return float(fixed_dt)

        # Try world fixed dt (sync mode usually sets this)
        try:
            settings = self.world.get_settings()
            if settings.fixed_delta_seconds is not None:
                return float(settings.fixed_delta_seconds)
        except Exception:
            pass

        return self._fixed_dt_default

    def _debounced(self) -> bool:
        now = time.time()
        if now - self._last_toggle_time < self.debounce_s:
            return True
        self._last_toggle_time = now
        return False

    def _announce_mode(self):
        print(f"[spectator] mode -> {self.mode}", flush=True)

    def _enter_glide(self):
        # Start GLIDE at current spectator location (no snap)
        tf = self.spectator.get_transform()
        self._glide_loc = carla.Location(tf.location.x, tf.location.y, tf.location.z)

    def _compute_manual_velocity_world(self, spectator_tf: carla.Transform) -> carla.Vector3D:
        """
        Manual movement is in spectator-local axes:
          W/S = forward/back (in the direction you're looking)
          A/D = left/right
          Q/E = down/up
        """
        # CARLA provides these vectors from transform
        fwd = spectator_tf.get_forward_vector()
        right = spectator_tf.get_right_vector()
        up = carla.Vector3D(0.0, 0.0, 1.0)

        x = 0.0
        y = 0.0
        z = 0.0

        if "w" in self._pressed:
            x += fwd.x; y += fwd.y; z += fwd.z
        if "s" in self._pressed:
            x -= fwd.x; y -= fwd.y; z -= fwd.z
        if "d" in self._pressed:
            x += right.x; y += right.y; z += right.z
        if "a" in self._pressed:
            x -= right.x; y -= right.y; z -= right.z
        if "e" in self._pressed:
            x += up.x; y += up.y; z += up.z
        if "q" in self._pressed:
            x -= up.x; y -= up.y; z -= up.z

        # Normalize so diagonals aren't faster
        mag = (x*x + y*y + z*z) ** 0.5
        if mag > 1e-6:
            x /= mag; y /= mag; z /= mag

        speed = self.glide_speed_mps * (self.glide_boost_mult if self._shift_down else 1.0)
        return carla.Vector3D(x * speed, y * speed, z * speed)

    # -----------------------------
    # Keyboard handlers
    # -----------------------------
    def _on_key_press(self, key):
        # Track shift for boost
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._shift_down = True
            return

        # Handle char keys
        try:
            ch = key.char.lower() if key.char is not None else None
        except Exception:
            ch = None

        if ch is None:
            return

        # Movement keys (for GLIDE)
        if ch in ("w", "a", "s", "d", "q", "e"):
            self._pressed.add(ch)
            return

        # Mode keys (debounced)
        if ch in ("f", "g"):
            with self._lock:
                if self._debounced():
                    return

                if ch == "f":
                    # FOLLOW <-> FREE (leave GLIDE if you're in it)
                    if self.mode == self.MODE_GLIDE:
                        self.mode = self.MODE_FREE
                    elif self.mode == self.MODE_FOLLOW:
                        self.mode = self.MODE_FREE
                    else:
                        self.mode = self.MODE_FOLLOW

                    if self.mode != self.MODE_GLIDE:
                        self._glide_loc = None

                    self._announce_mode()
                    return

                if ch == "g":
                    # Toggle GLIDE on/off (return to previous mode)
                    if self.mode == self.MODE_GLIDE:
                        self.mode = self._mode_before_glide
                        self._glide_loc = None
                    else:
                        self._mode_before_glide = self.mode
                        self.mode = self.MODE_GLIDE
                        self._enter_glide()

                    self._announce_mode()
                    return

    def _on_key_release(self, key):
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._shift_down = False
            return

        try:
            ch = key.char.lower() if key.char is not None else None
        except Exception:
            ch = None

        if ch in ("w", "a", "s", "d", "q", "e"):
            self._pressed.discard(ch)

    # -----------------------------
    # Per-tick update
    # -----------------------------
    def tick(self, fixed_dt: Optional[float] = None):
        """
        Call once per simulation tick.

        FOLLOW: hard set transform each tick (locked chase cam).
        FREE: do nothing (CARLA spectator controls fully active).
        GLIDE: integrate position from ego velocity + manual input velocity,
               but preserve current spectator rotation (mouse-look works).
        """
        if self.spectator is None or self.vehicle is None:
            return

        if self.mode == self.MODE_FOLLOW:
            update_spectator_follow(self.spectator, self.vehicle)

        elif self.mode == self.MODE_FREE:
            # Let CARLA do everything (WASD/QE + mouse)
            pass

        elif self.mode == self.MODE_GLIDE:
            dt = self._get_dt(fixed_dt)

            # Read current spectator rotation (so you can look around freely)
            spec_tf = self.spectator.get_transform()
            spec_rot = spec_tf.rotation

            # Initialize glide loc if needed (safety)
            if self._glide_loc is None:
                self._glide_loc = carla.Location(spec_tf.location.x, spec_tf.location.y, spec_tf.location.z)

            # Base motion: ride with ego velocity
            v = self.vehicle.get_velocity()  # m/s in world axes

            # Additive manual motion (in spectator-local axes -> world vector)
            manual_v = self._compute_manual_velocity_world(spec_tf)

            vx = v.x + manual_v.x
            vy = v.y + manual_v.y
            vz = v.z + manual_v.z

            self._glide_loc = carla.Location(
                x=self._glide_loc.x + vx * dt,
                y=self._glide_loc.y + vy * dt,
                z=self._glide_loc.z + vz * dt
            )

            self.spectator.set_transform(carla.Transform(self._glide_loc, spec_rot))

    def close(self):
        try:
            if self._listener is not None:
                self._listener.stop()
        except Exception:
            pass