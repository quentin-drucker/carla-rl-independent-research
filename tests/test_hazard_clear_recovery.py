import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from test5___scripted_pedestrian_steering import (  # noqa: E402
    HazardClearRecoveryController,
    build_evasive_offset_fn,
)

DT_S = 0.02


def _clear_info(*, pedestrian_behind_ego=False, original_corridor_clear=False):
    return {
        "pedestrian_behind_ego": pedestrian_behind_ego,
        "original_corridor_clear": original_corridor_clear,
        "drive_mode": "HAZARD_BRAKE",
    }


def _drive(controller, trigger_time_s, start_s, stop_s, clear_info_fn, dt=DT_S):
    """Call the controller every dt from start_s up to (not including) stop_s,
    mirroring the real scenario's one-call-per-tick usage. Returns the last
    offset returned.
    """
    offset = None
    n_ticks = int(round((stop_s - start_s) / dt))
    for i in range(n_ticks):
        sim_time_s = start_s + i * dt
        offset = controller(sim_time_s, True, trigger_time_s, clear_info_fn(sim_time_s))
    return offset


class HazardClearRecoveryControllerTests(unittest.TestCase):
    def test_returns_zero_before_trigger(self):
        controller = HazardClearRecoveryController(1.5)
        self.assertEqual(controller(0.0, False, None, _clear_info()), 0.0)
        self.assertEqual(controller.state, "IDLE")

    def test_shift_out_ramps_then_holds_while_hazard_active(self):
        controller = HazardClearRecoveryController(
            1.5, shift_duration_s=1.0, clear_confirm_s=0.5, max_hold_s=8.0
        )
        trigger_time_s = 2.0

        # Partway through the shift: strictly between 0 and peak.
        mid_shift = _drive(
            controller, trigger_time_s, trigger_time_s, trigger_time_s + 0.5,
            lambda t: _clear_info(),
        )
        self.assertGreater(mid_shift, 0.0)
        self.assertLess(mid_shift, 1.5)
        self.assertEqual(controller.state, "SHIFT_OUT")

        # Hazard never clears: must keep holding at peak, never time out early.
        held = _drive(
            controller, trigger_time_s, trigger_time_s + 0.5, trigger_time_s + 6.0,
            lambda t: _clear_info(),
        )
        self.assertAlmostEqual(held, 1.5, places=6)
        self.assertEqual(controller.state, "HOLD_AVOIDANCE")
        self.assertFalse(controller.recovered)
        self.assertFalse(controller.used_fallback_timeout)

    def test_single_tick_clear_glitch_does_not_trigger_recovery(self):
        controller = HazardClearRecoveryController(
            1.5, shift_duration_s=1.0, clear_confirm_s=0.5, max_hold_s=8.0
        )
        trigger_time_s = 0.0

        def clear_info_fn(t):
            # One isolated clear tick surrounded by hazard-active ticks.
            return _clear_info(original_corridor_clear=abs(t - 3.0) < 1e-9)

        _drive(controller, trigger_time_s, trigger_time_s, 5.0, clear_info_fn)
        self.assertEqual(controller.state, "HOLD_AVOIDANCE")
        self.assertFalse(controller.recovered)
        self.assertIsNone(controller._clear_since_s)

    def test_stable_clear_starts_recovery_and_returns_to_zero(self):
        controller = HazardClearRecoveryController(
            1.5, shift_duration_s=1.0, return_duration_s=2.0,
            clear_confirm_s=0.5, max_hold_s=8.0,
        )
        trigger_time_s = 0.0

        # Hazard active until t=3.0, then stably clear from t=3.0 onward.
        def clear_info_fn(t):
            return _clear_info(pedestrian_behind_ego=(t >= 3.0))

        final_offset = _drive(controller, trigger_time_s, trigger_time_s, 8.0, clear_info_fn)

        self.assertTrue(controller.recovered)
        self.assertEqual(controller.state, "RECOVERED")
        self.assertAlmostEqual(final_offset, 0.0, places=6)
        self.assertFalse(controller.used_fallback_timeout)
        self.assertIsNotNone(controller.time_hazard_clear_to_recover_start_s)
        # Recovery starts only after clear_confirm_s of stable clearance,
        # i.e. around t=3.5, not immediately at t=3.0.
        self.assertGreaterEqual(controller.time_hazard_clear_to_recover_start_s, 0.5 - 1e-6)

    def test_hazard_reappearing_during_return_re_arms_hold(self):
        controller = HazardClearRecoveryController(
            1.5, shift_duration_s=1.0, return_duration_s=2.0,
            clear_confirm_s=0.5, max_hold_s=8.0,
        )
        trigger_time_s = 0.0

        # Clear from t=2.0, long enough to enter RETURN, then hazard returns
        # partway through the return ramp (t=3.2) before it would complete.
        def clear_info_fn(t):
            return _clear_info(pedestrian_behind_ego=(2.0 <= t < 3.2))

        _drive(controller, trigger_time_s, trigger_time_s, 4.0, clear_info_fn)

        self.assertTrue(controller.hazard_reappeared_during_recovery)
        self.assertFalse(controller.recovered)
        self.assertEqual(controller.state, "HOLD_AVOIDANCE")

    def test_fallback_timeout_only_fires_after_max_hold(self):
        controller = HazardClearRecoveryController(
            1.5, shift_duration_s=1.0, return_duration_s=1.0,
            clear_confirm_s=0.5, max_hold_s=3.0,
        )
        trigger_time_s = 0.0

        # Hazard never clears at all.
        _drive(
            controller, trigger_time_s, trigger_time_s, trigger_time_s + 1.0 + 2.5,
            lambda t: _clear_info(),
        )
        self.assertFalse(controller.used_fallback_timeout)
        self.assertEqual(controller.state, "HOLD_AVOIDANCE")

        # Hazard still never clears, yet the fallback recovery must still
        # complete -- it must not re-arm HOLD_AVOIDANCE forever just because
        # hazard_clear_now stays False during the forced return.
        final_offset = _drive(
            controller, trigger_time_s, trigger_time_s + 1.0 + 2.5, trigger_time_s + 1.0 + 3.5 + 1.5,
            lambda t: _clear_info(),
        )
        self.assertTrue(controller.used_fallback_timeout)
        self.assertEqual(controller.state, "RECOVERED")
        self.assertTrue(controller.recovered)
        self.assertAlmostEqual(final_offset, 0.0, places=6)
        self.assertFalse(controller.hazard_reappeared_during_recovery)

    def test_build_evasive_offset_fn_returns_independent_controllers(self):
        a = build_evasive_offset_fn(1.0)
        b = build_evasive_offset_fn(1.0)
        self.assertIsInstance(a, HazardClearRecoveryController)
        self.assertIsNot(a, b)


if __name__ == "__main__":
    unittest.main()
