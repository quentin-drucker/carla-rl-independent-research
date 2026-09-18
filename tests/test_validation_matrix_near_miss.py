import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from test7___steering_validation_matrix import (  # noqa: E402
    ValidationObserver,
    NEAR_MISS_TTC_S,
)


def _telemetry(*, ttc_s, brake_cmd=0.0, steer_cmd=0.0, offset_m=1.0):
    return {
        "signed_route_lateral_offset_m": offset_m,
        "lateral_offset_requested_m": offset_m,
        "d_min_original_path_m": None,
        "d_min_commanded_path_m": None,
        "d_min_transition_path_m": None,
        "d_min_left_candidate_m": None,
        "d_min_right_candidate_m": None,
        "commanded_path_drivability": None,
        "transition_path_drivability": None,
        "hazard_governing_source": "original",
        "ttc_s": ttc_s,
        "brake_cmd": brake_cmd,
        "steer_cmd": steer_cmd,
    }


class ValidationObserverNearMissTests(unittest.TestCase):
    def test_low_ttc_with_no_braking_is_flagged_dangerous(self):
        """Reproduces the shape of the 2026-09-18 finding: a close encounter
        (low TTC) where braking never engaged at all must be detectable
        automatically, not only by a human reading console output.
        """
        observer = ValidationObserver()
        observer(0.0, True, _telemetry(ttc_s=float("nan"), brake_cmd=0.0))
        observer(0.02, True, _telemetry(ttc_s=0.37, brake_cmd=0.0))
        observer(0.04, True, _telemetry(ttc_s=2.0, brake_cmd=0.0))

        self.assertFalse(observer.brake_ever_engaged)
        self.assertLess(observer.min_ttc_s_observed, NEAR_MISS_TTC_S)

    def test_low_ttc_with_braking_engaged_is_not_flagged(self):
        observer = ValidationObserver()
        observer(0.0, True, _telemetry(ttc_s=0.37, brake_cmd=0.8))

        self.assertTrue(observer.brake_ever_engaged)
        self.assertLess(observer.min_ttc_s_observed, NEAR_MISS_TTC_S)

    def test_high_ttc_with_no_braking_is_not_flagged(self):
        observer = ValidationObserver()
        observer(0.0, True, _telemetry(ttc_s=5.0, brake_cmd=0.0))

        self.assertFalse(observer.brake_ever_engaged)
        self.assertGreaterEqual(observer.min_ttc_s_observed, NEAR_MISS_TTC_S)

    def test_nan_ttc_ticks_are_ignored_not_treated_as_zero(self):
        observer = ValidationObserver()
        observer(0.0, True, _telemetry(ttc_s=float("nan")))
        self.assertEqual(observer.min_ttc_s_observed, float("inf"))


if __name__ == "__main__":
    unittest.main()
