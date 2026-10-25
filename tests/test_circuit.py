import random
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from circuit import curve_speed_limit_mps, encounter_done, place_hazards, sample_hazard, straight_flags  # noqa: E402


class StraightFlagsTests(unittest.TestCase):
    def test_curves_and_junctions_and_their_surroundings_are_excluded(self):
        yaws = [0.0] * 30 + [float(k * 9) for k in range(10)] + [90.0] * 30   # a 90 deg turn in the middle
        junction = [False] * 70
        junction[60] = True
        ok = straight_flags(yaws, junction, window=5, max_turn_deg=10.0)
        self.assertTrue(all(ok[:24]))                  # straight, far from the turn
        self.assertFalse(any(ok[28:42]))               # the turn and its approach/exit
        self.assertFalse(any(ok[55:66]))               # around the junction
        self.assertTrue(ok[45])

    def test_yaw_wraps_around_180(self):
        ok = straight_flags([179.0, -179.0, 179.0, -179.0, 179.0], [False] * 5, window=1, max_turn_deg=10.0)
        self.assertTrue(all(ok))


class PlaceHazardsTests(unittest.TestCase):
    def test_hazards_are_on_valid_points_spaced_and_inside_the_route(self):
        s = [2.0 * k for k in range(2000)]                 # 4 km route, 2 m spacing
        ok = [not (800 <= k <= 900) for k in range(2000)]  # one 200 m stretch is invalid
        hz = place_hazards(s, ok, n=8, min_gap_m=300.0, start_m=150.0, end_margin_m=100.0, rng=random.Random(3))
        self.assertEqual(len(hz), 8)
        self.assertEqual(hz, sorted(hz))
        self.assertTrue(all(b - a >= 300.0 for a, b in zip(hz, hz[1:])))
        self.assertTrue(all(150.0 <= h <= s[-1] - 100.0 for h in hz))
        self.assertFalse(any(1600.0 <= h <= 1800.0 for h in hz))

    def test_asks_for_fewer_when_the_route_cannot_hold_them(self):
        s = [2.0 * k for k in range(500)]                  # 1 km
        hz = place_hazards(s, [True] * 500, n=10, min_gap_m=300.0, start_m=100.0, end_margin_m=100.0,
                           rng=random.Random(0))
        self.assertLessEqual(len(hz), 3)

    def test_same_seed_same_plan(self):
        s = [2.0 * k for k in range(2000)]
        a = place_hazards(s, [True] * 2000, n=5, min_gap_m=300.0, start_m=150.0, end_margin_m=100.0, rng=random.Random(9))
        b = place_hazards(s, [True] * 2000, n=5, min_gap_m=300.0, start_m=150.0, end_margin_m=100.0, rng=random.Random(9))
        self.assertEqual(a, b)


class SampleHazardTests(unittest.TestCase):
    def test_kinds_and_warning_times(self):
        rng = random.Random(1)
        hz = [sample_hazard(rng) for _ in range(300)]
        self.assertEqual({h["walker"] for h in hz}, {"right_edge", "left_edge", "stationary"})
        self.assertTrue(all(0.6 <= h["ttc_s"] <= 1.5 for h in hz))


class EncounterDoneTests(unittest.TestCase):
    def test_waits_for_the_window_and_the_return_then_times_out(self):
        self.assertFalse(encounter_done(5.0, "returned"))       # scoring window not over
        self.assertTrue(encounter_done(8.0, "returned"))
        self.assertFalse(encounter_done(9.0, "returning"))      # still merging back
        self.assertTrue(encounter_done(9.0, "holding_after_contact"))  # contact: nothing more to wait for
        self.assertTrue(encounter_done(20.0, "avoiding"))       # waiting beside a pedestrian: give up


class CurveSpeedTests(unittest.TestCase):
    """A 90 degree corner of radius ~12.7 m starting 100 m ahead (2 m spacing)."""
    S = [2.0 * k for k in range(200)]

    def test_straight_road_keeps_the_cruise_speed(self):
        self.assertEqual(curve_speed_limit_mps(self.S, [0.0] * 200, 0, v_max_mps=15.6), 15.6)

    def test_slows_before_and_in_a_corner_and_not_far_from_it(self):
        yaw = [0.0] * 50 + [9.0 * k for k in range(1, 11)] + [90.0] * 140   # 90 deg over 20 m: R ~ 12.7 m
        far = curve_speed_limit_mps(self.S, yaw, 0, v_max_mps=15.6)            # corner 100 m ahead: beyond lookahead
        near = curve_speed_limit_mps(self.S, yaw, 40, v_max_mps=15.6)          # 20 m before it
        inside = curve_speed_limit_mps(self.S, yaw, 52, v_max_mps=15.6)
        self.assertEqual(far, 15.6)
        self.assertLess(inside, near)
        self.assertLess(near, 15.6)
        self.assertAlmostEqual(inside, (3.0 * 12.7) ** 0.5, delta=0.8)          # sqrt(a_lat * R)


if __name__ == "__main__":
    unittest.main()
