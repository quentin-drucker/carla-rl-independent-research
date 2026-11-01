import random
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from circuit import (curve_speed_limit_mps, encounter_done, lane_change_offsets, place_hazards,  # noqa: E402
                     sample_hazard, straight_flags, training_spots, without_backtracking)


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


class LaneChangeOffsetsTests(unittest.TestCase):
    """The route planner's lane changes become smooth ramps (Quentin's lap watching, 2026-11-01)."""

    @staticmethod
    def _road(jump_at=50, step_m=3.5, n=100, junction_at=None):  # heading +x, 2 m spacing; CARLA +y = right
        xy = [(2.0 * k, step_m if k > jump_at else 0.0) for k in range(n)]
        junction = [junction_at is not None and abs(k - junction_at) <= 1 for k in range(n)]
        return xy, [0.0] * n, junction

    def test_jump_becomes_a_smooth_ramp_and_far_points_stay_put(self):
        xy, yaws, junction = self._road()
        off = lane_change_offsets(xy, yaws, junction, ramp_m=60.0)
        path = [y + o for (_, y), o in zip(xy, off)]
        self.assertTrue(all(abs(o) < 1e-9 for o in off[:35] + off[67:]))      # > 30 m from the jump
        self.assertAlmostEqual(path[50], 1.75, delta=0.2)                     # halfway across at the jump
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(path, path[1:])))   # never steps back
        self.assertLess(max(b - a for a, b in zip(path, path[1:])), 3.5 * 1.5 * 2.0 / 60.0 + 1e-6)  # max slope

    def test_left_jump_and_straight_road_and_junction(self):
        xy, yaws, junction = self._road(step_m=-3.5)
        self.assertAlmostEqual(min(lane_change_offsets(xy, yaws, junction)), -1.75, delta=0.2)
        self.assertEqual(set(lane_change_offsets(*self._road(step_m=0.0))), {0.0})
        self.assertEqual(set(lane_change_offsets(*self._road(junction_at=50))), {0.0})  # turns in junctions are not lane changes

    def test_ramp_stops_at_a_hazard_zone_and_the_zone_is_untouched(self):
        xy, yaws, junction = self._road()
        fixed = [55 <= k <= 70 for k in range(100)]  # a hazard zone starting 10 m after the jump
        off = lane_change_offsets(xy, yaws, junction, fixed=fixed, ramp_m=60.0)
        self.assertTrue(all(off[k] == 0.0 for k in range(55, 71)))
        path = [y + o for (_, y), o in zip(xy, off)]
        self.assertAlmostEqual(path[54], 3.5, delta=0.05)               # already across before the zone
        self.assertTrue(all(abs(o) < 1e-9 for o in off[:25]))           # slid back: still a 60 m ramp
        self.assertGreater(abs(off[30]), 0.01)
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(path, path[1:])))

    def test_lane_change_right_after_a_junction_starts_after_the_junction(self):
        xy, yaws, junction = self._road(junction_at=47)  # junction ends 2 points (4 m) before the jump
        off = lane_change_offsets(xy, yaws, junction, ramp_m=60.0)
        self.assertTrue(all(off[k] == 0.0 for k in range(46, 49)))      # nothing inside the junction
        path = [y + o for (_, y), o in zip(xy, off)]
        self.assertAlmostEqual(path[49], 0.0, delta=0.3)                # leaves the junction in the old lane
        self.assertAlmostEqual(path[78], 3.5, delta=0.05)               # across ~60 m later
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(path, path[1:])))


class WithoutBacktrackingTests(unittest.TestCase):
    def test_points_behind_or_repeated_are_interpolated_and_the_rest_unchanged(self):
        # Town05 route 1439-1451 m in miniature: a lane-change point 2.2 m BEHIND, repeated joint points
        xy = [(0.0, 0.0), (2.0, 0.0), (4.3, 0.0), (2.1, 0.1), (2.1, 0.1), (4.3, 0.1), (6.3, 0.1), (8.3, 0.1)]
        out = without_backtracking(xy, [0.0] * len(xy))
        self.assertEqual(out[:3] + out[6:], xy[:3] + xy[6:])
        self.assertTrue(all(b[0] > a[0] for a, b in zip(out, out[1:])))   # always moving forward
        self.assertAlmostEqual(out[4][0], 4.3 + 2.0 * 2 / 4, places=6)


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


class TrainingSpotsTests(unittest.TestCase):
    def test_spots_need_a_straight_approach_and_avoid_the_evaluation_hazards(self):
        s = [2.0 * k for k in range(1000)]                    # a 2 km loop
        ok = [not (300 <= k <= 320) for k in range(1000)]     # a curve at 600-640 m
        spots = training_spots(s, ok, approach_m=160.0, after_m=30.0, exclude_s=[1400.0], exclude_m=60.0)
        hazard_s = [s[i] for i in spots]
        self.assertTrue(all(h >= 180.0 for h in hazard_s))                        # room for the approach
        self.assertFalse(any(600.0 - 30.0 < h < 640.0 + 180.0 for h in hazard_s))  # no curve within approach/exit
        self.assertFalse(any(abs(h - 1400.0) < 60.0 for h in hazard_s))            # evaluation hazard held out
        self.assertTrue(any(1000.0 < h < 1300.0 for h in hazard_s))


if __name__ == "__main__":
    unittest.main()
