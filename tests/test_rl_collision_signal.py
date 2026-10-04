import json
import math
import os
import sys
import unittest
from itertools import product
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pedestrian_contact import EGO_HALF_LENGTH_M, EGO_HALF_WIDTH_M, DEFAULT_PEDESTRIAN_RADIUS_M  # noqa: E402
from rl_collision_signal import (  # noqa: E402
    COLLISION_SIGNAL_GEOMETRIC,
    COLLISION_SIGNAL_LEGACY,
    combine_collision_signals,
    geometric_clearance_m,
    is_geometric_contact,
    legacy_proximity_contact,
    validate_collision_signal_mode,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "test26_contact_cases.json"
RUNS_ROOT = SRC_DIR / "runs" / "steer_brake_baseline"


def _load_fixture():
    with open(FIXTURE) as f:
        data = json.load(f)
    fields = data["tick_fields"]
    for case in data["cases"]:
        case["ticks"] = [dict(zip(fields, row)) for row in case["ticks"]]
    return data["cases"]


def _clearance(tick):
    return geometric_clearance_m(
        ego_x_m=tick["pos_x_m"], ego_y_m=tick["pos_y_m"], ego_yaw_deg=tick["yaw_deg"],
        pedestrian_x_m=tick["pedestrian_x_m"], pedestrian_y_m=tick["pedestrian_y_m"],
    )


def _legacy_kwargs(**overrides):
    # Ego at origin facing +x; pedestrian 2.5 m ahead, centred.
    kwargs = dict(triggered=True, walker_cross="near", to_ped_x=2.5, to_ped_y=0.0,
                  ego_forward_x=1.0, ego_forward_y=0.0, ped_dist_m=2.5, ego_speed_mps=3.0)
    kwargs.update(overrides)
    return kwargs


class ModeTests(unittest.TestCase):
    def test_known_modes_accepted(self):
        self.assertEqual(validate_collision_signal_mode("legacy"), "legacy")
        self.assertEqual(validate_collision_signal_mode("geometric"), "geometric")

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            validate_collision_signal_mode("oriented")
        with self.assertRaises(ValueError):
            combine_collision_signals("Legacy", sensor=False, legacy_proximity=False, clearance_m=None)


class CombineTests(unittest.TestCase):
    CLEARANCES = (None, 1.0, 0.0, -0.2)

    def test_legacy_ignores_geometric_contact(self):
        for sensor, prox, clr in product((False, True), (False, True), self.CLEARANCES):
            s = combine_collision_signals(COLLISION_SIGNAL_LEGACY, sensor=sensor, legacy_proximity=prox, clearance_m=clr)
            self.assertEqual(s.collision, sensor or prox)

    def test_geometric_is_superset_of_legacy(self):
        for sensor, prox, clr in product((False, True), (False, True), self.CLEARANCES):
            legacy = combine_collision_signals(COLLISION_SIGNAL_LEGACY, sensor=sensor, legacy_proximity=prox, clearance_m=clr)
            geo = combine_collision_signals(COLLISION_SIGNAL_GEOMETRIC, sensor=sensor, legacy_proximity=prox, clearance_m=clr)
            if legacy.collision:
                self.assertTrue(geo.collision)
            self.assertEqual(geo.collision, sensor or prox or (clr is not None and clr <= 0.0))

    def test_components_reported_in_both_modes(self):
        for mode in (COLLISION_SIGNAL_LEGACY, COLLISION_SIGNAL_GEOMETRIC):
            s = combine_collision_signals(mode, sensor=False, legacy_proximity=False, clearance_m=-0.1)
            self.assertTrue(s.geometric_contact)
            self.assertAlmostEqual(s.clearance_m, -0.1)
            info = s.to_info()
            self.assertEqual(info["collision_mode"], mode)
            self.assertIn("collision_sensor", info)
            self.assertIn("collision_legacy_proximity", info)
            self.assertIn("collision_geometric_contact", info)
            self.assertIn("collision", info)

    def test_contact_boundary_is_inclusive(self):
        self.assertTrue(is_geometric_contact(0.0))
        self.assertFalse(is_geometric_contact(1e-9))
        self.assertFalse(is_geometric_contact(None))


class LegacyProximityTests(unittest.TestCase):
    """The original near-cross fallback, behaviour preserved."""

    def test_fires_for_near_cross_contact_ahead(self):
        self.assertTrue(legacy_proximity_contact(**_legacy_kwargs()))

    def test_only_near_cross_and_only_after_trigger(self):
        for cross in ("far", "stationary"):
            self.assertFalse(legacy_proximity_contact(**_legacy_kwargs(walker_cross=cross)))
        self.assertFalse(legacy_proximity_contact(**_legacy_kwargs(triggered=False)))

    def test_ignores_pedestrian_behind_or_beside(self):
        self.assertFalse(legacy_proximity_contact(**_legacy_kwargs(to_ped_x=-2.5)))
        self.assertFalse(legacy_proximity_contact(**_legacy_kwargs(to_ped_x=0.0, to_ped_y=1.3, ped_dist_m=1.3)))

    def test_thresholds(self):
        self.assertFalse(legacy_proximity_contact(**_legacy_kwargs(ped_dist_m=2.7)))
        self.assertFalse(legacy_proximity_contact(**_legacy_kwargs(ego_speed_mps=0.5)))
        self.assertFalse(legacy_proximity_contact(**_legacy_kwargs(to_ped_y=1.2)))
        self.assertTrue(legacy_proximity_contact(**_legacy_kwargs(to_ped_y=1.19)))


class SyntheticGeometryTests(unittest.TestCase):
    """Cases the legacy fallback cannot see: contact beside the car, and
    near-misses that must not count."""

    def _clr(self, ped_x, ped_y, yaw=0.0):
        return geometric_clearance_m(ego_x_m=0.0, ego_y_m=0.0, ego_yaw_deg=yaw, pedestrian_x_m=ped_x, pedestrian_y_m=ped_y)

    def test_pedestrian_touching_side_is_contact_but_not_legacy(self):
        side_y = EGO_HALF_WIDTH_M + DEFAULT_PEDESTRIAN_RADIUS_M - 0.05
        self.assertTrue(is_geometric_contact(self._clr(0.0, side_y)))
        self.assertFalse(legacy_proximity_contact(**_legacy_kwargs(to_ped_x=0.0, to_ped_y=side_y, ped_dist_m=side_y)))

    def test_pedestrian_just_beside_is_clear(self):
        side_y = EGO_HALF_WIDTH_M + DEFAULT_PEDESTRIAN_RADIUS_M + 0.10
        self.assertAlmostEqual(self._clr(0.0, side_y), 0.10, places=9)

    def test_pedestrian_crossing_ahead_of_stopped_car_is_clear(self):
        front_x = EGO_HALF_LENGTH_M + DEFAULT_PEDESTRIAN_RADIUS_M + 0.2
        for y in (-3.0, -1.0, 0.0, 1.0, 3.0):
            self.assertFalse(is_geometric_contact(self._clr(front_x, y)))

    def test_heading_matters(self):
        # 2.0 m to the side: clear when facing +x, inside the body when facing +y.
        self.assertFalse(is_geometric_contact(self._clr(0.0, 2.0, yaw=0.0)))
        self.assertTrue(is_geometric_contact(self._clr(0.0, 2.0, yaw=90.0)))


class Test26EvidenceTests(unittest.TestCase):
    """Week 4 test26 traces (CARLA default physics): 22 recorded contacts,
    17 clear runs, CARLA's sensor silent in all 39."""

    @classmethod
    def setUpClass(cls):
        cls.cases = _load_fixture()

    def test_fixture_counts(self):
        self.assertEqual(len(self.cases), 39)
        self.assertEqual(sum(c["recorded_outcome"] == "contact" for c in self.cases), 22)
        self.assertFalse(any(c["carla_collision_sensor"] for c in self.cases))

    def test_contact_runs_detected_on_the_recorded_first_contact_tick(self):
        for case in (c for c in self.cases if c["recorded_outcome"] == "contact"):
            with self.subTest(case=case["case"], sweep=case["sweep"]):
                expected_t = case["onset_time_s"] + case["recorded_first_contact_time_s"]
                hits = [t for t in case["ticks"] if is_geometric_contact(_clearance(t))]
                self.assertTrue(hits)
                self.assertAlmostEqual(hits[0]["sim_time_s"], expected_t, places=6)
                self.assertAlmostEqual(case["trace_first_contact_time_s"], expected_t, places=6)
                before = [t for t in case["ticks"] if t["sim_time_s"] < expected_t - 1e-6]
                self.assertTrue(before)
                self.assertFalse(any(is_geometric_contact(_clearance(t)) for t in before))

    def test_no_false_contact_in_clear_runs(self):
        clear = [c for c in self.cases if c["recorded_outcome"] in ("stopped_clear", "passed_clear")]
        self.assertEqual(len(clear), 17)
        for case in clear:
            with self.subTest(case=case["case"], sweep=case["sweep"]):
                self.assertEqual(case["trace_geometric_contact_ticks"], 0)
                clearances = [_clearance(t) for t in case["ticks"]]
                self.assertFalse(any(is_geometric_contact(c) for c in clearances))
                # metrics.json was computed from live floats; trace.csv stores
                # them rounded, so agreement is to micrometres, not bits.
                self.assertAlmostEqual(min(clearances), case["recorded_min_clearance_m"], delta=1e-5)

    def test_closest_clear_pass_is_not_contact(self):
        closest = min((c for c in self.cases if c["recorded_outcome"] == "passed_clear"),
                      key=lambda c: c["recorded_min_clearance_m"])
        self.assertLess(closest["recorded_min_clearance_m"], 0.15)
        self.assertEqual(closest["trace_geometric_contact_ticks"], 0)

    def test_legacy_signal_misses_every_test26_contact_and_geometric_catches_all(self):
        # Stationary pedestrian: the near-cross fallback never runs, and the
        # sensor never fired, so the legacy env signal would count none.
        missed_by_legacy = caught_by_geometric = 0
        for case in (c for c in self.cases if c["recorded_outcome"] == "contact"):
            first = next(t for t in case["ticks"] if is_geometric_contact(_clearance(t)))
            kw = dict(sensor=case["carla_collision_sensor"], legacy_proximity=False, clearance_m=_clearance(first))
            missed_by_legacy += not combine_collision_signals(COLLISION_SIGNAL_LEGACY, **kw).collision
            caught_by_geometric += combine_collision_signals(COLLISION_SIGNAL_GEOMETRIC, **kw).collision
        self.assertEqual((missed_by_legacy, caught_by_geometric), (22, 22))


@unittest.skipUnless(RUNS_ROOT.is_dir(), "test26 evidence not present on this clone")
class FullTraceEvidenceTests(unittest.TestCase):
    """Whole-trace check against the original evidence when it is on disk."""

    def test_whole_traces_match_fixture(self):
        from trace_schema import read_trace_csv

        for case in _load_fixture():
            with self.subTest(case=case["case"], sweep=case["sweep"]):
                ticks = read_trace_csv(os.path.join(RUNS_ROOT, case["sweep"], *case["case"].split("/"), "trace.csv"))
                contact = [t for t in ticks if t.pedestrian_x_m is not None and is_geometric_contact(
                    geometric_clearance_m(ego_x_m=t.pos_x_m, ego_y_m=t.pos_y_m, ego_yaw_deg=t.yaw_deg,
                                          pedestrian_x_m=t.pedestrian_x_m, pedestrian_y_m=t.pedestrian_y_m))]
                self.assertEqual(len(contact), case["trace_geometric_contact_ticks"])
                if contact:
                    self.assertGreaterEqual(contact[0].sim_time_s, case["onset_time_s"] - 1e-6)
                    self.assertTrue(math.isclose(contact[0].sim_time_s, case["trace_first_contact_time_s"]))


if __name__ == "__main__":
    unittest.main()
