import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from encounter_metrics import (  # noqa: E402
    OUTCOME_CONTACT,
    OUTCOME_INCOMPLETE,
    OUTCOME_NO_ONSET,
    OUTCOME_PASSED_CLEAR,
    OUTCOME_STOPPED_CLEAR,
    OUTCOME_UNRESOLVED,
    EncounterProtocol,
    compute_encounter_metrics,
)
from pedestrian_contact import EGO_HALF_LENGTH_M, EGO_HALF_WIDTH_M  # noqa: E402
from trace_schema import TraceTick  # noqa: E402

DT = 0.02
PED = (50.0, 0.0)
ONSET_S = 1.0


def _trace(n_ticks, *, x0=10.0, speed_fn, lateral_fn=lambda t: 0.0, requested_fn=None,
           accel_override=None, ped=PED):
    """Straight-road ego along +x, yaw 0. speed_fn(t) -> m/s; lateral_fn(t)
    -> y (also used as route_lateral_m). Accel is finite-differenced."""
    ticks, x, prev_v = [], x0, None
    for i in range(n_ticks):
        t = i * DT
        v = speed_fn(t)
        if i > 0:
            x += v * DT
        accel = None if prev_v is None else (v - prev_v) / DT
        if accel_override is not None:
            accel = accel_override(t, v, accel)
        y = lateral_fn(t)
        ticks.append(TraceTick(
            tick_index=i, sim_time_s=t, pos_x_m=x, pos_y_m=y, pos_z_m=0.0,
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, speed_mps=v,
            accel_mps2=accel, yaw_rate_dps=0.0,
            route_lateral_m=y,
            requested_lateral_offset_m=(requested_fn(t) if requested_fn else 0.0),
            pedestrian_x_m=ped[0] if ped else None, pedestrian_y_m=ped[1] if ped else None,
        ))
        prev_v = v
    return ticks


def _brake_speed(t, *, v0=15.0, decel=8.0):
    return v0 if t < ONSET_S else max(0.0, v0 - decel * (t - ONSET_S))


PROTOCOL = EncounterProtocol(horizon_s=4.0)
N_FULL = int((ONSET_S + PROTOCOL.horizon_s) / DT) + 1


class OutcomeTests(unittest.TestCase):
    def test_no_intervention_is_contact_with_impact_speed(self):
        m = compute_encounter_metrics(_trace(N_FULL, speed_fn=lambda t: 15.0),
                                      onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertEqual(m.outcome, OUTCOME_CONTACT)
        self.assertFalse(m.safe_success)
        self.assertAlmostEqual(m.contact_speed_mps, 15.0)
        # Front bumper reaches the pedestrian circle: x + half_length >= 50 - 0.3.
        # x(t) = 10 + 15 t  ->  t ~= (49.7 - 2.396 - 10) / 15 = 2.487 s abs.
        self.assertAlmostEqual(m.first_contact_time_s + ONSET_S, (49.7 - EGO_HALF_LENGTH_M - 10.0) / 15.0, delta=DT)

    def test_braking_to_a_stop_measures_from_shared_onset(self):
        m = compute_encounter_metrics(_trace(N_FULL, speed_fn=_brake_speed),
                                      onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertEqual(m.outcome, OUTCOME_STOPPED_CLEAR)
        self.assertTrue(m.safe_success)
        self.assertTrue(m.stopped)
        self.assertFalse(m.passed)
        # First tick below 0.3 m/s: (15 - 0.3) / 8 = 1.8375 s after onset.
        self.assertAlmostEqual(m.time_to_stop_s, (15.0 - 0.3) / 8.0, delta=DT)
        self.assertAlmostEqual(m.stopping_distance_m, (15.0 ** 2 - 0.3 ** 2) / (2 * 8.0), delta=0.4)
        self.assertAlmostEqual(m.speed_at_onset_mps, 15.0)
        self.assertGreater(m.min_clearance_m, 5.0)
        self.assertEqual(m.contact_ticks, 0)

    def test_swerve_clearance_is_measured_beside_the_car_not_only_ahead(self):
        # Regression for RunResult.min_ped_distance_m, which ignores the
        # pedestrian once it is no longer ahead. Ego shifts 3 m sideways
        # before reaching the pedestrian and keeps speed.
        lateral = lambda t: 3.0 * min(1.0, max(0.0, (t - ONSET_S) / 1.0))  # noqa: E731
        m = compute_encounter_metrics(_trace(N_FULL, speed_fn=lambda t: 15.0, lateral_fn=lateral),
                                      onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertEqual(m.outcome, OUTCOME_PASSED_CLEAR)
        self.assertTrue(m.safe_success)
        self.assertAlmostEqual(m.min_clearance_m, 3.0 - EGO_HALF_WIDTH_M - 0.3, delta=0.02)
        # Closest point is when the pedestrian is beside the ego (x ~= 50).
        x_at_min = 10.0 + 15.0 * (m.min_clearance_time_s + ONSET_S)
        self.assertLess(abs(x_at_min - PED[0]), EGO_HALF_LENGTH_M)
        self.assertAlmostEqual(m.speed_when_passing_mps, 15.0)
        self.assertAlmostEqual(m.max_abs_route_lateral_m, 3.0)
        self.assertTrue(m.departed_route)
        self.assertFalse(m.returned_to_route)

    def test_swerve_then_return_counts_physical_route_recovery(self):
        def lateral(t):
            if t < ONSET_S:
                return 0.0
            if t < ONSET_S + 1.0:
                return 3.0 * (t - ONSET_S)
            if t < ONSET_S + 2.0:
                return 3.0
            return max(0.0, 3.0 - 3.0 * (t - ONSET_S - 2.0))
        requested = lambda t: 0.0 if t >= ONSET_S + 2.0 else (3.0 if t >= ONSET_S else 0.0)  # noqa: E731
        m = compute_encounter_metrics(
            _trace(N_FULL, speed_fn=lambda t: 15.0, lateral_fn=lateral, requested_fn=requested),
            onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertTrue(m.departed_route)
        self.assertTrue(m.returned_to_route)
        self.assertAlmostEqual(m.final_route_lateral_m, 0.0)

    def test_short_trace_is_incomplete_not_success(self):
        m = compute_encounter_metrics(_trace(int(2.0 / DT), speed_fn=_brake_speed),
                                      onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertFalse(m.window_complete)
        self.assertEqual(m.outcome, OUTCOME_INCOMPLETE)
        self.assertFalse(m.safe_success)

    def test_contact_before_trace_ends_is_still_contact(self):
        m = compute_encounter_metrics(_trace(int(2.8 / DT), speed_fn=lambda t: 15.0),
                                      onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertFalse(m.window_complete)
        self.assertEqual(m.outcome, OUTCOME_CONTACT)

    def test_no_onset(self):
        m = compute_encounter_metrics(_trace(10, speed_fn=lambda t: 15.0), onset_time_s=None)
        self.assertEqual(m.outcome, OUTCOME_NO_ONSET)
        self.assertFalse(m.safe_success)

    def test_neither_stopped_nor_passed_is_unresolved(self):
        # Creeping at 1 m/s from far away never reaches the pedestrian.
        m = compute_encounter_metrics(_trace(N_FULL, x0=0.0, speed_fn=lambda t: 1.0),
                                      onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertEqual(m.outcome, OUTCOME_UNRESOLVED)
        self.assertFalse(m.safe_success)

    def test_contact_before_onset_is_outside_the_window(self):
        ticks = _trace(N_FULL, speed_fn=_brake_speed)
        # Pedestrian sits on the ego only before onset (e.g. a spawn artifact).
        for r in ticks[:10]:
            r.pedestrian_x_m, r.pedestrian_y_m = r.pos_x_m, r.pos_y_m
        m = compute_encounter_metrics(ticks, onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertEqual(m.contact_ticks, 0)


class SafetyAndDynamicsTests(unittest.TestCase):
    def test_drivable_violation_blocks_safe_success(self):
        lateral = lambda t: 3.0 * min(1.0, max(0.0, (t - ONSET_S) / 1.0))  # noqa: E731
        ticks = _trace(N_FULL, speed_fn=lambda t: 15.0, lateral_fn=lateral)
        flags = [None if r.sim_time_s < ONSET_S else (r.pos_y_m < 2.5) for r in ticks]
        m = compute_encounter_metrics(ticks, onset_time_s=ONSET_S, protocol=PROTOCOL,
                                      footprint_drivable=flags)
        self.assertEqual(m.outcome, OUTCOME_PASSED_CLEAR)
        self.assertGreater(m.drivable_violation_ticks, 0)
        self.assertFalse(m.safe_success)

    def test_unmeasured_drivability_is_none_not_zero(self):
        m = compute_encounter_metrics(_trace(N_FULL, speed_fn=_brake_speed),
                                      onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertIsNone(m.drivable_violation_ticks)

    def test_drivability_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_encounter_metrics(_trace(5, speed_fn=lambda t: 15.0), onset_time_s=0.0,
                                      footprint_drivable=[True])

    def test_low_speed_snap_excluded_from_peak_decel_and_jerk(self):
        def accel(t, v, a):
            return -27.0 if 0.0 < v < 5.0 else a  # injected Week-3-style snap
        m = compute_encounter_metrics(_trace(N_FULL, speed_fn=_brake_speed, accel_override=accel),
                                      onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertAlmostEqual(m.peak_decel_normal_speed_mps2, -8.0, delta=0.01)
        self.assertLess(m.max_abs_jerk_normal_speed_mps3, 8.0 / DT + 1e-6)

    def test_protocol_is_recorded_and_validated(self):
        m = compute_encounter_metrics(_trace(N_FULL, speed_fn=_brake_speed),
                                      onset_time_s=ONSET_S, protocol=PROTOCOL)
        self.assertEqual(m.protocol["horizon_s"], 4.0)
        self.assertEqual(m.to_dict()["protocol"]["protocol_version"], 1)
        with self.assertRaises(ValueError):
            EncounterProtocol(horizon_s=0.0)


if __name__ == "__main__":
    unittest.main()
