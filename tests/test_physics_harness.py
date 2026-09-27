import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physics_harness import (  # noqa: E402
    compute_derived_kinematics,
    compute_lateral_displacement_m,
    compute_body_slip_angle_deg,
    compute_turn_radius_m,
    compute_lateral_accel_from_yaw_rate,
    classify_stop_outcome,
    classify_rollover,
    classify_upright_recovery,
    classify_throttle_brake_symmetry,
    get_wheel_steer_angle_deg,
    get_front_wheel_steer_angles_deg,
    inner_wheel_steer_angle_deg,
    summarize_deceleration,
    summarize_acceleration,
    compute_rise_time_s,
    detect_sustained_near_stop_onset_s,
    detect_low_speed_snap_events,
    compute_forward_velocity_components,
    LOW_SPEED_ARTIFACT_THRESHOLD_MPS,
)


class ComputeDerivedKinematicsTests(unittest.TestCase):
    def test_first_tick_has_no_previous_sample_returns_none(self):
        accel, yaw_rate = compute_derived_kinematics(
            prev_speed_mps=None, curr_speed_mps=5.0,
            prev_yaw_deg=None, curr_yaw_deg=10.0, dt_s=0.02,
        )
        self.assertIsNone(accel)
        self.assertIsNone(yaw_rate)

    def test_zero_or_negative_dt_returns_none(self):
        accel, yaw_rate = compute_derived_kinematics(
            prev_speed_mps=5.0, curr_speed_mps=6.0,
            prev_yaw_deg=0.0, curr_yaw_deg=1.0, dt_s=0.0,
        )
        self.assertIsNone(accel)
        self.assertIsNone(yaw_rate)

    def test_simple_acceleration_and_yaw_rate(self):
        accel, yaw_rate = compute_derived_kinematics(
            prev_speed_mps=10.0, curr_speed_mps=10.5,
            prev_yaw_deg=0.0, curr_yaw_deg=1.0, dt_s=0.5,
        )
        self.assertAlmostEqual(accel, 1.0)
        self.assertAlmostEqual(yaw_rate, 2.0)

    def test_yaw_wraparound_takes_shortest_turn(self):
        # 179 -> -179 is really a +2 degree turn, not a -358 degree jump.
        accel, yaw_rate = compute_derived_kinematics(
            prev_speed_mps=5.0, curr_speed_mps=5.0,
            prev_yaw_deg=179.0, curr_yaw_deg=-179.0, dt_s=1.0,
        )
        self.assertAlmostEqual(yaw_rate, 2.0)

    def test_yaw_wraparound_other_direction(self):
        accel, yaw_rate = compute_derived_kinematics(
            prev_speed_mps=5.0, curr_speed_mps=5.0,
            prev_yaw_deg=-179.0, curr_yaw_deg=179.0, dt_s=1.0,
        )
        self.assertAlmostEqual(yaw_rate, -2.0)


class ComputeLateralDisplacementTests(unittest.TestCase):
    def test_zero_displacement_on_the_reference_line(self):
        d = compute_lateral_displacement_m(
            start_x_m=0.0, start_y_m=0.0, start_yaw_deg=0.0,
            curr_x_m=10.0, curr_y_m=0.0,
        )
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_positive_to_the_right_of_heading(self):
        # Heading along +x; "right" per this module's convention is +y.
        d = compute_lateral_displacement_m(
            start_x_m=0.0, start_y_m=0.0, start_yaw_deg=0.0,
            curr_x_m=10.0, curr_y_m=2.0,
        )
        self.assertAlmostEqual(d, 2.0, places=6)

    def test_negative_to_the_left_of_heading(self):
        d = compute_lateral_displacement_m(
            start_x_m=0.0, start_y_m=0.0, start_yaw_deg=0.0,
            curr_x_m=10.0, curr_y_m=-2.0,
        )
        self.assertAlmostEqual(d, -2.0, places=6)


class ClassifyStopOutcomeTests(unittest.TestCase):
    def test_below_threshold_is_stopped(self):
        self.assertEqual(classify_stop_outcome(final_speed_mps=0.05), "stopped")

    def test_at_or_above_threshold_is_not_stopped(self):
        self.assertEqual(classify_stop_outcome(final_speed_mps=0.15), "not_stopped")
        self.assertEqual(classify_stop_outcome(final_speed_mps=2.0), "not_stopped")


class ClassifyRolloverTests(unittest.TestCase):
    def test_below_threshold_is_not_observed(self):
        result = classify_rollover(max_abs_roll_deg=15.0, threshold_deg=60.0)
        self.assertFalse(result.rollover_detected)
        self.assertEqual(result.evidence_status, "not_observed_in_tested_range")

    def test_at_threshold_is_confirmed(self):
        result = classify_rollover(max_abs_roll_deg=60.0, threshold_deg=60.0)
        self.assertTrue(result.rollover_detected)
        self.assertEqual(result.evidence_status, "confirmed")

    def test_above_threshold_is_confirmed(self):
        result = classify_rollover(max_abs_roll_deg=95.0, threshold_deg=60.0)
        self.assertTrue(result.rollover_detected)
        self.assertEqual(result.evidence_status, "confirmed")


class ClassifyUprightRecoveryTests(unittest.TestCase):
    def test_below_threshold_is_upright(self):
        self.assertEqual(classify_upright_recovery(final_abs_roll_deg=2.0), "upright")

    def test_at_or_above_threshold_is_not_upright(self):
        self.assertEqual(classify_upright_recovery(final_abs_roll_deg=15.0), "not_upright")
        self.assertEqual(classify_upright_recovery(final_abs_roll_deg=90.0), "not_upright")


class ClassifyThrottleBrakeSymmetryTests(unittest.TestCase):
    def test_equal_magnitudes_are_symmetric(self):
        self.assertEqual(
            classify_throttle_brake_symmetry(
                accel_response_mps2=3.0, decel_response_mps2=-3.0
            ),
            "symmetric",
        )

    def test_within_tolerance_is_symmetric(self):
        self.assertEqual(
            classify_throttle_brake_symmetry(
                accel_response_mps2=3.0, decel_response_mps2=-3.3, tolerance_ratio=0.15
            ),
            "symmetric",
        )

    def test_beyond_tolerance_is_asymmetric(self):
        self.assertEqual(
            classify_throttle_brake_symmetry(
                accel_response_mps2=2.0, decel_response_mps2=-6.0, tolerance_ratio=0.15
            ),
            "asymmetric",
        )

    def test_both_zero_is_inconclusive(self):
        self.assertEqual(
            classify_throttle_brake_symmetry(
                accel_response_mps2=0.0, decel_response_mps2=0.0
            ),
            "inconclusive",
        )


class ComputeBodySlipAngleTests(unittest.TestCase):
    def test_velocity_aligned_with_heading_is_zero_slip(self):
        angle = compute_body_slip_angle_deg(vel_x_mps=10.0, vel_y_mps=0.0, yaw_deg=0.0)
        self.assertAlmostEqual(angle, 0.0, places=6)

    def test_velocity_perpendicular_to_heading_is_90_degrees(self):
        angle = compute_body_slip_angle_deg(vel_x_mps=0.0, vel_y_mps=10.0, yaw_deg=0.0)
        self.assertAlmostEqual(angle, 90.0, places=6)

    def test_below_min_speed_returns_none(self):
        # Near-zero speed makes the velocity heading numerically meaningless
        # -- must not report a bogus large slip angle from sensor noise.
        angle = compute_body_slip_angle_deg(vel_x_mps=0.05, vel_y_mps=0.05, yaw_deg=0.0, min_speed_mps=0.5)
        self.assertIsNone(angle)

    def test_default_threshold_excludes_the_2026_09_26_artifact_speed(self):
        # Reproduces the exact artifact tick from the 2026-09-26 steering-lock
        # finding: speed collapsed from 6.38 to 2.02 m/s in one 0.02s tick
        # (>20g, physically impossible), corrupting the slip angle to >150deg.
        # The default threshold must exclude this speed, not just literal zero.
        angle = compute_body_slip_angle_deg(vel_x_mps=-0.391, vel_y_mps=-1.891, yaw_deg=105.24)
        self.assertIsNone(angle)
        self.assertLess(2.017, LOW_SPEED_ARTIFACT_THRESHOLD_MPS)

    def test_wraps_to_shortest_signed_angle(self):
        angle = compute_body_slip_angle_deg(vel_x_mps=-10.0, vel_y_mps=-0.01, yaw_deg=179.0)
        self.assertLess(abs(angle), 5.0)


class ComputeTurnRadiusTests(unittest.TestCase):
    def test_none_yaw_rate_returns_none(self):
        self.assertIsNone(compute_turn_radius_m(speed_mps=10.0, yaw_rate_dps=None))

    def test_near_zero_yaw_rate_returns_none_not_huge_radius(self):
        self.assertIsNone(compute_turn_radius_m(speed_mps=10.0, yaw_rate_dps=0.1, min_yaw_rate_dps=0.5))

    def test_known_speed_and_yaw_rate_gives_expected_radius(self):
        import math
        # v = 10 m/s, omega = 1 rad/s (converted to deg/s) -> radius = 10 m
        radius = compute_turn_radius_m(speed_mps=10.0, yaw_rate_dps=math.degrees(1.0))
        self.assertAlmostEqual(radius, 10.0, places=4)

    def test_negative_yaw_rate_gives_positive_radius(self):
        import math
        radius = compute_turn_radius_m(speed_mps=10.0, yaw_rate_dps=-math.degrees(1.0))
        self.assertAlmostEqual(radius, 10.0, places=4)


class ComputeLateralAccelFromYawRateTests(unittest.TestCase):
    def test_none_yaw_rate_returns_none(self):
        self.assertIsNone(compute_lateral_accel_from_yaw_rate(speed_mps=10.0, yaw_rate_dps=None))

    def test_known_values(self):
        import math
        a_lat = compute_lateral_accel_from_yaw_rate(speed_mps=10.0, yaw_rate_dps=math.degrees(1.0))
        self.assertAlmostEqual(a_lat, 10.0, places=4)

    def test_zero_yaw_rate_is_zero_lateral_accel(self):
        self.assertAlmostEqual(compute_lateral_accel_from_yaw_rate(speed_mps=10.0, yaw_rate_dps=0.0), 0.0)


class SummarizeAccelerationTests(unittest.TestCase):
    def test_splits_normal_and_low_speed_regimes(self):
        pairs = [
            (10.0, 3.0), (7.0, 4.0), (6.0, 5.0),  # normal speed (>=5)
            (4.0, 20.0), (2.0, 25.0), (0.5, 18.0),  # low speed (<5)
        ]
        summary = summarize_acceleration(speed_accel_pairs=pairs, threshold_mps=5.0)
        self.assertAlmostEqual(summary.normal_speed_peak_accel_mps2, 5.0)
        self.assertAlmostEqual(summary.normal_speed_mean_accel_mps2, 4.0)
        self.assertAlmostEqual(summary.low_speed_transient_peak_accel_mps2, 25.0)

    def test_skips_none_accel_samples(self):
        pairs = [(10.0, None), (10.0, 3.0), (2.0, None)]
        summary = summarize_acceleration(speed_accel_pairs=pairs, threshold_mps=5.0)
        self.assertAlmostEqual(summary.normal_speed_peak_accel_mps2, 3.0)
        self.assertIsNone(summary.low_speed_transient_peak_accel_mps2)

    def test_empty_regime_reports_none_not_zero(self):
        pairs = [(10.0, 3.0), (8.0, 4.0)]
        summary = summarize_acceleration(speed_accel_pairs=pairs, threshold_mps=5.0)
        self.assertIsNone(summary.low_speed_transient_peak_accel_mps2)


class ComputeRiseTimeTests(unittest.TestCase):
    def test_empty_input_returns_none(self):
        self.assertIsNone(compute_rise_time_s(time_speed_value_pairs=[]))

    def test_zero_final_value_returns_none(self):
        pairs = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)]
        self.assertIsNone(compute_rise_time_s(time_speed_value_pairs=pairs))

    def test_finds_first_time_crossing_target_fraction(self):
        pairs = [(0.0, 0.0), (0.1, 3.0), (0.2, 8.0), (0.3, 9.5), (0.4, 10.0)]
        rise_time = compute_rise_time_s(time_speed_value_pairs=pairs, target_fraction=0.9)
        self.assertAlmostEqual(rise_time, 0.3)

    def test_negative_values_use_absolute_magnitude(self):
        pairs = [(0.0, 0.0), (0.1, -3.0), (0.2, -9.0), (0.3, -10.0)]
        rise_time = compute_rise_time_s(time_speed_value_pairs=pairs, target_fraction=0.9)
        self.assertAlmostEqual(rise_time, 0.2)


class DetectSustainedNearStopOnsetTests(unittest.TestCase):
    def test_never_drops_below_threshold_returns_none(self):
        pairs = [(10.0, 0.0), (8.0, 0.02), (6.0, 0.04)]
        self.assertIsNone(detect_sustained_near_stop_onset_s(speed_time_pairs=pairs, threshold_mps=0.5))

    def test_sustained_drop_returns_onset_time(self):
        pairs = [(6.0, 0.0), (3.0, 0.02), (0.9, 0.04), (0.4, 0.06), (0.2, 0.08)]
        onset = detect_sustained_near_stop_onset_s(speed_time_pairs=pairs, threshold_mps=0.5)
        self.assertAlmostEqual(onset, 0.06)

    def test_transient_dip_that_recovers_does_not_count(self):
        # A momentary noisy low reading that recovers afterward must not be
        # reported as "came to a stop" -- only a drop that holds through the
        # end of the run counts.
        pairs = [(6.0, 0.0), (0.3, 0.02), (5.0, 0.04), (4.0, 0.06)]
        self.assertIsNone(detect_sustained_near_stop_onset_s(speed_time_pairs=pairs, threshold_mps=0.5))

    def test_already_stopped_from_the_start(self):
        pairs = [(0.1, 0.0), (0.05, 0.02)]
        onset = detect_sustained_near_stop_onset_s(speed_time_pairs=pairs, threshold_mps=0.5)
        self.assertAlmostEqual(onset, 0.0)

    def test_empty_input_returns_none(self):
        self.assertIsNone(detect_sustained_near_stop_onset_s(speed_time_pairs=[], threshold_mps=0.5))


class DetectLowSpeedSnapEventsTests(unittest.TestCase):
    def test_no_events_below_threshold(self):
        pairs = [(-5.0, 0.0), (-10.0, 0.02), (-27.0, 0.04)]  # normal + braking-transient magnitudes
        self.assertEqual(detect_low_speed_snap_events(pairs), [])

    def test_flags_single_extreme_tick(self):
        # Reproduces the exact 2026-09-26 test10 finding (45mph_left_step,
        # tick 58): speed 6.38 -> 2.02 m/s in one 0.02s tick.
        pairs = [(-13.6, 0.92), (-12.2, 1.02), (-218.23, 1.18), (-32.6, 1.20)]
        events = detect_low_speed_snap_events(pairs)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0][0], 1.18)
        self.assertAlmostEqual(events[0][1], -218.23)

    def test_flags_positive_and_negative_extremes(self):
        pairs = [(150.0, 0.5), (-150.0, 0.6)]
        events = detect_low_speed_snap_events(pairs)
        self.assertEqual(len(events), 2)

    def test_none_accel_is_skipped_not_a_crash(self):
        pairs = [(None, 0.0), (-5.0, 0.02)]
        self.assertEqual(detect_low_speed_snap_events(pairs), [])

    def test_custom_threshold(self):
        pairs = [(-27.1, 0.5)]  # the already-documented braking transient magnitude
        self.assertEqual(detect_low_speed_snap_events(pairs), [])
        self.assertEqual(len(detect_low_speed_snap_events(pairs, accel_threshold_mps2=20.0)), 1)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(detect_low_speed_snap_events([]), [])


class SummarizeDecelerationTests(unittest.TestCase):
    def test_splits_normal_and_low_speed_regimes(self):
        pairs = [
            (10.0, -3.0), (7.0, -4.0), (6.0, -5.0),  # normal speed (>=5)
            (4.0, -20.0), (2.0, -25.0), (0.5, -18.0),  # low speed (<5)
        ]
        summary = summarize_deceleration(speed_accel_pairs=pairs, threshold_mps=5.0)
        self.assertAlmostEqual(summary.normal_speed_peak_decel_mps2, -5.0)
        self.assertAlmostEqual(summary.normal_speed_mean_decel_mps2, -4.0)
        self.assertAlmostEqual(summary.low_speed_transient_peak_decel_mps2, -25.0)

    def test_skips_none_accel_samples(self):
        pairs = [(10.0, None), (10.0, -3.0), (2.0, None)]
        summary = summarize_deceleration(speed_accel_pairs=pairs, threshold_mps=5.0)
        self.assertAlmostEqual(summary.normal_speed_peak_decel_mps2, -3.0)
        self.assertIsNone(summary.low_speed_transient_peak_decel_mps2)

    def test_empty_regime_reports_none_not_zero(self):
        # A run that never drops below threshold must not report a fake
        # 0.0 "low speed transient" -- that would misrepresent an untested
        # regime as a measured zero-deceleration result.
        pairs = [(10.0, -3.0), (8.0, -4.0)]
        summary = summarize_deceleration(speed_accel_pairs=pairs, threshold_mps=5.0)
        self.assertIsNone(summary.low_speed_transient_peak_decel_mps2)


class _RaisingVehicle:
    def get_wheel_steer_angle(self, wheel_location):
        raise RuntimeError("not supported by this actor")


class _WorkingVehicle:
    def get_wheel_steer_angle(self, wheel_location):
        return 12.5


class _AsymmetricWheelVehicle:
    """FL/FR return different values, as a real vehicle does under
    Ackermann geometry -- used to test that get_front_wheel_steer_angles_deg
    reads both independently rather than assuming symmetry."""

    def get_wheel_steer_angle(self, wheel_location):
        import carla
        if wheel_location == carla.VehicleWheelLocation.FL_Wheel:
            return 46.7
        return 68.8


class GetWheelSteerAngleTests(unittest.TestCase):
    def test_returns_none_not_zero_when_api_call_fails(self):
        # A failed query must read as "not measurable", never as a
        # measured 0.0 -- silently defaulting to zero would misrepresent
        # a missing signal as "wheels pointed straight ahead".
        self.assertIsNone(get_wheel_steer_angle_deg(_RaisingVehicle()))

    def test_returns_value_when_available(self):
        self.assertEqual(get_wheel_steer_angle_deg(_WorkingVehicle()), 12.5)


class GetFrontWheelSteerAnglesTests(unittest.TestCase):
    def test_reads_fl_and_fr_independently(self):
        fl, fr = get_front_wheel_steer_angles_deg(_AsymmetricWheelVehicle())
        self.assertAlmostEqual(fl, 46.7)
        self.assertAlmostEqual(fr, 68.8)

    def test_fl_failure_does_not_prevent_fr_reading(self):
        class _FlFailsOnly:
            def get_wheel_steer_angle(self, wheel_location):
                import carla
                if wheel_location == carla.VehicleWheelLocation.FL_Wheel:
                    raise RuntimeError("fails")
                return 68.8

        fl, fr = get_front_wheel_steer_angles_deg(_FlFailsOnly())
        self.assertIsNone(fl)
        self.assertAlmostEqual(fr, 68.8)


class InnerWheelSteerAngleTests(unittest.TestCase):
    def test_picks_larger_magnitude_regardless_of_sign(self):
        # Reproduces the 2026-09-26 finding: a right turn's FL (outer) read
        # 46.7 deg while its FR (inner) read 68.8 deg -- the inner wheel is
        # the fair "achieved full-lock angle" summary, not FL unconditionally.
        self.assertAlmostEqual(inner_wheel_steer_angle_deg(front_left_deg=46.7, front_right_deg=68.8), 68.8)
        self.assertAlmostEqual(inner_wheel_steer_angle_deg(front_left_deg=-68.8, front_right_deg=-46.7), -68.8)

    def test_one_none_falls_back_to_the_other(self):
        self.assertAlmostEqual(inner_wheel_steer_angle_deg(front_left_deg=None, front_right_deg=68.8), 68.8)

    def test_both_none_returns_none(self):
        self.assertIsNone(inner_wheel_steer_angle_deg(front_left_deg=None, front_right_deg=None))


class ComputeForwardVelocityComponentsTests(unittest.TestCase):
    def test_yaw_zero_points_along_positive_x(self):
        vx, vy = compute_forward_velocity_components(yaw_deg=0.0, speed_mps=10.0)
        self.assertAlmostEqual(vx, 10.0)
        self.assertAlmostEqual(vy, 0.0, places=9)

    def test_yaw_90_points_along_positive_y(self):
        vx, vy = compute_forward_velocity_components(yaw_deg=90.0, speed_mps=10.0)
        self.assertAlmostEqual(vx, 0.0, places=9)
        self.assertAlmostEqual(vy, 10.0)

    def test_yaw_180_points_along_negative_x(self):
        vx, vy = compute_forward_velocity_components(yaw_deg=180.0, speed_mps=10.0)
        self.assertAlmostEqual(vx, -10.0, places=9)
        self.assertAlmostEqual(vy, 0.0, places=9)

    def test_magnitude_matches_speed_at_arbitrary_yaw(self):
        # Reproduces the actual spawn-90 heading found live 2026-09-27
        # (Town03_Opt open-location rollover retest).
        vx, vy = compute_forward_velocity_components(yaw_deg=-144.4, speed_mps=40.2)
        self.assertAlmostEqual((vx ** 2 + vy ** 2) ** 0.5, 40.2, places=6)

    def test_zero_speed_returns_zero_vector(self):
        vx, vy = compute_forward_velocity_components(yaw_deg=37.0, speed_mps=0.0)
        self.assertAlmostEqual(vx, 0.0, places=9)
        self.assertAlmostEqual(vy, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
