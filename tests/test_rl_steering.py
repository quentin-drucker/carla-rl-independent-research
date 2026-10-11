import math
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from passage import Obstacle, compute_passage  # noqa: E402
from rl_steering import (  # noqa: E402
    STEER_COMMIT_LOCK_AT_ONSET,
    STEER_COMMIT_PER_TICK,
    STEERING_STATE_DIM,
    STEERING_STATE_NAMES,
    SteeringCommitment,
    SteeringRewardConfig,
    build_steering_observation,
    compute_steering_reward,
    frame_coordinates_m,
    heading_error_rad,
    policy_should_act,
    split_action,
)

# Town04 at the test26 pedestrian: ego heading -x, so the right vector is (0, -1).
ORIGIN, RIGHT = (100.0, 50.0), (0.0, -1.0)


def passage(pedestrian_lateral_m=0.0):
    """The measured Town04 road: 4 lanes, pedestrian in the ego's lane by default.
    Edges about -3.87 / +7.42 m, as in the live step 1 runs."""
    return compute_passage(lambda lat: -5.25 <= lat <= 8.80, [Obstacle(pedestrian_lateral_m)])


class ActionTests(unittest.TestCase):
    def test_clips_brake_and_right_only_u(self):
        self.assertEqual(split_action([1.5, -0.3]), (1.0, 0.0))
        self.assertEqual(split_action([-0.2, 2.0]), (0.0, 1.0))

    def test_both_sides_when_the_bounds_are_opened(self):
        self.assertEqual(split_action([0.4, -0.3], u_min=-1.0), (0.4, -0.3))

    def test_non_finite_action_raises(self):
        with self.assertRaises(ValueError):
            split_action([0.5, float("nan")])


class CommitmentTests(unittest.TestCase):
    def test_no_steering_before_the_hazard_starts(self):
        for mode in (STEER_COMMIT_PER_TICK, STEER_COMMIT_LOCK_AT_ONSET):
            c = SteeringCommitment(mode)
            self.assertEqual(c.target_offset_m(1.0, passage(), triggered=False), 0.0)
            self.assertFalse(c.locked)

    def test_per_tick_follows_u_and_the_passage_every_tick(self):
        c = SteeringCommitment(STEER_COMMIT_PER_TICK)
        self.assertAlmostEqual(c.target_offset_m(1.0, passage(), triggered=True), 7.42, places=2)
        self.assertAlmostEqual(c.target_offset_m(0.5, passage(), triggered=True), 3.71, places=2)
        # The passage narrows (road ends at 5.0 m): the same u now means a smaller swerve.
        narrower = compute_passage(lambda lat: -5.25 <= lat <= 5.0, [Obstacle(0.0)])
        self.assertAlmostEqual(c.target_offset_m(1.0, narrower, triggered=True), narrower.right_edge_m)
        self.assertLess(narrower.right_edge_m, 4.0)
        self.assertEqual(c.u_applied, 1.0)
        self.assertFalse(c.locked)

    def test_lock_freezes_the_route_at_the_first_onset_tick(self):
        c = SteeringCommitment(STEER_COMMIT_LOCK_AT_ONSET)
        first = passage()
        locked_target = c.target_offset_m(0.5, first, triggered=True)
        self.assertAlmostEqual(locked_target, 3.71, places=2)
        self.assertTrue(c.locked)
        # Later choices and later passages change nothing.
        moved = compute_passage(lambda lat: -5.25 <= lat <= 3.0, [Obstacle(0.0)])
        self.assertEqual(c.target_offset_m(1.0, moved, triggered=True), locked_target)
        self.assertEqual(c.target_offset_m(0.0, moved, triggered=True), locked_target)
        self.assertEqual(c.u_applied, 0.5)
        self.assertIs(c.observed_passage(moved), first)

    def test_per_tick_shows_the_current_passage(self):
        c = SteeringCommitment(STEER_COMMIT_PER_TICK)
        c.target_offset_m(0.5, passage(), triggered=True)
        moved = passage(2.0)
        self.assertIs(c.observed_passage(moved), moved)

    def test_missing_passage_is_counted_and_never_swerves_on_nothing(self):
        c = SteeringCommitment(STEER_COMMIT_LOCK_AT_ONSET)
        self.assertEqual(c.target_offset_m(1.0, None, triggered=True), 0.0)
        self.assertEqual(c.missing_passage_ticks, 1)
        self.assertFalse(c.locked)  # the lock waits for the first real passage
        self.assertAlmostEqual(c.target_offset_m(1.0, passage(), triggered=True), 7.42, places=2)
        self.assertTrue(c.locked)

    def test_per_tick_holds_the_last_target_through_a_missing_passage(self):
        c = SteeringCommitment(STEER_COMMIT_PER_TICK)
        held = c.target_offset_m(0.5, passage(), triggered=True)
        self.assertEqual(c.target_offset_m(1.0, None, triggered=True), held)
        self.assertEqual(c.missing_passage_ticks, 1)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            SteeringCommitment("sometimes")


class FrameGeometryTests(unittest.TestCase):
    def test_along_is_negative_before_the_line_and_lateral_is_plus_right(self):
        along, lateral = frame_coordinates_m(ORIGIN, RIGHT, (120.0, 48.0))  # 20 m short, 2 m to the right
        self.assertAlmostEqual(along, -20.0)
        self.assertAlmostEqual(lateral, 2.0)
        along, _ = frame_coordinates_m(ORIGIN, RIGHT, (95.0, 50.0))
        self.assertAlmostEqual(along, 5.0)  # past the line

    def test_heading_error_is_positive_when_pointing_right(self):
        self.assertAlmostEqual(heading_error_rad(180.0, RIGHT), 0.0)
        # CARLA yaw grows clockwise seen from above, i.e. turning right.
        self.assertAlmostEqual(math.degrees(heading_error_rad(190.0, RIGHT)), 10.0)
        self.assertAlmostEqual(math.degrees(heading_error_rad(-170.0, RIGHT)), 10.0)
        self.assertAlmostEqual(math.degrees(heading_error_rad(135.0, RIGHT)), -45.0)


class ObservationTests(unittest.TestCase):
    BASE = [0.5, 0.2, 0.3, 0.0, 1.0, 0.8]

    def _obs(self, **overrides):
        kwargs = dict(passage=passage(), pedestrian_lateral_m=0.0, pedestrian_lateral_speed_mps=0.0,
                      distance_to_pedestrian_line_m=20.0, ego_lateral_m=0.0, heading_error_rad=0.0, previous_u=0.0)
        kwargs.update(overrides)
        return build_steering_observation(self.BASE, **kwargs)

    def test_layout_is_the_base_six_then_the_named_eight(self):
        obs = self._obs(previous_u=0.5, ego_lateral_m=3.0, heading_error_rad=math.pi / 4,
                        pedestrian_lateral_speed_mps=1.5)
        self.assertEqual(len(obs), STEERING_STATE_DIM)
        self.assertEqual(STEERING_STATE_DIM, 14)
        self.assertEqual(obs[:6], self.BASE)
        named = dict(zip(STEERING_STATE_NAMES, obs[6:]))
        self.assertAlmostEqual(named["passage_left_edge_norm"], -3.87 / 12.0, places=2)
        self.assertAlmostEqual(named["passage_right_edge_norm"], 7.42 / 12.0, places=2)
        self.assertAlmostEqual(named["distance_to_pedestrian_line_norm"], 0.25)
        self.assertAlmostEqual(named["ego_lateral_norm"], 0.25)
        self.assertAlmostEqual(named["heading_error_norm"], 0.5)
        self.assertAlmostEqual(named["pedestrian_lateral_speed_norm"], 0.5)
        self.assertEqual(named["previous_u"], 0.5)

    def test_everything_stays_within_minus_one_and_one(self):
        obs = self._obs(pedestrian_lateral_m=-99.0, pedestrian_lateral_speed_mps=50.0,
                        distance_to_pedestrian_line_m=-500.0, ego_lateral_m=40.0, heading_error_rad=-9.0,
                        previous_u=3.0)
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in obs))
        self.assertEqual(obs[6 + STEERING_STATE_NAMES.index("distance_to_pedestrian_line_norm")], -1.0)

    def test_no_passage_or_no_room_on_a_side_reads_as_zero(self):
        self.assertEqual(self._obs(passage=None)[6:8], [0.0, 0.0])
        narrow = compute_passage(lambda lat: -1.0 <= lat <= 1.0)  # too narrow for the car at all
        self.assertEqual(self._obs(passage=narrow)[6:8], [0.0, 0.0])
        left_only = compute_passage(lambda lat: -5.25 <= lat <= 1.2, [Obstacle(0.0)])
        self.assertEqual(self._obs(passage=left_only)[7], 0.0)
        self.assertLess(self._obs(passage=left_only)[6], 0.0)

    def test_wrong_base_length_rejected(self):
        with self.assertRaises(ValueError):
            build_steering_observation([0.0] * 5, passage=None, pedestrian_lateral_m=0.0,
                                       pedestrian_lateral_speed_mps=0.0, distance_to_pedestrian_line_m=0.0,
                                       ego_lateral_m=0.0, heading_error_rad=0.0, previous_u=0.0)


def reward(**overrides):
    kwargs = dict(contact_this_tick=False, ego_speed_mps=10.0, off_road_this_tick=False, window_ended=False,
                  min_clearance_m=None, went_off_road=False, jerk_mps3=0.0, brake_cmd_this_tick=0.0,
                  brake_cmd_prev_tick=0.0, lateral_accel_mps2=0.0, hazard_active=True)
    kwargs.update(overrides)
    return compute_steering_reward(**kwargs)


class RewardOrderingTests(unittest.TestCase):
    def test_a_quiet_tick_costs_nothing(self):
        total, breakdown = reward()
        self.assertEqual(total, 0.0)
        self.assertEqual(set(breakdown), {"r_total", "r_safety", "r_comfort", "r_contact", "r_off_road", "r_clearance"})

    def test_a_faster_hit_is_worse(self):
        slow, _ = reward(contact_this_tick=True, ego_speed_mps=5.0)
        fast, _ = reward(contact_this_tick=True, ego_speed_mps=20.0)
        self.assertAlmostEqual(slow, -375.0)
        self.assertAlmostEqual(fast, -600.0)

    def test_the_lightest_hit_is_worse_than_a_whole_window_off_the_road(self):
        lightest_hit, _ = reward(contact_this_tick=True, ego_speed_mps=0.0)
        whole_window_off_road = 400 * reward(off_road_this_tick=True, went_off_road=True)[0]  # 8 s at 50 Hz
        best_clean_run = reward(window_ended=True, min_clearance_m=5.0)[0]
        self.assertLess(lightest_hit, whole_window_off_road - best_clean_run)

    def test_clearance_counts_only_up_to_enough(self):
        def bonus(clearance_m, **kw):
            return reward(window_ended=True, min_clearance_m=clearance_m, **kw)[1]["r_clearance"]
        self.assertEqual(bonus(0.5), bonus(5.0))       # more than enough earns nothing extra
        self.assertAlmostEqual(bonus(0.5), 50.0)
        self.assertAlmostEqual(bonus(0.2), 20.0)       # a narrow escape earns less
        self.assertEqual(bonus(5.0, went_off_road=True), 0.0)       # road-keeping outranks clearance
        self.assertEqual(bonus(5.0, contact_this_tick=True), 0.0)
        self.assertEqual(reward(min_clearance_m=5.0)[1]["r_clearance"], 0.0)  # only paid when the window ends

    def test_a_gentler_swerve_wins_when_safety_is_equal(self):
        gentle, _ = reward(lateral_accel_mps2=5.0)
        violent, _ = reward(lateral_accel_mps2=25.0)
        self.assertGreater(gentle, violent)

    def test_comfort_is_small_next_to_safety(self):
        # A slam brake with the worst jerk and lateral acceleration seen in step 1, for 2 s.
        harsh_two_seconds = 100 * reward(jerk_mps3=770.0, lateral_accel_mps2=26.0)[1]["r_comfort"]
        self.assertGreater(harsh_two_seconds, -SteeringRewardConfig().w_contact)
        self.assertLess(harsh_two_seconds, 0.0)

    def test_comfort_is_not_charged_before_the_hazard(self):
        # Before onset the cruise controller drives, so nothing it does is the policy's fault.
        self.assertEqual(reward(jerk_mps3=300.0, lateral_accel_mps2=20.0, brake_cmd_this_tick=1.0,
                                hazard_active=False)[0], 0.0)
        self.assertLess(reward(brake_cmd_this_tick=1.0)[0], 0.0)

    def test_config_refuses_weights_that_break_the_ordering(self):
        with self.assertRaises(ValueError):
            SteeringRewardConfig(w_contact=100.0)
        with self.assertRaises(ValueError):
            SteeringRewardConfig(enough_clearance_m=0.0)
        self.assertAlmostEqual(
            reward(window_ended=True, min_clearance_m=0.5,
                   config=SteeringRewardConfig(enough_clearance_m=1.0))[1]["r_clearance"], 25.0)


class TrainingHelperTests(unittest.TestCase):
    def test_policy_takes_over_one_second_before_onset(self):
        # 20 m/s, onset 0.8 s out: trigger at 160 - 16 = 144 m, hand-over 20 m earlier.
        kw = dict(encounter_distance_m=160.0, trigger_ttc_s=0.8, speed_mps=20.0)
        self.assertFalse(policy_should_act(dist_travelled_m=100.0, **kw))
        self.assertFalse(policy_should_act(dist_travelled_m=123.9, **kw))
        self.assertTrue(policy_should_act(dist_travelled_m=124.0, **kw))
        self.assertTrue(policy_should_act(dist_travelled_m=150.0, **kw))  # already past onset

    def test_a_stopped_car_is_handed_over_only_at_the_trigger_point(self):
        self.assertFalse(policy_should_act(dist_travelled_m=159.0, encounter_distance_m=160.0,
                                           trigger_ttc_s=1.0, speed_mps=0.0))


if __name__ == "__main__":
    unittest.main()
