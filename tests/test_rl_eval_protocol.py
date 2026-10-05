import csv
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_eval_protocol import (  # noqa: E402
    LEGACY_COLLISION,
    LEGACY_FULL_STOP,
    LEGACY_SLOWED_AVOIDED,
    SUMMARY_COLUMNS,
    LegacyLabelTracker,
    check_manifest,
    extended_sim_seconds,
    file_sha256,
    legacy_outcome_label,
    legacy_truncated,
    load_sweep_configs,
    resolve_model_path,
    summary_row,
    trace_telemetry,
    window_decision,
)
from scenario_config import ScenarioConfig  # noqa: E402

DT = 0.02


def _info(**kw):
    base = dict(collision_sensor=False, collision_legacy_proximity=False, collision_geometric_contact=False,
                full_stop_achieved=False, ped_crossed=False)
    base.update(kw)
    return base


class WindowDecisionTests(unittest.TestCase):
    def test_runs_until_onset_plus_horizon_plus_one_tick(self):
        kw = dict(trigger_time_s=10.0, horizon_s=8.0, truncated=False, fixed_dt_s=DT)
        self.assertFalse(window_decision(sim_time_s=18.0, **kw).stop)
        d = window_decision(sim_time_s=18.02, **kw)
        self.assertEqual((d.stop, d.reason), (True, "window_complete"))

    def test_env_termination_is_not_a_stop_reason(self):
        # window_decision takes no `terminated` input on purpose: a collision,
        # first full stop, or far-cross clearing must not shorten the window.
        d = window_decision(sim_time_s=11.0, trigger_time_s=10.0, horizon_s=8.0, truncated=False, fixed_dt_s=DT)
        self.assertFalse(d.stop)

    def test_truncation_stops(self):
        d = window_decision(sim_time_s=5.0, trigger_time_s=None, horizon_s=8.0, truncated=True, fixed_dt_s=DT)
        self.assertEqual((d.stop, d.reason), (True, "env_truncated"))

    def test_no_trigger_keeps_running(self):
        self.assertFalse(window_decision(sim_time_s=99.0, trigger_time_s=None, horizon_s=8.0,
                                         truncated=False, fixed_dt_s=DT).stop)

    def test_waits_for_the_legacy_episode_to_end(self):
        kw = dict(sim_time_s=18.02, trigger_time_s=10.0, horizon_s=8.0, truncated=False, fixed_dt_s=DT)
        self.assertFalse(window_decision(legacy_ended=False, **kw).stop)
        self.assertTrue(window_decision(legacy_ended=True, **kw).stop)

    def test_extended_sim_seconds_covers_window_and_legacy_timeout(self):
        self.assertEqual(extended_sim_seconds(18.0, 8.0, 12.0), 31.0)
        self.assertEqual(extended_sim_seconds(18.0, 8.0, 5.0), 27.0)

    def test_legacy_truncation_uses_the_original_cap(self):
        kw = dict(original_sim_seconds=18.0, fixed_dt_s=DT, post_trigger_timeout=False)
        self.assertFalse(legacy_truncated(tick=899, **kw))
        self.assertTrue(legacy_truncated(tick=900, **kw))
        self.assertTrue(legacy_truncated(tick=10, original_sim_seconds=18.0, fixed_dt_s=DT, post_trigger_timeout=True))


class LegacyLabelTests(unittest.TestCase):
    def test_original_rule(self):
        self.assertEqual(legacy_outcome_label(collision_seen=True, terminated=True), LEGACY_COLLISION)
        self.assertEqual(legacy_outcome_label(collision_seen=False, terminated=True), LEGACY_FULL_STOP)
        self.assertEqual(legacy_outcome_label(collision_seen=False, terminated=False), LEGACY_SLOWED_AVOIDED)

    def test_far_cross_clearing_reproduces_the_known_full_stop_mislabel(self):
        t = LegacyLabelTracker()
        t.update(tick=1, info=_info(), truncated=False)
        t.update(tick=2, info=_info(ped_crossed=True), truncated=False)
        self.assertEqual((t.label, t.end_reason, t.end_tick), (LEGACY_FULL_STOP, "ped_crossed", 2))

    def test_geometric_only_contact_is_invisible_to_legacy(self):
        t = LegacyLabelTracker()
        t.update(tick=1, info=_info(collision_geometric_contact=True), truncated=False)
        t.update(tick=2, info=_info(collision_geometric_contact=True, full_stop_achieved=True), truncated=False)
        self.assertEqual(t.label, LEGACY_FULL_STOP)

    def test_sensor_or_proximity_is_collision(self):
        for key in ("collision_sensor", "collision_legacy_proximity"):
            t = LegacyLabelTracker()
            t.update(tick=1, info=_info(**{key: True}), truncated=False)
            self.assertEqual((t.label, t.end_reason), (LEGACY_COLLISION, "collision"))

    def test_collision_wins_over_full_stop_on_same_tick(self):
        t = LegacyLabelTracker()
        t.update(tick=3, info=_info(collision_sensor=True, full_stop_achieved=True), truncated=False)
        self.assertEqual(t.label, LEGACY_COLLISION)

    def test_truncation_without_termination_is_slowed_avoided(self):
        t = LegacyLabelTracker()
        t.update(tick=9, info=_info(), truncated=True)
        self.assertEqual((t.label, t.end_reason), (LEGACY_SLOWED_AVOIDED, "truncated"))

    def test_ticks_after_the_legacy_end_are_ignored(self):
        t = LegacyLabelTracker()
        t.update(tick=1, info=_info(full_stop_achieved=True), truncated=False)
        t.update(tick=2, info=_info(collision_sensor=True), truncated=False)
        self.assertEqual((t.label, t.end_tick), (LEGACY_FULL_STOP, 1))

    def test_unended_episode_has_no_label(self):
        self.assertIsNone(LegacyLabelTracker().label)


class TraceTelemetryTests(unittest.TestCase):
    def test_adds_pedestrian_position_without_mutating_input(self):
        tel = {"speed_mps": 5.0, "pos_x_m": 1.0}
        out = trace_telemetry(tel, {"pedestrian_x_m": 3.0, "pedestrian_y_m": 4.0})
        self.assertEqual((out["pedestrian_x_m"], out["pedestrian_y_m"]), (3.0, 4.0))
        self.assertNotIn("pedestrian_x_m", tel)

    def test_none_telemetry(self):
        self.assertIsNone(trace_telemetry(None, {}))


class ModelPathTests(unittest.TestCase):
    def test_resolves_extensionless_name_in_search_dir_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sac_test.zip")
            with open(p, "wb") as f:
                f.write(b"abc")
            self.assertEqual(resolve_model_path("sac_test", search_dir=d), os.path.abspath(p))
            self.assertEqual(file_sha256(p), hashlib.sha256(b"abc").hexdigest())

    def test_missing_model_raises_instead_of_defaulting(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                resolve_model_path("sac_does_not_exist", search_dir=d)


class SweepConfigTests(unittest.TestCase):
    def test_reads_only_canonical_profile_rows(self):
        cols = ["run_id", "brake_profile", "target_mph", "encounter_distance_m", "walker_speed_mps", "walker_side",
                "walker_cross", "trigger_ttc_s", "sim_seconds"]
        rows = [
            ["run_0001", "proportional_ramp", "35", "60", "1.5", "left", "far", "2.0", "18"],
            ["run_0002", "step_constant", "35", "60", "1.5", "left", "far", "2.0", "18"],
            ["run_0003", "proportional_ramp", "45", "70", "1.2", "right", "near", "", "18"],
        ]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sweep_summary.csv")
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(rows)
            cfgs = load_sweep_configs(p)
        self.assertEqual([c.run_id for c in cfgs], ["run_0001", "run_0003"])
        self.assertEqual(cfgs[0].trigger_ttc_s, 2.0)
        self.assertIsNone(cfgs[1].trigger_ttc_s)
        self.assertEqual(cfgs[1].walker_cross, "near")


class SummaryAndManifestTests(unittest.TestCase):
    def test_summary_row_has_exactly_the_columns(self):
        cfg = ScenarioConfig(target_mph=35.0, walker_cross="far", trigger_ttc_s=2.0)
        row = summary_row(episode=1, cfg=cfg, metrics={"outcome": "passed_clear", "min_clearance_m": 1.2,
                                                       "not_a_column": 1},
                          stop_reason="window_complete", legacy_label=LEGACY_FULL_STOP,
                          legacy_end_reason="ped_crossed", signal_counts={"geometric_contact": 0},
                          physics_label="avoidable")
        self.assertEqual(list(row), SUMMARY_COLUMNS)
        self.assertEqual((row["outcome"], row["legacy_label"], row["walker_cross"]),
                         ("passed_clear", LEGACY_FULL_STOP, "far"))

    def test_manifest_requires_provenance(self):
        with self.assertRaises(ValueError):
            check_manifest({"git_commit": "abc"})


class RunnerArgsTests(unittest.TestCase):
    def test_model_is_required(self):
        import contextlib
        import io

        from eval_policy_encounters import _parse_args

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parse_args(["--suite", "random"])
        self.assertEqual(_parse_args(["--model", "m", "--suite", "random"]).collision_signal, "geometric")


if __name__ == "__main__":
    unittest.main()
