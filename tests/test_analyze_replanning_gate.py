import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_replanning_gate import controller_name, gate, walker_of  # noqa: E402

PAIR = (("A", "B"),)
FIXED = ["brake_u0", "brake_u1"]
REP = "rule"


def row(safe):
    return {"ego_safe_success": str(safe), "safe_success": str(safe)}


def table(case_rows):
    """case_rows: {(walker, mph, ttc): {controller: safe}}"""
    return {k: {c: row(s) for c, s in v.items()} for k, v in case_rows.items()}


class GateTests(unittest.TestCase):
    def test_a_universal_fixed_option_means_no_separation(self):
        t = table({("A", 30.0, 0.8): {"brake_u0": False, "brake_u1": True, REP: True},
                   ("B", 30.0, 0.8): {"brake_u0": True, "brake_u1": True, REP: True}})
        res = gate(t, fixed=FIXED, replanner=REP, pairs=PAIR)[("A", "B")]
        self.assertEqual((res["separating"], res["passes"]), ([], False))

    def test_disjoint_safe_sets_with_a_replanner_safe_in_both_pass(self):
        t = table({("A", 30.0, 0.8): {"brake_u0": False, "brake_u1": True, REP: True},
                   ("B", 30.0, 0.8): {"brake_u0": True, "brake_u1": False, REP: True}})
        res = gate(t, fixed=FIXED, replanner=REP, pairs=PAIR)[("A", "B")]
        self.assertEqual((res["separating"], res["replanner_wins"], res["passes"]), ([(30.0, 0.8)], [(30.0, 0.8)], True))

    def test_separation_without_a_working_replanner_does_not_pass(self):
        t = table({("A", 30.0, 0.8): {"brake_u0": False, "brake_u1": True, REP: True},
                   ("B", 30.0, 0.8): {"brake_u0": True, "brake_u1": False, REP: False}})
        res = gate(t, fixed=FIXED, replanner=REP, pairs=PAIR)[("A", "B")]
        self.assertEqual((len(res["separating"]), res["passes"], res["replanner_loses"]), (1, False, [("B", 30.0, 0.8)]))

    def test_strong_win_where_every_fixed_option_fails(self):
        t = table({("A", 30.0, 0.8): {"brake_u0": False, "brake_u1": False, REP: True},
                   ("B", 30.0, 0.8): {"brake_u0": True, "brake_u1": True, REP: True}})
        res = gate(t, fixed=FIXED, replanner=REP, pairs=PAIR)[("A", "B")]
        self.assertEqual((res["strong_wins"], res["passes"]), ([("A", 30.0, 0.8)], True))

    def test_names(self):
        self.assertEqual(walker_of("35mph_ttc0.8_right_turn_back_chg0.6s"), "right_turn_back")
        self.assertEqual(walker_of("35mph_ttc0.8_right_cross"), "right_cross")
        self.assertEqual(controller_name({"fixed_u": [0, 1], "fixed_brake": 0.0}, "1.0"), "coast_u1")
        self.assertEqual(controller_name({"scripted_aim": "pass_behind", "steer_commit": "lock_at_onset",
                                          "lock_delay_s": 0.5}, ""), "rule_pass_behind_lock_at_onset_0.5s")
        self.assertEqual(controller_name({"model_path": "x/final_model.zip", "steer_commit": "per_tick"}, ""), "sac_per_tick")


if __name__ == "__main__":
    unittest.main()
