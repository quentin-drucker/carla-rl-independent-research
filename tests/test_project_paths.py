import os
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import ANALYSIS_DIR, CHECKPOINTS_DIR, PROJECT_ROOT, RUNS_DIR  # noqa: E402
from rl_eval_protocol import resolve_model_path  # noqa: E402


class ProjectPathTests(unittest.TestCase):
    def test_artifact_directories_are_root_anchored(self):
        self.assertEqual(RUNS_DIR, PROJECT_ROOT / "artifacts" / "runs")
        self.assertEqual(CHECKPOINTS_DIR, PROJECT_ROOT / "artifacts" / "checkpoints")
        self.assertEqual(ANALYSIS_DIR, RUNS_DIR / "analysis")

    def test_retained_model_name_resolves_after_relocation(self):
        resolved = resolve_model_path("sac_v3_1600k")
        self.assertEqual(resolved, os.path.abspath(CHECKPOINTS_DIR / "sac_v3_1600k.zip"))


if __name__ == "__main__":
    unittest.main()
