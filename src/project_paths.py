"""Canonical filesystem locations for source code and research artifacts.

All paths are anchored to the repository root so scripts behave the same when
launched from the project root, ``src/``, an IDE, or a test runner.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
ANALYSIS_DIR = RUNS_DIR / "analysis"
