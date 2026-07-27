"""Pytest wrapper for the SPEC §S.6.4 single-source exact-recovery gate —
run_experiment_B.py's `single_source_gate()`. Any failure here means
Experiment B's B1/B3/mini-pilot run must not proceed (solver bug, not
science). Distinct from tests/test_solvers.py's lower-level solver checks:
this exercises the actual runner function used by `main()`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from run_experiment_B import single_source_gate  # noqa: E402


def test_single_source_gate_passes():
    result = single_source_gate()
    assert result["passed"], result["failures"]
