"""run_geometry_rescale.py pytest gate (ROADMAP §8, 2026-07-27 geometry-
rescale pass). Structural checks only -- the full 50-trial confirmatory
mini-pilot re-run is executed by the script itself (`python
run_geometry_rescale.py`), not by pytest; these tests keep `pytest -q` fast
while still catching a broken grid, a broken gate call, or a broken d=3
placement at the rescaled geometry.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from run_geometry_rescale import (  # noqa: E402
    RESCALED_COL_A, RESCALED_GRID_SHAPE, RESCALED_N_POLE_LAT,
    RESCALED_N_POLE_LON, MINIPILOT_D, build_rescaled_grid,
    run_minipilot_rescaled, single_source_gate)
from simulate import row_center_cols  # noqa: E402


def test_rescaled_grid_shape_and_span_match_original_footprint():
    """Same 18x28 deg span as the original 11x11 grid -- density only."""
    tm = build_rescaled_grid()
    assert tm.n_grid == RESCALED_N_POLE_LAT * RESCALED_N_POLE_LON
    assert np.isrealobj(tm.A) and np.isfinite(tm.A).all()
    assert tm.pole_lat.max() - tm.pole_lat.min() == pytest.approx(18.0)
    assert tm.pole_lon.max() - tm.pole_lon.min() == pytest.approx(28.0)


def test_d3_placement_stays_pinned_at_three_cells():
    idx_a, idx_b = row_center_cols(RESCALED_COL_A, MINIPILOT_D, grid_shape=RESCALED_GRID_SHAPE)
    row_a, col_a = divmod(idx_a, RESCALED_GRID_SHAPE[1])
    row_b, col_b = divmod(idx_b, RESCALED_GRID_SHAPE[1])
    assert row_a == row_b
    assert col_b - col_a == MINIPILOT_D == 3


def test_single_source_gate_passes_at_rescaled_geometry():
    tm = build_rescaled_grid()
    result = single_source_gate(tm=tm)
    assert result["passed"], result["failures"]


def test_minipilot_rescaled_runs_and_reports_real_variance_over_a_few_trials():
    """Not the pinned 50-trial confirmatory run (that's the script's job) --
    a small n_trials smoke test that the rescaled mini-pilot machinery
    produces finite, well-formed output and doesn't silently degenerate
    (e.g. via a broken grid_shape reshape or an out-of-range placement)."""
    import run_geometry_rescale as rgr
    tm = build_rescaled_grid()
    old_n = rgr.MINIPILOT_N_TRIALS
    rgr.MINIPILOT_N_TRIALS = 5
    try:
        result = run_minipilot_rescaled(tm)
    finally:
        rgr.MINIPILOT_N_TRIALS = old_n
    assert result["n_trials"] == 5
    for name in ("l2", "gibf", "mmv"):
        assert len(result["delta_r_bar"][name]) == 5
        assert np.isfinite(result["delta_r_bar"][name]).all()
        assert 0.0 <= result["p_sep_rate"][name] <= 1.0
    assert np.isfinite(result["gap_mean"])
    assert np.isfinite(result["gap_sd"])
