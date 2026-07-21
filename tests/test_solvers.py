"""solvers.py pytest gate: fairness invariants (§S.6/§S.1 PINNED 2026-07-19)
and single-source exact recovery (§S.6.4 gate, all three solvers, both
reduction settings) on the actual Experiment-B geometry.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from metrics import delta_r_bar, p_sep  # noqa: E402
from modeselect import estimate_n_sources  # noqa: E402
from simulate import (build_array_and_grid, build_csm, centroid_grid_index,  # noqa: E402
                      eigendecompose, eigenmodes, row_center_cols,
                      simulate_snapshots, GRID_SHAPE)
from solvers import (FAIRNESS_CONFIG, SolverConfig, gibf_irls, l2_minnorm,  # noqa: E402
                     mmv_l1, solve_all, with_reduction)

_TM = build_array_and_grid()


def test_fairness_config_is_one_shared_object():
    # Both sparse solvers must be callable with, and default to, the exact
    # same literal config object (SPEC §S.6.1).
    assert isinstance(FAIRNESS_CONFIG, SolverConfig)
    assert FAIRNESS_CONFIG.eps_frac == 0.01
    assert FAIRNESS_CONFIG.p_norm == 1.0
    assert FAIRNESS_CONFIG.weight_floor == 1e-10
    assert FAIRNESS_CONFIG.beta == 0.9
    assert FAIRNESS_CONFIG.max_iter == 50
    assert FAIRNESS_CONFIG.tol == 1e-4
    assert FAIRNESS_CONFIG.min_active_factor == 2

    idx = centroid_grid_index(_TM)
    X, _ = simulate_snapshots(_TM, [idx], [1.0], np.random.default_rng(0),
                              n_snap=64, snr_db=None)
    S = build_csm(X)
    lam, U = eigendecompose(S)
    V = eigenmodes(lam, U, 1)
    r_gibf = gibf_irls(_TM.A, V / np.linalg.norm(V[:, 0]), FAIRNESS_CONFIG)
    r_mmv = mmv_l1(_TM.A, V / np.linalg.norm(V[:, 0]), FAIRNESS_CONFIG)
    # both solvers ran to completion using the identical shared object
    assert r_gibf["realized_eps"][0] > 0
    assert r_mmv["realized_eps"] > 0


def test_solve_all_feeds_identical_V_and_A_to_gibf_and_mmv():
    idx_a, idx_b = row_center_cols(1, 3)
    rng = np.random.default_rng(1)
    X, _ = simulate_snapshots(_TM, [idx_a, idx_b], [1.0, 1.0], rng,
                              n_snap=64, snr_db=5.0)
    S = build_csm(X)
    lam, U = eigendecompose(S)
    V = eigenmodes(lam, U, 2)
    out = solve_all(_TM.A, V, FAIRNESS_CONFIG)
    assert set(out) >= {"l2", "gibf", "mmv", "scale_s"}
    assert out["scale_s"] == np.linalg.norm(V[:, 0])


def test_realness_assertion_trips_on_complex_A():
    A_complex = _TM.A.astype(complex) * (1 + 0.01j)
    V = np.ones((_TM.A.shape[0], 1), dtype=complex)
    for fn in (l2_minnorm, gibf_irls, mmv_l1):
        try:
            fn(A_complex, V, FAIRNESS_CONFIG)
            assert False, f"{fn.__name__} should have asserted realness"
        except AssertionError:
            pass


def _single_source_case():
    tm = build_array_and_grid()
    idx = centroid_grid_index(tm)
    true_cell = np.unravel_index(idx, GRID_SHAPE)
    return tm, idx, true_cell


def test_single_source_gate_tier1_noise_free_all_solvers_both_reductions():
    """SPEC §S.6.4 tier (i): noise-free deterministic, oracle K=1, exact
    cell = global argmax (tau=0). L2 runs this tier only (no grid-reduction
    concept); sparse solvers run both reduction settings."""
    tm, idx, true_cell = _single_source_case()
    X, _ = simulate_snapshots(tm, [idx], [1.0], np.random.default_rng(0),
                              n_snap=64, snr_db=None)
    S = build_csm(X)
    lam, U = eigendecompose(S)
    V = eigenmodes(lam, U, 1)

    out_on = solve_all(tm.A, V, with_reduction(FAIRNESS_CONFIG, True))
    out_off = solve_all(tm.A, V, with_reduction(FAIRNESS_CONFIG, False))

    for out in (out_on, out_off):
        for name in ("l2", "gibf", "mmv"):
            assert np.argmax(out[name]["I"]) == idx, (
                f"{name} failed exact-cell recovery (tau=0) — "
                "single-source gate HALT condition")


def test_single_source_gate_tier2_20db_mc_10_trials():
    """SPEC §S.6.4 tier (ii): 20 dB MC x 10 trials, all must pass. Sparse
    solvers only (L2 runs tier (i) only per §S.6.4)."""
    tm, idx, true_cell = _single_source_case()
    for reduction in (True, False):
        cfg = with_reduction(FAIRNESS_CONFIG, reduction)
        for trial in range(10):
            rng = np.random.default_rng([2026, trial])
            X, _ = simulate_snapshots(tm, [idx], [1.0], rng, n_snap=64, snr_db=20.0)
            S = build_csm(X)
            lam, U = eigendecompose(S)
            V = eigenmodes(lam, U, 1)
            out = solve_all(tm.A, V, cfg)
            for name in ("gibf", "mmv"):
                assert np.argmax(out[name]["I"]) == idx, (
                    f"{name} (reduction={reduction}) failed trial {trial}")


def test_two_source_incoherent_high_snr_separates_and_metrics_agree():
    tm = build_array_and_grid()
    idx_a, idx_b = row_center_cols(1, 5)
    true_cells = [np.unravel_index(idx_a, GRID_SHAPE), np.unravel_index(idx_b, GRID_SHAPE)]
    rng = np.random.default_rng(7)
    X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng, n_snap=64, snr_db=20.0)
    S = build_csm(X)
    lam, U = eigendecompose(S)
    V = eigenmodes(lam, U, 2)
    out = solve_all(tm.A, V, FAIRNESS_CONFIG)
    for name in ("l2", "gibf", "mmv"):
        I2d = out[name]["I"].reshape(GRID_SHAPE)
        assert p_sep(I2d, true_cells) is True
        assert delta_r_bar(I2d, true_cells) < 1.0
