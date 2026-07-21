"""simulate.py pytest gate: complex phasor snapshots, CSM, eigenmodes
(archived brief §11 test_simulate.py / test_csm.py, ROADMAP §2 invariants)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from simulate import (build_array_and_grid, build_csm, centroid_grid_index,  # noqa: E402
                      eigendecompose, eigenmodes, row_center_cols,
                      simulate_snapshots)


def test_grid_geometry_offset_from_coincidence():
    # Would raise ValueError inside build_transfer_matrix if any station/pole
    # (lat, lon) coincided exactly (Gate V V1 guard).
    tm = build_array_and_grid()
    assert tm.n_channels == 75
    assert tm.n_grid == 121
    assert np.isrealobj(tm.A)


def test_noise_free_clean_field_reconstructs_A_a():
    tm = build_array_and_grid()
    idx = centroid_grid_index(tm)
    X, X_clean = simulate_snapshots(tm, [idx], [1.0], np.random.default_rng(0),
                                    n_snap=8, snr_db=None)
    assert np.array_equal(X, X_clean)
    col = tm.col_slice("divfree").start + idx
    # X_clean columns must lie in span{A[:, col]} exactly (single source).
    a_col = tm.A[:, col]
    for t in range(8):
        # project out a_col and check near-zero residual
        coeff = np.vdot(a_col, X_clean[:, t]) / np.vdot(a_col, a_col)
        resid = X_clean[:, t] - coeff * a_col
        assert np.linalg.norm(resid) < 1e-9 * (np.linalg.norm(X_clean[:, t]) + 1e-300)


def test_incoherent_sources_uncorrelated_csm_rank():
    tm = build_array_and_grid()
    idx_a, idx_b = row_center_cols(1, 3)
    rng = np.random.default_rng(1)
    X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                              n_snap=2000, snr_db=None, coherence=None)
    S = build_csm(X)
    lam, _ = eigendecompose(S)
    # Noise-free, incoherent, equal-power two-source: exactly 2 informative
    # eigenvalues (up to numerical floor), the rest ~0.
    assert lam[0] > 0 and lam[1] > 0
    assert lam[2] < 1e-6 * lam[1]


def test_csm_hermitian_psd_and_not_near_real_for_coherent_phase():
    tm = build_array_and_grid()
    idx_a, idx_b = row_center_cols(1, 3)
    rng = np.random.default_rng(2)
    rho = 0.85 * np.exp(1j * np.deg2rad(90.0))
    X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                              n_snap=512, snr_db=20.0, coherence=rho)
    S = build_csm(X)
    assert np.allclose(S, S.conj().T)
    lam, _ = eigendecompose(S)
    assert (lam >= -1e-20).all()
    # phi=90 deg coherent pair must leave real, exploitable imaginary
    # structure in S (ROADMAP §2 invariant 3 / repo CLAUDE.md invariant 2).
    im_frac = np.linalg.norm(S.imag) / np.linalg.norm(np.abs(S))
    assert im_frac > 0.05


def test_eigenmodes_carries_sqrt_lambda_not_unit_vector():
    tm = build_array_and_grid()
    idx = centroid_grid_index(tm)
    rng = np.random.default_rng(3)
    X, _ = simulate_snapshots(tm, [idx], [1.0], rng, n_snap=64, snr_db=10.0)
    S = build_csm(X)
    lam, U = eigendecompose(S)
    V = eigenmodes(lam, U, 1)
    assert np.isclose(np.linalg.norm(V[:, 0]), np.sqrt(lam[0]))
    assert not np.isclose(np.linalg.norm(V[:, 0]), 1.0)


def test_row_center_cols_out_of_range_raises():
    import pytest
    with pytest.raises(ValueError):
        row_center_cols(9, 8)  # col_a=9, d=8 -> col_b=17, grid width is 11
