"""simulate.py pytest gate: complex phasor snapshots, CSM, eigenmodes
(archived brief §11 test_simulate.py / test_csm.py, ROADMAP §2 invariants)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from simulate import (CONFIRMATORY_D, GRID_SHAPE, N_POLE_LAT,  # noqa: E402
                      N_POLE_LON, POLE_LAT_STEP_DEG, POLE_LON_STEP_DEG,
                      POLE_SPACING_KM, STATION_LON_SPACING_KM,
                      build_array_and_grid, build_csm, centroid_grid_index,
                      eigendecompose, eigenmodes, geometry_record,
                      row_center_cols, simulate_snapshots)


def test_grid_geometry_offset_from_coincidence():
    # Would raise ValueError inside build_transfer_matrix if any station/pole
    # (lat, lon) coincided exactly (Gate V V1 guard).
    tm = build_array_and_grid()
    assert tm.n_channels == 75
    assert tm.n_grid == N_POLE_LAT * N_POLE_LON
    assert np.isrealobj(tm.A)


def test_refined_grid_pins_confirmatory_d_to_one_station_spacing():
    """ROADMAP §8 2026-07-27 ruling 1 (the geometry-rescale pass). The rule
    that fixes the grid: the PRE-REGISTERED confirmatory coordinate d=3
    (§8-iii) equals the array's along-separation-axis station spacing; the
    pole spacing follows as that / 3 (image-grid oversampling x3). This test
    is the executable form of that rule — if the grid is ever re-refined, the
    relation must still hold or the refinement was improvised."""
    assert CONFIRMATORY_D == 3                       # pre-registered, never moved
    assert np.isclose(CONFIRMATORY_D * POLE_SPACING_KM, STATION_LON_SPACING_KM)
    assert np.isclose(POLE_SPACING_KM, 190.358 / 3, rtol=1e-3)
    # isotropic in physical distance (the old grid was 200.4 km x 106.6 km,
    # which made metrics.py's Euclidean index distance mix two lengths)
    km_lat = POLE_LAT_STEP_DEG * 111.32
    km_lon = POLE_LON_STEP_DEG * 111.32 * np.cos(np.deg2rad(70.0))
    assert np.isclose(km_lat, km_lon, rtol=1e-9)
    # the pre-registered d axis now spans the resolution transition
    rec = geometry_record()
    assert rec["d_axis_km"][1] < STATION_LON_SPACING_KM < rec["d_axis_km"][5]
    assert rec["tau_r_cells"] == 1.0                 # §8-v, pinned, unchanged


def test_placement_rule_is_centred_and_refinement_invariant():
    """The pair sits symmetric about the grid centre column on the centre row,
    at any grid width — so the sources stay in the array interior when the
    grid is refined (the inherited fixed col_a=1 did not)."""
    for d in (1, 2, 3, 5, 8):
        idx_a, idx_b = row_center_cols(d)
        row_a, col_a = divmod(idx_a, N_POLE_LON)
        row_b, col_b = divmod(idx_b, N_POLE_LON)
        assert row_a == row_b == N_POLE_LAT // 2
        assert col_b - col_a == d
        # centred: the pair's midpoint is within half a cell of grid centre
        assert abs((col_a + col_b) / 2.0 - (N_POLE_LON - 1) / 2.0) <= 0.5
    # and both sources sit inside the station array footprint in longitude
    tm = build_array_and_grid()
    idx_a, idx_b = row_center_cols(8)
    half_span_deg = 20.0 / 2.0
    for idx in (idx_a, idx_b):
        assert abs(tm.pole_lon[idx]) < half_span_deg


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
    idx_a, idx_b = row_center_cols(3)
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
    idx_a, idx_b = row_center_cols(3)
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
        row_center_cols(N_POLE_LON)  # d == grid width -> cannot fit a pair
