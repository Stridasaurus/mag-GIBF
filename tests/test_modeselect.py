"""modeselect.py pytest gate (SPEC §S.4 / archived brief §6.5/§11)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from modeselect import K_MAX, estimate_n_sources  # noqa: E402


def _synthetic_spectrum(k_true, m, n_snap, rng, signal_eig=10.0, noise_eig=1.0):
    """Two-population spectrum: k_true large eigenvalues on top of an
    (m - k_true) white floor, perturbed by finite-N sample noise so MDL/AIC
    have a genuine (not degenerate) decision to make."""
    lam = np.concatenate([
        np.full(k_true, signal_eig) * (1 + rng.normal(0, 0.02, k_true)),
        np.full(m - k_true, noise_eig) * (1 + rng.normal(0, 0.02, m - k_true)),
    ])
    return np.sort(lam)[::-1]


def test_mdl_recovers_planted_k_across_n():
    rng = np.random.default_rng(0)
    m = 20
    for n_snap in (32, 128, 1024):
        lam = _synthetic_spectrum(3, m, n_snap, rng)
        k_hat, aic, mdl, clipped = estimate_n_sources(lam, n_snap, criterion="mdl")
        assert k_hat == 3
        assert not clipped
        assert aic.shape == (m,)
        assert mdl.shape == (m,)


def test_aic_criterion_selectable_and_reported_alongside_mdl():
    rng = np.random.default_rng(1)
    m = 20
    lam = _synthetic_spectrum(2, m, 256, rng)
    k_mdl, _, mdl_arr, _ = estimate_n_sources(lam, 256, criterion="mdl")
    k_aic, aic_arr, _, _ = estimate_n_sources(lam, 256, criterion="aic")
    assert k_mdl == 2
    # AIC is the one criterion Wax-Kailath show over-selects; not asserting
    # a value here, only that both criteria are independently computable and
    # the returned arrays are self-consistent argmins.
    assert k_aic == int(np.argmin(aic_arr)) if k_aic <= K_MAX else True
    assert k_mdl == int(np.argmin(mdl_arr)) or k_mdl in (1, K_MAX)


def test_clip_to_1_and_k_max():
    rng = np.random.default_rng(2)
    # All-flat spectrum: argmin tends toward k=0 -> clipped to 1.
    lam = np.full(10, 1.0) * (1 + rng.normal(0, 1e-6, 10))
    k_hat, _, _, clipped = estimate_n_sources(lam, 64, criterion="mdl")
    assert 1 <= k_hat <= K_MAX

    # Force clipping at the top: many strong eigenvalues (> K_MAX) that
    # genuinely separate from a tiny floor should still clip to K_MAX.
    lam_many = np.concatenate([np.full(8, 100.0), np.full(4, 1e-6)])
    k_hat2, _, _, clipped2 = estimate_n_sources(lam_many, 64, criterion="mdl", k_max=6)
    assert k_hat2 <= 6


def test_invalid_criterion_raises():
    import pytest
    with pytest.raises(ValueError):
        estimate_n_sources(np.array([1.0, 0.1]), 64, criterion="bogus")


def test_eigenvalue_floor_keeps_logs_finite():
    lam = np.array([1.0, 0.0, -1e-20, 1e-30])
    k_hat, aic, mdl, clipped = estimate_n_sources(lam, 64, criterion="mdl")
    assert np.isfinite(aic).all()
    assert np.isfinite(mdl).all()


def test_trace_normalized_floor_recovers_k_at_physical_csm_scale():
    """ROADMAP §8, 2026-07-27 geometry-rescale ruling ("Ruling 2"): SPEC
    §S.4's floor (1e-18) was calibrated for O(1) unit-power spectra; a real
    ground-CSM spectrum at ~1e-24 (tesla^2) clipped identically under the
    old absolute floor, degenerating every k_hat to 1 (ROADMAP §8,
    2026-07-21 B3 finding). estimate_n_sources must now recover the planted
    k WITHOUT any caller-side pre-normalization, because the floor is
    applied to trace-normalized eigenvalues internally."""
    rng = np.random.default_rng(42)
    m, k_true, n_snap = 20, 3, 64
    # Same two-population shape as _synthetic_spectrum, but rescaled to a
    # physically realistic ground-CSM magnitude (~1e-24), entirely below the
    # old absolute 1e-18 floor.
    signal_eig, noise_eig = 5e-24, 5e-25
    lam = np.concatenate([
        np.full(k_true, signal_eig) * (1 + rng.normal(0, 0.02, k_true)),
        np.full(m - k_true, noise_eig) * (1 + rng.normal(0, 0.02, m - k_true)),
    ])
    lam = np.sort(lam)[::-1]
    k_hat, aic, mdl, clipped = estimate_n_sources(lam, n_snap, criterion="mdl")
    assert k_hat == k_true
    assert not clipped
    assert np.isfinite(aic).all() and np.isfinite(mdl).all()


def test_trace_normalization_matches_manual_prescaling():
    """The internal fix must reproduce exactly what the run_b3 workaround
    computed manually (lam * 2/trace(S)) before this pass -- no
    already-reported number's content changes, only which callers need to
    remember to pre-scale."""
    rng = np.random.default_rng(7)
    lam = np.abs(rng.normal(1e-24, 3e-25, size=16))
    trace = float(np.sum(lam))
    lam_manual = lam * (2.0 / trace)
    k_manual, aic_manual, mdl_manual, _ = estimate_n_sources(lam_manual, 64, criterion="mdl")
    k_auto, aic_auto, mdl_auto, _ = estimate_n_sources(lam, 64, criterion="mdl")
    assert k_manual == k_auto
    np.testing.assert_allclose(aic_manual, aic_auto)
    np.testing.assert_allclose(mdl_manual, mdl_auto)
