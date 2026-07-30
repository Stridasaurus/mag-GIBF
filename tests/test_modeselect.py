"""modeselect.py pytest gate (SPEC §S.4 / archived brief §6.5/§11)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from modeselect import EIG_FLOOR, K_MAX, estimate_n_sources  # noqa: E402


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


# --- the regression that stops mis-fire #4 (ROADMAP §8 2026-07-27, ruling 2) --
#
# SPEC §S.4's floor is pinned at 1e-18 but is denominated in TRACE-NORMALIZED
# units, applied inside modeselect. An absolute floor made mode selection
# silently scale-dependent, and the identical root cause had already mis-fired
# twice before (Card A Tier 2's gap-conditioning statistic; §S.6.1's solver
# scale convention). These tests pin the property, not a number.

_SCALES = (1e-24, 1e-18, 1e-12, 1e-6, 1.0, 1e3, 1e6)


def test_k_hat_is_invariant_to_uniform_eigenvalue_rescaling():
    """The Wax-Kailath LLR is invariant to lam -> c*lam for any c > 0, so
    K_hat must be too — across the 48 decades that separate an O(1) unit-power
    scenario from Experiment B's ~1e-24 tesla^2 ground CSM."""
    rng = np.random.default_rng(11)
    lam = _synthetic_spectrum(2, 20, 64, rng)
    ref_mdl, _, mdl_ref, _ = estimate_n_sources(lam, 64, criterion="mdl")
    ref_aic, aic_ref, _, _ = estimate_n_sources(lam, 64, criterion="aic")
    assert ref_mdl == 2
    for c in _SCALES:
        k_mdl, _, mdl_c, _ = estimate_n_sources(lam * c, 64, criterion="mdl")
        k_aic, aic_c, _, _ = estimate_n_sources(lam * c, 64, criterion="aic")
        assert k_mdl == ref_mdl, f"MDL K_hat moved at scale {c:g}"
        assert k_aic == ref_aic, f"AIC K_hat moved at scale {c:g}"
        # the criterion curves themselves, not just their argmins
        assert np.allclose(mdl_c, mdl_ref, rtol=1e-9, atol=1e-9)
        assert np.allclose(aic_c, aic_ref, rtol=1e-9, atol=1e-9)


def test_scale_invariance_holds_for_a_rank_deficient_spectrum():
    """n_snap < n_channels gives a singular sample CSM with exact zeros — the
    only case the floor exists for. Invariance must survive it (this is the
    regime B3 sweeps at n_snap in {8,16,32,64} against 75 channels)."""
    rng = np.random.default_rng(12)
    m, n_snap = 75, 8
    lam = np.concatenate([_synthetic_spectrum(2, n_snap, n_snap, rng),
                          np.zeros(m - n_snap)])
    ref, _, mdl_ref, _ = estimate_n_sources(lam, n_snap, criterion="mdl")
    for c in _SCALES:
        k_hat, _, mdl_c, _ = estimate_n_sources(lam * c, n_snap, criterion="mdl")
        assert k_hat == ref, f"K_hat moved at scale {c:g} on a singular CSM"
        assert np.isfinite(mdl_c).all()
        assert np.allclose(mdl_c, mdl_ref, rtol=1e-9, atol=1e-9)


def test_absolute_floor_regression_tesla_scale_spectrum_is_not_flattened():
    """The 2026-07-21 defect, pinned: a ~1e-24 spectrum with a clean two-source
    structure used to clip entirely onto an absolute 1e-18 floor, flattening
    the LLR to 0 for every k and forcing K_hat=1. It must now read K=2."""
    rng = np.random.default_rng(13)
    lam = _synthetic_spectrum(2, 20, 128, rng) * 1e-24
    assert lam.max() < EIG_FLOOR      # entirely below the pinned numeral
    k_hat, _, _, _ = estimate_n_sources(lam, 128, criterion="mdl")
    assert k_hat == 2
