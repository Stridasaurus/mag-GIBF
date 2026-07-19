"""Card A Tier 2 pytest gate (handoff.md workflow step 4).

Covers, at minimum (per the handoff): the verdict-aggregation logic given a
synthetic kappa_ground surface, and that the A1-pinned permutation test
correctly reproduces against a known archived floor distribution.

Run: conda run -n mhd-env python -m pytest tests/test_tier2.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from tier2_flr_coherence import (ALPHA, aggregate_verdict,  # noqa: E402
                                 batched_csm_stats, calibrate_sigma_bg,
                                 classify_cell, coherence_pair_indices,
                                 flr_profile, perm_test_greater, pole_grid,
                                 synthetic_stations)
from tier1_flr_coherence import im_frac, kappa  # noqa: E402

FLOORS_NPZ = (Path(__file__).resolve().parent.parent / "results" /
              "A_flr_coherence" / "floor_distributions.npz")


@pytest.fixture(scope="module")
def floors():
    d = np.load(FLOORS_NPZ)
    return dict(kappa_phi0=d["kappa_phi0_coherent_N64"],
                gap_incoherent=d["gap_incoherent_N64"],
                imfrac_incoherent=d["imfrac_incoherent_N64"],
                imfrac_phi0=d["imfrac_phi0_coherent_N64"],
                kappa_incoherent=d["kappa_incoherent_N64"])


PINS = dict(imfrac_holds=0.3941, imfrac_fails=0.1369)


def _trials(kappa_mean, imfrac_mean, gap_mean=0.5, n=400, seed=1):
    """Synthetic per-trial arrays with the requested means (tight spread)."""
    rng = np.random.default_rng(seed)
    return dict(kappa=np.clip(kappa_mean + 0.01 * rng.standard_normal(n), 0, 1),
                gap=np.abs(gap_mean + 0.01 * rng.standard_normal(n)),
                imfrac=np.clip(imfrac_mean + 0.01 * rng.standard_normal(n), 0, 1),
                trace=np.full(n, 2.0))


# ------------------------------------------------- permutation test (A1 pin)

def test_perm_test_reproduces_archived_floor_separation(floors):
    """Known separation in the audit-A2 archive: the incoherent kappa floor
    (mean 0.283) is significantly above the phi=0 coherent kappa floor
    (mean 0.012); a floor tested against itself is indistinguishable."""
    p_sep = perm_test_greater(floors["kappa_incoherent"],
                              floors["kappa_phi0"], [0, 1])
    assert p_sep < ALPHA
    p_self = perm_test_greater(floors["kappa_phi0"], floors["kappa_phi0"],
                               [0, 2])
    assert p_self >= ALPHA
    # direction matters: the lower floor is NOT significantly above the higher
    p_rev = perm_test_greater(floors["kappa_phi0"],
                              floors["kappa_incoherent"], [0, 3])
    assert p_rev >= ALPHA


def test_perm_test_deterministic_given_seed(floors):
    a = perm_test_greater(floors["imfrac_incoherent"], floors["imfrac_phi0"],
                          [7, 7])
    b = perm_test_greater(floors["imfrac_incoherent"], floors["imfrac_phi0"],
                          [7, 7])
    assert a == b


# ------------------------------------------------- verdict aggregation (§8)

def test_verdict_holds_is_existence_claim(floors):
    """One holding cell among failing/marginal ones -> H-A holds."""
    hold = classify_cell(_trials(0.45, 0.55), floors, PINS, [1, 0])
    fail = classify_cell(_trials(0.02, 0.02), floors, PINS, [1, 1])
    marg = classify_cell(_trials(0.20, 0.25), floors, PINS, [1, 2])
    assert hold["cell_class_raw"] == "holds"
    assert fail["cell_class_raw"] == "fails"
    assert marg["cell_class_raw"] == "marginal"
    assert aggregate_verdict([c["cell_class_raw"]
                              for c in (hold, fail, marg)]) == "holds"


def test_verdict_fails_is_universal(floors):
    fails = [classify_cell(_trials(0.02, 0.02, seed=s), floors, PINS, [2, s])
             for s in range(3)]
    classes = [c["cell_class_raw"] for c in fails]
    assert aggregate_verdict(classes) == "fails"
    classes[-1] = "marginal"
    assert aggregate_verdict(classes) == "marginal"


def test_holds_requires_coprimary_concurrence(floors):
    """kappa above 0.30 with the co-primary below 0.394 must NOT hold."""
    c = classify_cell(_trials(0.45, 0.20), floors, PINS, [3, 0])
    assert not c["holds"]
    assert c["cell_class_raw"] == "marginal"


def test_fail_arm_is_gap_conditioned(floors):
    """A cell whose kappa sits at the INCOHERENT floor level (0.283 —
    'marginal' by raw band) but is floor-indistinguishable in the co-primary:
    near-degenerate conditioning must hand the verdict to the co-primary
    (fails); rank-1-like conditioning must not (kappa arm blocks: 0.283 is
    neither < 0.10 nor indistinguishable from the phi=0 coherent floor)."""
    n = 400
    rng = np.random.default_rng(5)
    trials = dict(
        kappa=np.clip(0.283 + 0.01 * rng.standard_normal(n), 0, 1),
        imfrac=floors["imfrac_incoherent"] + 0.0005 * rng.standard_normal(n),
        # gap at the incoherent floor's own level -> near-degenerate even
        # after trace normalisation (trace fixed at Tier 1's expected 2.0)
        gap=np.abs(np.median(floors["gap_incoherent"]) +
                   0.01 * rng.standard_normal(n)),
        trace=np.full(n, 2.0),
    )
    c = classify_cell(trials, floors, PINS, [4, 0])
    assert c["near_deg_raw"] and c["near_deg_scaled"]
    assert c["cell_class_raw"] == "fails" and c["cell_class_scaled"] == "fails"
    # same statistics but a decisively rank-1-like gap: kappa arm now blocks
    trials2 = dict(trials, gap=np.abs(10.0 + 0.1 * rng.standard_normal(n)))
    c2 = classify_cell(trials2, floors, PINS, [4, 1])
    assert not c2["near_deg_scaled"]
    assert c2["cell_class_scaled"] == "marginal"


# ------------------------------------------------- model building blocks

def test_batched_stats_match_tier1_scalar_stats():
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((4, 8, 64)) +
         1j * rng.standard_normal((4, 8, 64))) / np.sqrt(2.0)
    kap, gap, imf, trace = batched_csm_stats(x)
    for t in range(4):
        s = x[t] @ x[t].conj().T / 64
        w, v = np.linalg.eigh(s)
        assert kap[t] == pytest.approx(kappa(v[:, -1]), abs=1e-12)
        assert imf[t] == pytest.approx(im_frac(s), abs=1e-12)
        assert gap[t] == pytest.approx(float(w[-1] - w[-2]), abs=1e-12)
        assert trace[t] == pytest.approx(float(np.trace(s).real), abs=1e-12)


def test_pinned_geometry_and_coincidence_offset():
    """10 stations centered 68.0N at 20.0E; pole grid at half-station spacing
    with 2-spacing margins, lon offset half a pole spacing (V1 guard)."""
    lat, lon = synthetic_stations(100.0)
    assert lat.size == 10 and np.allclose(lon, 20.0)
    assert lat.mean() == pytest.approx(68.0)
    assert np.diff(lat) == pytest.approx(100.0 / 111.19)
    plat, plon, plat_1d, plon_1d, meridian = pole_grid(lat, lon, 6.0, 68.0)
    assert meridian == 20.0
    d_deg = 100.0 / 111.19
    assert np.diff(plat_1d) == pytest.approx(d_deg / 2.0)
    assert plat_1d.min() == pytest.approx(lat.min() - 2 * d_deg)
    assert plat_1d.max() == pytest.approx(lat.max() + 2 * d_deg, abs=1e-6)
    # no pole longitude on the station meridian, by half-spacing offset
    dlon = (d_deg / 2.0) / np.cos(np.radians(68.0))
    assert np.min(np.abs(plon_1d - 20.0)) == pytest.approx(dlon / 2.0)
    assert np.all(np.abs(plon_1d - 20.0) <= 6.0)


def test_rho_calibration_hits_target():
    """sigma_bg must realize the pinned |rho| definition exactly at the pair
    nearest x_r +/- one resonance half-width."""
    lat, lon = synthetic_stations(150.0)
    plat, plon, plat_1d, plon_1d, meridian = pole_grid(lat, lon, 6.0, 68.0)
    for rho in (0.70, 0.85, 0.95):
        c, x_w = flr_profile(plat, plon, 68.3, 10.0, 0.10, meridian)
        i, j = coherence_pair_indices(plat_1d, plon_1d, 68.3, x_w, meridian)
        assert i != j
        sig = calibrate_sigma_bg(c, i, j, rho)
        realized = abs(c[i] * np.conj(c[j])) / np.sqrt(
            (abs(c[i]) ** 2 + sig ** 2) * (abs(c[j]) ** 2 + sig ** 2))
        assert realized == pytest.approx(rho, abs=1e-12)


def test_flr_phase_sweeps_through_resonance():
    """arg R must sweep ~0 -> -90 -> -180 deg through the resonance (the
    Tier 1 physics, reused not reinvented)."""
    lat, lon = synthetic_stations(100.0)
    plat, plon, plat_1d, plon_1d, meridian = pole_grid(lat, lon, 6.0, 68.0)
    c, x_w = flr_profile(plat_1d, np.full_like(plat_1d, meridian), 68.0,
                         10.0, 0.10, meridian)
    ph = np.degrees(np.angle(c))
    assert ph[0] == pytest.approx(-180.0, abs=15.0)
    assert ph[-1] == pytest.approx(0.0, abs=15.0)
    i0 = int(np.argmin(np.abs(plat_1d - 68.0)))
    assert ph[i0] == pytest.approx(-90.0, abs=10.0)
