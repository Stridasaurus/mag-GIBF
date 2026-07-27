"""modeselect.py — complex Wax-Kailath mode selection (AIC / MDL).

Per SPEC §S.4 / archived build brief §6.5. MDL is the pinned consumer-facing
default (ROADMAP §8-viii, Wax & Kailath 1985: MDL is consistent, AIC
over-estimates even asymptotically); `criterion` stays an argument so B3 can
score both, which is B3's entire point. Log-domain throughout. K_hat clipped
to [1, K_MAX=6], with clip events counted and reported (SPEC §S.4).

EIGENVALUE FLOOR — RE-EXPRESSED IN TRACE-NORMALIZED UNITS, 2026-07-27
(SPEC §S.4 amendment; authorized by Strider's §8-ii sign-off ruling 2,
ROADMAP §8 2026-07-27). The pinned numeral `1e-18` is UNCHANGED; what changed
is the units it is denominated in. It was calibrated for an O(1) unit-power
scenario (Tier 1's convention), and applied to raw eigenvalues it is an
ABSOLUTE floor — so on Experiment B's real ground CSM, whose eigenvalues sit
at ~1e-24 tesla^2, every eigenvalue clipped to the identical floor value, the
log-domain LLR degenerated to 0 for every candidate k, and MDL/AIC reported
k_hat = 1 at 100% of cells (the 2026-07-21 B3 finding). That was the THIRD
mis-fire of one root cause (Card A Tier 2's gap-conditioning statistic;
§S.6.1's solver scale convention; then §S.4), so the fix is in the units, not
at the call site: eigenvalues are trace-normalized (lam * 2/trace) INSIDE this
module before the floor is applied, making `estimate_n_sources` scale-free by
construction. Any future rescaling of the CSM — including the 2026-07-27 grid
refinement, which moves the CSM scale again — leaves it correct.

Why this changes no content: the Wax-Kailath LLR
`N*p*(mean(log(tail)) - log(mean(tail)))` is provably invariant to a uniform
positive rescaling lam -> c*lam (mean(log) shifts by ln c, log(mean) shifts by
ln c, the difference is unchanged), and the penalty terms do not involve the
eigenvalues at all. So the normalization is a no-op on every full-rank
spectrum; it changes only WHICH eigenvalues the floor catches — and the floor's
only job is to keep `ln` finite on the exact zeros of a rank-deficient sample
CSM (n_snap < n_channels). `tests/test_modeselect.py` pins the invariance
across 48 decades, including the rank-deficient case.
"""

import numpy as np

EIG_FLOOR = 1e-18   # in TRACE-NORMALIZED units (lam * 2/trace); see docstring
K_MAX = 6


def _trace_normalize(lam):
    """lam * 2/trace — the repo-wide convention for making a covariance-derived
    statistic comparable across physical scales (repo CLAUDE.md; ROADMAP §8
    2026-07-19 gap-conditioning entry; SPEC §S.6.1 solver scale convention).
    Idempotent up to the factor 2 target sum, and a no-op for the LLR."""
    total = float(np.sum(lam))
    if not np.isfinite(total) or total <= 0.0:
        return lam
    return lam * (2.0 / total)


def _aic_mdl_arrays(eigvals, n_snapshots):
    """AIC(k)/MDL(k) for k = 0..M-1 candidate signal-subspace dimensions
    (archived brief §6.5 / Appendix A formulas, p = M - k)."""
    lam = np.asarray(eigvals, dtype=np.float64)
    lam = np.clip(_trace_normalize(np.clip(lam, 0.0, None)), EIG_FLOOR, None)
    lam = np.sort(lam)[::-1]
    M = lam.size
    N = n_snapshots
    aic = np.empty(M, dtype=np.float64)
    mdl = np.empty(M, dtype=np.float64)
    for k in range(M):
        p = M - k
        tail = lam[k:]
        log_geo = np.mean(np.log(tail))
        ari = np.mean(tail)
        llr = N * p * (log_geo - np.log(ari))  # <= 0
        aic[k] = -2.0 * llr + 2.0 * k * (2 * M - k)
        mdl[k] = -llr + 0.5 * k * (2 * M - k) * np.log(N)
    return aic, mdl


def estimate_n_sources(eigvals, n_snapshots, criterion="mdl", k_max=K_MAX):
    """argmin_k {AIC(k) or MDL(k)}, clipped to [1, k_max].

    Returns (k_hat, aic_array, mdl_array, was_clipped).
    """
    if criterion not in ("aic", "mdl"):
        raise ValueError(f"unknown criterion {criterion!r}; expected 'aic' or 'mdl'")
    aic, mdl = _aic_mdl_arrays(eigvals, n_snapshots)
    arr = aic if criterion == "aic" else mdl
    k_raw = int(np.argmin(arr))
    k_hat = k_raw
    clipped = False
    if k_hat < 1:
        k_hat = 1
        clipped = True
    elif k_hat > k_max:
        k_hat = k_max
        clipped = True
    return k_hat, aic, mdl, clipped
