"""modeselect.py — complex Wax-Kailath mode selection (AIC / MDL).

Per SPEC §S.4 / archived build brief §6.5. MDL is the pinned consumer-facing
default (ROADMAP §8-viii, Wax & Kailath 1985: MDL is consistent, AIC
over-estimates even asymptotically); `criterion` stays an argument so B3 can
score both, which is B3's entire point. Log-domain throughout. K_hat clipped
to [1, K_MAX=6], with clip events counted and reported (SPEC §S.4).

EIGENVALUE FLOOR — re-expressed in TRACE-NORMALIZED units (ROADMAP §8,
2026-07-27 geometry-rescale ruling; vault `secs-gibf-viability-AT-DISPATCH.md`
§8-ii sign-off, "Ruling 2"). SPEC §S.4 originally pinned an ABSOLUTE floor of
1e-18 -- calibrated for the O(1) unit-power spectra the archived brief's own
desk-checks used. Experiment B's real ground-CSM eigenvalues live at ~1e-24
(tesla^2), entirely below 1e-18, so every eigenvalue clipped to the identical
floor value, the log-domain LLR degenerated to 0 for every k, and MDL/AIC
always selected k_hat=1 -- a floor-clipping degeneracy, not a mode-selection
result (ROADMAP §8, 2026-07-21 B3 finding). run_experiment_B.py's B3 runner
worked around this at the CALL SITE by pre-scaling eigenvalues by 2/trace(S)
before calling estimate_n_sources -- but that leaves every OTHER caller (and
any future geometry, which changes the CSM's physical scale again) exposed
to the identical bug. This is the third instance of the same root cause
(Tier 2's gap-conditioning statistic, §S.6.1's solver scale convention, now
this) -- so the fix belongs in the UNIT CONVENTION, not at another call site.
`estimate_n_sources` now normalizes eigenvalues by 2/trace(S) INTERNALLY,
before the (still-1e-18) floor is applied, so every caller gets a
floor that is relative to the CSM's own physical scale, automatically, with
no manual pre-scaling required. The Wax-Kailath LLR is provably invariant to
a uniform positive rescaling of all eigenvalues (log_geo shifts by ln(c),
ari scales by c, log_geo - ln(ari) is unchanged) EXCEPT where the floor
itself would otherwise have clipped -- so this changes no already-reported
result's content, only which spectra the floor spuriously degenerates.
"""

import numpy as np

EIG_FLOOR = 1e-18
K_MAX = 6
TRACE_NORM_TARGET = 2.0  # matches Tier 2's gap-conditioning / §S.6.1 convention


def _aic_mdl_arrays(eigvals, n_snapshots):
    """AIC(k)/MDL(k) for k = 0..M-1 candidate signal-subspace dimensions
    (archived brief §6.5 / Appendix A formulas, p = M - k).

    Eigenvalues are trace-normalized (x TRACE_NORM_TARGET / trace) BEFORE the
    absolute EIG_FLOOR is applied -- see the module docstring's 2026-07-27
    re-expression. The LLR (and hence AIC/MDL/k_hat) is unaffected by this
    rescaling except in spectra where the floor would otherwise have clipped
    real, non-negligible eigenvalues."""
    lam_raw = np.asarray(eigvals, dtype=np.float64)
    trace = float(np.sum(lam_raw))
    scale = (TRACE_NORM_TARGET / trace) if trace > 0 else 1.0
    lam = np.clip(lam_raw * scale, EIG_FLOOR, None)
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
