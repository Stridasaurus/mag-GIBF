"""modeselect.py — complex Wax-Kailath mode selection (AIC / MDL).

Per SPEC §S.4 / archived build brief §6.5. MDL is the pinned consumer-facing
default (ROADMAP §8-viii, Wax & Kailath 1985: MDL is consistent, AIC
over-estimates even asymptotically); `criterion` stays an argument so B3 can
score both, which is B3's entire point. Log-domain throughout; eigenvalues
floored at 1e-18 before `ln` (SPEC pin). K_hat clipped to [1, K_MAX=6], with
clip events counted and reported (SPEC §S.4).
"""

import numpy as np

EIG_FLOOR = 1e-18
K_MAX = 6


def _aic_mdl_arrays(eigvals, n_snapshots):
    """AIC(k)/MDL(k) for k = 0..M-1 candidate signal-subspace dimensions
    (archived brief §6.5 / Appendix A formulas, p = M - k)."""
    lam = np.clip(np.asarray(eigvals, dtype=np.float64), EIG_FLOOR, None)
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
