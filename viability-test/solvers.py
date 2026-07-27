"""solvers.py — L2 baseline, per-mode GIBF (L1-IRLS + grid reduction), joint
MMV-L1 (L2,1 row-sparse). Per archived build brief §7 / SPEC §S.6/§S.1
(fairness protocol PINNED 2026-07-19).

Fairness invariants (do not violate):
  - ONE shared literal config object (`FAIRNESS_CONFIG`) for both sparse
    solvers; `min_active_factor`, `beta`, `weight_floor`, `p_norm`,
    `max_iter`, `tol` are the SAME values, read from the SAME object.
  - eps is tie-BY-RULE, not tie-by-value: eps = eps_frac * lambda_max(A W A^T)
    (underdetermined branch) / eps_frac * lambda_max(A^T A) (overdetermined),
    computed on the solver's OWN current weighted system each iteration.
    Realized eps is recorded and returned.
  - Scale convention: `solve_all` is the single entry point that applies the
    V/s (s = sqrt(lambda_1)) rescaling before calling any solver, and rescales
    intensities by s^2 after — so GIBF and MMV (and L2) can never accidentally
    receive different inputs (SPEC §S.6.1).
  - `assert np.isrealobj(A)` in every solver (B4's phantom-phase arm is out
    of this handoff's scope; never triggered here).

Resolved implementation ambiguity (documented, not a pre-registered value):
the archived brief's IRLS pseudocode checks `rel_change(a, a_prev) < tol` for
its stopping rule, but under grid reduction the active set shrinks every
iteration, so comparing `a` to the previous iterate's shape is ill-defined.
Substituted here with a dimension-agnostic relative change in the L1 (GIBF)
/ L2,1 (MMV) cost, which is well-defined regardless of grid reduction and is
a strictly more conservative reading of "converged" than the ill-defined
original. The non-increasing-cost termination (also ill-defined for the same
reason if compared naively) is applied to the cost value, which IS always
comparable across iterations irrespective of the active-set size.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    eps_frac: float = 0.01
    p_norm: float = 1.0
    weight_floor: float = 1e-10
    beta: float = 0.9
    max_iter: int = 50
    tol: float = 1e-4
    min_active_factor: int = 2
    grid_reduction: bool = True


# The one shared literal config object (SPEC §S.6.1). A pytest asserts both
# sparse solvers are exercised through this exact object.
FAIRNESS_CONFIG = SolverConfig()


def with_reduction(cfg, grid_reduction):
    """Return a copy of cfg with only grid_reduction changed — used to run
    the symmetric on/off pair without duplicating the other pinned fields."""
    return SolverConfig(eps_frac=cfg.eps_frac, p_norm=cfg.p_norm,
                         weight_floor=cfg.weight_floor, beta=cfg.beta,
                         max_iter=cfg.max_iter, tol=cfg.tol,
                         min_active_factor=cfg.min_active_factor,
                         grid_reduction=grid_reduction)


def _max_eig_psd(M):
    return float(np.linalg.eigvalsh(M)[-1])


def l2_minnorm(A, V, cfg=FAIRNESS_CONFIG):
    """Tikhonov min-norm per mode: a = A^T (A A^T + eps I)^-1 v, same eps
    rule, unweighted, on the unreduced system (context row, never
    adjudicated, never tuned; no grid-reduction concept)."""
    assert np.isrealobj(A)
    n_channels = A.shape[0]
    AAt = A @ A.T
    eps = cfg.eps_frac * _max_eig_psd(AAt)
    sol = np.linalg.solve(AAt + eps * np.eye(n_channels), V)
    Amat = A.T @ sol
    I = np.sum(np.abs(Amat) ** 2, axis=1)
    return dict(I=I, realized_eps=eps, Amat=Amat)


def _irls_gibf_mode(A, v, cfg):
    n_grid_full = A.shape[1]
    n_channels = A.shape[0]
    cols = np.arange(n_grid_full)
    Ak = A

    AAt0 = Ak @ Ak.T
    eps0 = cfg.eps_frac * _max_eig_psd(AAt0)
    a = Ak.T @ np.linalg.solve(AAt0 + eps0 * np.eye(n_channels), v)
    realized_eps = eps0
    prev_cost = np.inf
    n_iter = 0

    for it in range(cfg.max_iter):
        n_iter = it + 1
        w = (np.abs(a) + cfg.weight_floor) ** (2 - cfg.p_norm)
        n_active = len(cols)
        if 2 * n_active > n_channels:
            M = Ak @ (w[:, None] * Ak.T)
            eps = cfg.eps_frac * _max_eig_psd(M)
            a = w * (Ak.T @ np.linalg.solve(M + eps * np.eye(n_channels), v))
        else:
            AtA = Ak.T @ Ak
            eps = cfg.eps_frac * _max_eig_psd(AtA)
            a = np.linalg.solve(AtA + eps * np.diag(1.0 / w), Ak.T @ v)
        realized_eps = eps

        if cfg.grid_reduction:
            n_keep = max(1, int(cfg.beta * n_active))
            keep = np.argsort(np.abs(a))[::-1][:n_keep]
            cols, Ak, a = cols[keep], Ak[:, keep], a[keep]

        cost = float(np.sum(np.abs(a)))
        converged = False
        if it > 0:
            if cost > prev_cost * (1 + 1e-12):
                converged = True
            elif abs(prev_cost - cost) < cfg.tol * (abs(prev_cost) + 1e-300):
                converged = True
        if cfg.grid_reduction and len(cols) < cfg.min_active_factor * n_channels:
            converged = True
        prev_cost = cost
        if converged:
            break

    a_full = np.zeros(n_grid_full, dtype=complex)
    a_full[cols] = a
    return a_full, realized_eps, n_iter


def gibf_irls(A, V, cfg=FAIRNESS_CONFIG):
    """Per-mode independent L1-IRLS (Suzuki). Runs each eigenmode of V
    through its own IRLS with grid reduction, then accumulates intensity
    across modes: I(j) = sum_i |a_i(j)|^2."""
    assert np.isrealobj(A)
    n_grid_full = A.shape[1]
    K = V.shape[1]
    I_full = np.zeros(n_grid_full)
    eps_per_mode = []
    iters_per_mode = []
    for k in range(K):
        a_full, eps, n_iter = _irls_gibf_mode(A, V[:, k], cfg)
        I_full += np.abs(a_full) ** 2
        eps_per_mode.append(eps)
        iters_per_mode.append(n_iter)
    return dict(I=I_full, realized_eps=eps_per_mode, n_iter=iters_per_mode)


def mmv_l1(A, V, cfg=FAIRNESS_CONFIG):
    """Joint L2,1 row-sparse (shared support) MMV over all K modes at once.
    Same accumulation as GIBF: I(j) = sum_k |Abar[j,k]|^2 = ||Abar[j,:]||^2 —
    the ONLY experimental difference from GIBF is shared-support (this) vs
    independent-per-mode (GIBF) inversion of the identical V, A."""
    assert np.isrealobj(A)
    n_channels, n_grid_full = A.shape
    K = V.shape[1]
    cols = np.arange(n_grid_full)
    Ak = A

    AAt0 = Ak @ Ak.T
    eps0 = cfg.eps_frac * _max_eig_psd(AAt0)
    Abar = Ak.T @ np.linalg.solve(AAt0 + eps0 * np.eye(n_channels), V)
    realized_eps = eps0
    prev_cost = np.inf
    n_iter = 0

    for it in range(cfg.max_iter):
        n_iter = it + 1
        rownorm = np.linalg.norm(Abar, axis=1)
        w = (rownorm + cfg.weight_floor) ** (2 - cfg.p_norm)
        n_active = len(cols)
        if 2 * n_active > n_channels:
            M = Ak @ (w[:, None] * Ak.T)
            eps = cfg.eps_frac * _max_eig_psd(M)
            Abar = w[:, None] * (Ak.T @ np.linalg.solve(M + eps * np.eye(n_channels), V))
        else:
            AtA = Ak.T @ Ak
            eps = cfg.eps_frac * _max_eig_psd(AtA)
            Abar = np.linalg.solve(AtA + eps * np.diag(1.0 / w), Ak.T @ V)
        realized_eps = eps

        if cfg.grid_reduction:
            n_keep = max(1, int(cfg.beta * n_active))
            keep = np.argsort(rownorm)[::-1][:n_keep]
            cols, Ak, Abar = cols[keep], Ak[:, keep], Abar[keep, :]

        cost = float(np.sum(np.linalg.norm(Abar, axis=1)))
        converged = False
        if it > 0:
            if cost > prev_cost * (1 + 1e-12):
                converged = True
            elif abs(prev_cost - cost) < cfg.tol * (abs(prev_cost) + 1e-300):
                converged = True
        if cfg.grid_reduction and len(cols) < cfg.min_active_factor * n_channels:
            converged = True
        prev_cost = cost
        if converged:
            break

    Abar_full = np.zeros((n_grid_full, K), dtype=complex)
    Abar_full[cols, :] = Abar
    I_full = np.sum(np.abs(Abar_full) ** 2, axis=1)
    return dict(I=I_full, realized_eps=realized_eps, support=cols, n_iter=n_iter)


def solve_all(A, V, cfg=FAIRNESS_CONFIG):
    """Single entry point: applies the S.6.1 scale convention (V/s,
    s = sqrt(lambda_1) = ||V[:,0]||, since V's columns are already
    sqrt(lambda)-weighted and descending), runs L2 / GIBF / MMV with
    IDENTICAL A, V(scaled), cfg, then rescales intensities by s^2. This is
    the only call site run_experiment_B.py uses, so GIBF and MMV can never
    be fed different inputs by construction."""
    s = float(np.linalg.norm(V[:, 0])) if V.shape[1] > 0 else 1.0
    if s == 0.0:
        s = 1.0
    Vs = V / s

    out = {"scale_s": s}
    out["l2"] = l2_minnorm(A, Vs, cfg)
    out["gibf"] = gibf_irls(A, Vs, cfg)
    out["mmv"] = mmv_l1(A, Vs, cfg)
    for name in ("l2", "gibf", "mmv"):
        out[name] = dict(out[name])
        out[name]["I"] = out[name]["I"] * (s ** 2)
    return out
