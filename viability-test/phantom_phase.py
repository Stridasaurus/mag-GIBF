"""phantom_phase.py — Experiment B4's only complex-kernel code (SPEC S.15).

ROADMAP S2 invariant: the transfer matrix A is REAL, and every standard solver
asserts it (`assert np.isrealobj(A)` in solvers.py). B4 is the one deliberate
exception: it asks what a kernel carrying a propagation phase would buy GIBF.
So the complex-capable code lives HERE, apart from solvers.py, and the
standard solvers are left untouched and still reject a complex A.

Two things:

  phantom_phase_matrix(tm, wavelength_km)
      A_phase = A (.) exp(i Phi),  Phi[m, j] = 2 pi r[m, j] / lambda,
      r[m, j] = straight-line distance station m (ground radius) to pole j
      (shell radius). The three field components of a station share its phase.
      A phase common to one pole (a column) is a gauge and does not change
      intensities |a_j|^2.

  gibf_irls_complex(A, V, cfg)
      Suzuki-style per-mode L1-IRLS with grid reduction, identical in every
      step to solvers.gibf_irls except that adjoints are Hermitian
      (`.conj().T`), so it accepts a complex A. On a real A it must agree with
      solvers.gibf_irls (tests/test_b4_b6b.py pins that).
"""

import numpy as np

from solvers import FAIRNESS_CONFIG

C_KM_S = 299_792.458          # speed of light in vacuum, km/s


def _unit(lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def station_pole_distance_km(tm):
    """(n_stations, n_poles) straight-line distances in km."""
    st = tm.r * _unit(tm.station_lat, tm.station_lon)          # metres
    po = tm.RI * _unit(tm.pole_lat, tm.pole_lon)
    d = np.linalg.norm(st[:, None, :] - po[None, :, :], axis=2)
    return d / 1.0e3


def phantom_phase_matrix(tm, wavelength_km):
    """Complex phantom kernel A (.) exp(i Phi) for the DF-only grid."""
    if tm.types != ("divfree",):
        raise ValueError("phantom kernel is defined for the DF-only kernel")
    phi = 2.0 * np.pi * station_pole_distance_km(tm) / float(wavelength_km)
    phi_rows = np.tile(phi, (3, 1))        # rows are [Ge; Gn; Gr] stacked
    if phi_rows.shape != tm.A.shape:
        raise ValueError(f"phase {phi_rows.shape} vs kernel {tm.A.shape}")
    return tm.A * np.exp(1j * phi_rows)


def _max_eig_herm(M):
    return float(np.linalg.eigvalsh(M)[-1])


def _irls_gibf_mode_complex(A, v, cfg):
    n_grid_full = A.shape[1]
    n_channels = A.shape[0]
    cols = np.arange(n_grid_full)
    Ak = A

    AAt0 = Ak @ Ak.conj().T
    eps0 = cfg.eps_frac * _max_eig_herm(AAt0)
    a = Ak.conj().T @ np.linalg.solve(AAt0 + eps0 * np.eye(n_channels), v)
    realized_eps = eps0
    prev_cost = np.inf
    n_iter = 0
    stop_reason = "max_iter"

    for it in range(cfg.max_iter):
        n_iter = it + 1
        w = (np.abs(a) + cfg.weight_floor) ** (2 - cfg.p_norm)
        n_active = len(cols)
        if 2 * n_active > n_channels:
            M = Ak @ (w[:, None] * Ak.conj().T)
            eps = cfg.eps_frac * _max_eig_herm(M)
            a = w * (Ak.conj().T @ np.linalg.solve(M + eps * np.eye(n_channels), v))
        else:
            AtA = Ak.conj().T @ Ak
            eps = cfg.eps_frac * _max_eig_herm(AtA)
            a = np.linalg.solve(AtA + eps * np.diag(1.0 / w), Ak.conj().T @ v)
        realized_eps = eps

        if cfg.grid_reduction:
            n_keep = max(1, int(cfg.beta * n_active))
            keep = np.argsort(np.abs(a))[::-1][:n_keep]
            cols, Ak, a = cols[keep], Ak[:, keep], a[keep]

        cost = float(np.sum(np.abs(a)))
        converged = False
        if cfg.stop_early and it > 0:
            if cost > prev_cost * (1 + 1e-12):
                converged = True
                stop_reason = "cost_increase"
            elif abs(prev_cost - cost) < cfg.tol * (abs(prev_cost) + 1e-300):
                converged = True
                stop_reason = "relative_tolerance"
        if (cfg.stop_early and cfg.grid_reduction
                and len(cols) < cfg.min_active_factor * n_channels):
            converged = True
            stop_reason = "active_set_floor"
        prev_cost = cost
        if converged:
            break

    a_full = np.zeros(n_grid_full, dtype=complex)
    a_full[cols] = a
    return a_full, realized_eps, n_iter, stop_reason


def gibf_irls_complex(A, V, cfg=FAIRNESS_CONFIG):
    """Per-mode independent L1-IRLS (Suzuki) for a real OR complex kernel."""
    n_grid_full = A.shape[1]
    K = V.shape[1]
    I_full = np.zeros(n_grid_full)
    eps_per_mode, iters_per_mode, stop_reasons = [], [], []
    for k in range(K):
        a_full, eps, n_iter, stop_reason = _irls_gibf_mode_complex(A, V[:, k], cfg)
        I_full += np.abs(a_full) ** 2
        eps_per_mode.append(eps)
        iters_per_mode.append(n_iter)
        stop_reasons.append(stop_reason)
    return dict(I=I_full, realized_eps=eps_per_mode, n_iter=iters_per_mode,
                stop_reason=stop_reasons)


def gibf_irls_scaled(A, V, cfg=FAIRNESS_CONFIG):
    """The S.6.1 scale convention (V/s, intensities x s^2) around the complex
    solver, matching solvers.solve_all for the GIBF arm."""
    s = float(np.linalg.norm(V[:, 0])) if V.shape[1] > 0 else 1.0
    if s == 0.0:
        s = 1.0
    out = gibf_irls_complex(A, V / s, cfg)
    out["I"] = out["I"] * (s ** 2)
    return out
