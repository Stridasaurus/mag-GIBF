"""simulate.py — Experiment B geometry, complex phasor snapshot simulation, CSM.

GEOMETRY (documented implementation choice, not a re-opened pre-registration):
no absolute B1/B3 array+grid geometry is pinned anywhere in ROADMAP.md,
SPEC_experiment_B.md, or EXPERIMENT_CARD_A.md — those pin phi/snr/d/N in
RELATIVE terms only ("d in grid cells", "N snapshots"). The only concrete
geometry on record is the archived build brief's `config/base.yaml` (§5.1):
5x5=25-station array (lat_span 12 deg, lon_span 20 deg), 11x11=121-pole SECS
grid (lat_span 18 deg, lon_span 28 deg), both centered at the "high" latitude
preset (70 deg). SPEC §S.9's supersession list only touches role/|rho|/tau_r/
AIC-MDL/B5 — geometry is not superseded, so this is inherited unchanged.
Card A Tier 2's meridional chain is a DIFFERENT experiment (continuum FLR
forward model); it is not inherited here. Recorded in every run manifest.

The pole grid is offset by half its own lat/lon spacing from the array's
center so no station and pole share an exact (lat, lon) — Gate V's V1
coincidence guard (transfer.py) forbids exact overlap and the naive centered
grids collide exactly at the shared center point.

Two invariants honoured here (ROADMAP §2 / repo CLAUDE.md):
  - A stays real (asserted downstream in every solver, not here).
  - Snapshots are COMPLEX frequency-bin phasors (CN(0, power) draws), never
    raw real time samples -> CSM is genuinely complex Hermitian.
"""

import numpy as np

from transfer import build_transfer_matrix

CENTER_LAT = 70.0
CENTER_LON = 0.0

N_STATION_LAT, N_STATION_LON = 5, 5
STATION_LAT_SPAN, STATION_LON_SPAN = 12.0, 20.0

N_POLE_LAT, N_POLE_LON = 11, 11
POLE_LAT_SPAN, POLE_LON_SPAN = 18.0, 28.0
GRID_SHAPE = (N_POLE_LAT, N_POLE_LON)


def _linspace_centered(center, span, n):
    return center + np.linspace(-span / 2.0, span / 2.0, n)


def build_array_and_grid(row_normalisation="none", cache_dir=None):
    """Build the pinned-choice Experiment-B station array + SECS pole grid
    and the real transfer matrix A (DF-only, per Gate V / transfer.py)."""
    st_lat_ax = _linspace_centered(CENTER_LAT, STATION_LAT_SPAN, N_STATION_LAT)
    st_lon_ax = _linspace_centered(CENTER_LON, STATION_LON_SPAN, N_STATION_LON)
    st_lat, st_lon = np.meshgrid(st_lat_ax, st_lon_ax, indexing="ij")

    po_lat_ax = _linspace_centered(CENTER_LAT, POLE_LAT_SPAN, N_POLE_LAT)
    po_lon_ax = _linspace_centered(CENTER_LON, POLE_LON_SPAN, N_POLE_LON)
    # Offset by half the pole grid's own spacing (V1 coincidence guard).
    po_lat_ax = po_lat_ax + 0.5 * (po_lat_ax[1] - po_lat_ax[0])
    po_lon_ax = po_lon_ax + 0.5 * (po_lon_ax[1] - po_lon_ax[0])
    po_lat, po_lon = np.meshgrid(po_lat_ax, po_lon_ax, indexing="ij")

    tm = build_transfer_matrix(
        station_lat=st_lat.ravel(), station_lon=st_lon.ravel(),
        pole_lat=po_lat.ravel(), pole_lon=po_lon.ravel(),
        current_types=("divfree",),
        row_normalisation=row_normalisation, cache_dir=cache_dir,
    )
    return tm


def centroid_grid_index(tm):
    """Grid index nearest the array centroid (CENTER_LAT, CENTER_LON) —
    used for the single-source exact-recovery gate."""
    d2 = (tm.pole_lat - CENTER_LAT) ** 2 + (tm.pole_lon - CENTER_LON) ** 2
    return int(np.argmin(d2))


def grid_rc(idx, grid_shape=GRID_SHAPE):
    """Row-major (row, col) for a flat grid index."""
    return divmod(idx, grid_shape[1])


def row_center_cols(col_a, d, grid_shape=GRID_SHAPE):
    """Two grid indices `d` columns apart along the grid's center row —
    the fixed two-source placement used for B1/B3/mini-pilot. col_a is the
    fixed left-source column; the right source sits at col_a + d, which must
    stay inside [0, n_lon-1] for every d this handoff uses (max d=8)."""
    n_lat, n_lon = grid_shape
    row = n_lat // 2
    col_b = col_a + d
    if not (0 <= col_b < n_lon):
        raise ValueError(f"d={d} places source 2 at col {col_b}, outside "
                          f"grid width {n_lon} (col_a={col_a})")
    idx_a = row * n_lon + col_a
    idx_b = row * n_lon + col_b
    return idx_a, idx_b


def cn(rng, shape, power):
    """Circularly-symmetric complex normal CN(0, power): real & imag each
    N(0, power/2), so E[|z|^2] = power (archived brief §6.3)."""
    sigma = np.sqrt(max(power, 0.0) / 2.0)
    return rng.normal(0.0, sigma, size=shape) + 1j * rng.normal(0.0, sigma, size=shape)


def simulate_snapshots(tm, active_indices, powers, rng, n_snap, snr_db,
                        coherence=None, current_type="divfree"):
    """Complex frequency-bin phasor snapshots (§0.2 rule 2 / repo CLAUDE.md).

    active_indices: grid indices (into tm's pole grid) of the active sources.
    powers: per-source power (equal-power pairs used throughout this handoff).
    coherence: None -> incoherent (each source ~ CN(0, power) independently
      across snapshots); complex rho -> source 2 = rho*source1 +
      sqrt(1-|rho|^2)*CN(0, power2), rho FIXED across snapshots (requires
      exactly 2 active sources).

    Returns (X, X_clean): both (n_channels, n_snap) complex. SNR = mean
    per-channel signal power / mean per-channel noise power (archived brief
    §6.3 convention). snr_db=None -> noise-free deterministic (no draw at
    all; used by the single-source gate's tier (i))."""
    n_src = len(active_indices)
    col0 = tm.col_slice(current_type).start
    cols = np.array([col0 + i for i in active_indices])

    if coherence is None:
        s = np.stack([cn(rng, (n_snap,), p) for p in powers], axis=0)
    else:
        if n_src != 2:
            raise ValueError("coherent pair model requires exactly 2 sources")
        rho = coherence
        s1 = cn(rng, (n_snap,), powers[0])
        s2 = rho * s1 + np.sqrt(max(0.0, 1.0 - abs(rho) ** 2)) * cn(rng, (n_snap,), powers[1])
        s = np.stack([s1, s2], axis=0)

    A_active = tm.A[:, cols]
    X_clean = A_active @ s

    if snr_db is None:
        return X_clean.copy(), X_clean

    sig_power = float(np.mean(np.abs(X_clean) ** 2))
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise = cn(rng, X_clean.shape, noise_power)
    X = X_clean + noise
    return X, X_clean


def build_csm(X):
    """S = (1/N) X X^H — complex Hermitian PSD (archived brief §6.4)."""
    n_snap = X.shape[1]
    return (X @ X.conj().T) / n_snap


def eigendecompose(S):
    """Hermitian eigendecomposition, eigenvalues descending, tiny negatives
    clipped to 0 (numerical PSD guard)."""
    lam, U = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    lam = np.clip(lam[order], 0.0, None)
    U = U[:, order]
    return lam, U


def eigenmodes(lam, U, k):
    """V = [sqrt(lam_1) u_1, ..., sqrt(lam_k) u_k] (Suzuki Eq. 3). Carries
    the eigenvalue magnitude — never the unit eigenvector."""
    return U[:, :k] * np.sqrt(lam[:k])[None, :]
