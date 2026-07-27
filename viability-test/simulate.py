"""simulate.py — Experiment B geometry, complex phasor snapshot simulation, CSM.

GEOMETRY — REFINED 2026-07-27 (the "geometry-rescale pass"; authorized by
Strider's §8-ii sign-off ruling 1, ROADMAP §8 2026-07-27).

*What moved and why.* No absolute B1/B3 array+grid geometry is pinned anywhere
in ROADMAP.md, SPEC_experiment_B.md, or EXPERIMENT_CARD_A.md — those pin
phi/snr/d/|rho|/N in RELATIVE terms only ("d in grid cells", "N snapshots").
The 2026-07-21 kickoff run therefore inherited the archived build brief's
`config/base.yaml` geometry unchanged (5x5=25-station array, 11x11=121-pole
SECS grid over lat_span 18 deg / lon_span 28 deg at the "high" 70 deg preset)
and flagged it as a documented, NON-pre-registered choice. At that geometry one
pole-grid column step was 2.8 deg lon = 106.6 km at 70 deg N, so the
pre-registered confirmatory coordinate d=3 was a 320 km physical separation —
well above the array's sampling limit — and every solver recovered both sources
exactly on every trial (delta_r_bar = 0.000, P_sep = 1.00, gap SD = 0.0000): a
degenerate cell that NO n_trials could power. Strider ruled that the geometry
moves and every pre-registered coordinate stays pinned (d=3, phi=90 deg,
snr in {5,10}, |rho|=0.85, n_snap=64, tau_r=1, reduction ON are untouched).

*The rule that sets the new grid (fixed before any re-run; no result was
consulted in choosing it).* The confirmatory coordinate d=3 is pinned to the
array's along-separation-axis station spacing — the scale below which a ground
magnetometer array cannot sample horizontal structure:

    d=3  ==  STATION_LON_SPACING_KM  =  190.4 km   (5 stations over 20 deg lon
                                                    at 70 deg N)
    =>  POLE_SPACING_KM = 190.4 / 3 = 63.5 km      (image-grid oversampling x3)

An imaging grid is deliberately oversampled relative to the resolution element
(as a beamforming scan grid is) so that peak position is not quantization-
limited; x3 is the standard low end of that convention. The consequence is that
the pre-registered d axis {1,2,3,5,8} now SPANS the array's resolution
transition — {0.33, 0.67, 1.0, 1.67, 2.67} x station spacing = {63, 127, 190,
317, 508} km — instead of stepping over it in one jump (the old axis was
{107, 213, 320, 533, 853} km, i.e. d=1 below any resolvable scale and d>=3
already saturated).

*Also fixed by the same refinement (both consequences, not free choices):*
  - The grid is now ISOTROPIC in physical distance (63.5 km in both lat and
    lon). The old grid was 200 km x 107 km per cell, so `metrics.py`'s
    Euclidean index distance — which sets tau_r = 1 "grid cell" (ROADMAP §8-v)
    and delta_r_bar — mixed two different physical lengths along its two axes.
  - Source placement is now CENTRED: the pair sits symmetric about the grid's
    centre column on the centre row (`row_center_cols`). The old fixed
    `col_a = 1` was ~1 cell inside the array's western edge on an 11-wide grid;
    on the refined 17-wide grid the same literal index would place both sources
    OUTSIDE the array footprint. Centring is the only refinement-invariant
    rule. Note the direction of this one: it moves sources toward the array's
    most sensitive region, i.e. ceiling-ward, partially opposing the
    refinement — recorded so the attribution stays legible.

*Extents and conditioning.* The inherited imaging extents are preserved to
within one cell (lat 18 deg -> 33 poles x 63.5 km = 18.2 deg; lon 28 deg -> 17
poles x 63.5 km = 26.7 deg), so the imaged REGION is unchanged and only the
cell size moves. Pole count goes 121 -> 561, so overcompleteness against the
75 channels goes 1.6x -> 7.5x (adjacent-column correlation ~0.997); recorded
because a reviewer will ask.

*What did NOT move.* The station array (5x5, 12 deg x 20 deg, 70 deg N/0 deg E)
is untouched — the ruling refines the SECS grid. Card A Tier 2's meridional
chain is a DIFFERENT experiment (continuum FLR forward model) and is not
inherited here. Every field above is recorded structurally in each run
manifest.

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

# --- station array: UNCHANGED from the inherited base geometry -------------
N_STATION_LAT, N_STATION_LON = 5, 5
STATION_LAT_SPAN, STATION_LON_SPAN = 12.0, 20.0

# --- the refinement rule (see module docstring) ----------------------------
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = KM_PER_DEG_LAT * float(np.cos(np.deg2rad(CENTER_LAT)))
STATION_LON_SPACING_KM = (STATION_LON_SPAN / (N_STATION_LON - 1)) * KM_PER_DEG_LON
CONFIRMATORY_D = 3                      # pre-registered (§8-iii); NOT moved
GRID_OVERSAMPLING = CONFIRMATORY_D      # d=3 == one station spacing
POLE_SPACING_KM = STATION_LON_SPACING_KM / GRID_OVERSAMPLING

POLE_LAT_STEP_DEG = POLE_SPACING_KM / KM_PER_DEG_LAT
POLE_LON_STEP_DEG = POLE_SPACING_KM / KM_PER_DEG_LON

# Odd pole counts preserving the inherited imaging extents (18 deg lat,
# 28 deg lon) to within one cell — see module docstring.
LEGACY_POLE_LAT_SPAN, LEGACY_POLE_LON_SPAN = 18.0, 28.0


def _odd_count_for_span(span_deg, step_deg):
    """Odd pole count whose (n-1)*step best matches the inherited span."""
    intervals = span_deg / step_deg
    even = int(round(intervals / 2.0)) * 2
    return max(even, 2) + 1


N_POLE_LAT = _odd_count_for_span(LEGACY_POLE_LAT_SPAN, POLE_LAT_STEP_DEG)   # 33
N_POLE_LON = _odd_count_for_span(LEGACY_POLE_LON_SPAN, POLE_LON_STEP_DEG)   # 17
POLE_LAT_SPAN = (N_POLE_LAT - 1) * POLE_LAT_STEP_DEG
POLE_LON_SPAN = (N_POLE_LON - 1) * POLE_LON_STEP_DEG
GRID_SHAPE = (N_POLE_LAT, N_POLE_LON)

PLACEMENT_RULE = ("two-source pair on the grid centre row, symmetric about the "
                  "grid centre column (refinement-invariant; supersedes the "
                  "inherited fixed col_a=1)")


def geometry_record():
    """Structural, machine-readable geometry pin for run manifests (SPEC §S.5).
    Freeform notes go stale; these fields are the geometry."""
    return dict(
        center_lat=CENTER_LAT, center_lon=CENTER_LON,
        n_station_lat=N_STATION_LAT, n_station_lon=N_STATION_LON,
        station_lat_span_deg=STATION_LAT_SPAN, station_lon_span_deg=STATION_LON_SPAN,
        station_lon_spacing_km=round(STATION_LON_SPACING_KM, 3),
        station_lat_spacing_km=round(
            (STATION_LAT_SPAN / (N_STATION_LAT - 1)) * KM_PER_DEG_LAT, 3),
        n_pole_lat=N_POLE_LAT, n_pole_lon=N_POLE_LON,
        n_poles=N_POLE_LAT * N_POLE_LON,
        pole_spacing_km=round(POLE_SPACING_KM, 3),
        pole_lat_step_deg=round(POLE_LAT_STEP_DEG, 6),
        pole_lon_step_deg=round(POLE_LON_STEP_DEG, 6),
        pole_lat_span_deg=round(POLE_LAT_SPAN, 4),
        pole_lon_span_deg=round(POLE_LON_SPAN, 4),
        confirmatory_d=CONFIRMATORY_D,
        confirmatory_d_km=round(CONFIRMATORY_D * POLE_SPACING_KM, 3),
        grid_oversampling_factor=GRID_OVERSAMPLING,
        d_axis_km={d: round(d * POLE_SPACING_KM, 1) for d in (1, 2, 3, 5, 8)},
        tau_r_cells=1.0, tau_r_km=round(POLE_SPACING_KM, 3),
        overcompleteness=round(N_POLE_LAT * N_POLE_LON /
                               (3.0 * N_STATION_LAT * N_STATION_LON), 3),
        placement_rule=PLACEMENT_RULE,
        refinement_rule=("d=3 (pre-registered, §8-iii) pinned to the array's "
                         "along-axis station spacing; pole spacing = that / 3 "
                         "(image-grid oversampling x3). Extents preserved to "
                         "within one cell; station array unchanged."),
        authority="ROADMAP §8 2026-07-27 (Strider's §8-ii sign-off, ruling 1)",
    )


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


def row_center_cols(d, grid_shape=GRID_SHAPE):
    """Two grid indices `d` columns apart on the grid's centre row, placed
    symmetric about the grid's centre column (PLACEMENT_RULE).

    Refinement-invariant by construction: the pair stays in the array's
    interior at any grid spacing. Supersedes the inherited fixed `col_a = 1`,
    which was an artefact of the 11-wide grid and would sit outside the array
    footprint on the refined 17-wide grid (see module docstring)."""
    n_lat, n_lon = grid_shape
    row = n_lat // 2
    col_a = (n_lon - 1 - d) // 2
    col_b = col_a + d
    if not (0 <= col_a and col_b < n_lon):
        raise ValueError(f"d={d} does not fit on a centre row of width "
                          f"{n_lon} (would need cols {col_a}..{col_b})")
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
