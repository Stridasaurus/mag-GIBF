"""Card A, Tier 2 — the H-A hinge: FLR source coherence through the real DF-only A.

Runs AFTER the 2026-07-11 design pin (ROADMAP.md §8) and the audit A1-A3 gate
(2026-07-10), per repo-root handoff.md. Executed by Shane Gilbertie. Every design
parameter below is the pinned value from the §8 2026-07-11 entry — nothing here
was chosen after seeing a Tier-2 number.

Pushes the FLR source-coherence model (Tier 1's driven damped oscillator,
Southwood 1974 / Chen & Hasegawa 1974) through the real DF-only transfer matrix
A (transfer.py, Gate-V-pinned adapter), builds the ground CSM from complex
phasor snapshots, and adjudicates H-A per the pre-registered §A.4 rule with the
§8 2026-07-11 verdict-aggregation pin:

  holds    >= 1 realistic-grid cell with MC-averaged max-over-placement
           kappa_ground >= 0.30 AND co-primary >= 0.394 at that cell
  fails    every realistic cell fails its arm (kappa < 0.10 or floor-
           indistinguishable per the A1-pinned gap-conditioned permutation
           test; co-primary concurring in the same floor-referenced sense)
  marginal otherwise

Pinned design (§8 2026-07-11): 10-station synthetic meridional chain at 20.0E
centered 68.0N, d in {100,150,250} km; pole grid = station span + 2 spacings
margin at half-station spacing, lon +/-6 deg at ~half-station azimuthal spacing,
offset half a pole spacing in lon (Gate V V1 coincidence guard); primary run
sensor-noiseless, flat longitude profile; n_mc=400 at N=64 / 100 at N=1024;
Q in {5,10,20}, |rho| in {0.70,0.85,0.95}, beta in {0.05,0.10,0.20}/100km;
x_r swept over the station span in 41 uniform steps, max over placement;
seed 20260712. Descriptive rows (reported, never adjudicating): real IMAGE
Finnish chain, lon extent +/-12 deg, Gaussian azimuthal envelope sigma=2 deg,
sensor noise snr_db in {10,20}.

|rho| operationalization (pinned): sigma_bg calibrated per cell/placement so the
complex coherence magnitude between the two poles nearest x_r +/- 1 resonance
half-width equals the target. kappa_source from the SAMPLED source snapshots'
CSM at the same N. kappa = top-1 throughout (audit A1).

Operational readings this script had to fix (documented, flagged in the PR for
Strider's review; none is a pre-registered threshold change):
  * GAP-SCALE NOTE: the A1 gap-conditioning test compares Tier-2 per-trial
    top-2 eigenvalue gaps against Tier 1's archived incoherent floor gaps, but
    the two live on different physical scales (Tier 1: unit-power abstract
    scenario, O(1); Tier 2: ground CSM in tesla^2, ~1e-24). The test is run
    exactly as pinned (raw gaps) AND as a trace-normalized sensitivity
    (Tier-2 gaps rescaled by 2/trace, Tier 1's expected CSM trace being 2).
    Both classifications are archived; if the verdict differed between them
    the run would be a stop-and-ask-Strider blocker.
  * Co-primary floor = the canonical incoherent ImFrac floor (0.068 at N=64,
    ROADMAP §3); p against the phi=0 coherent floor is archived descriptively.
  * Both statistics are adjudicated at the kappa-argmax placement (one
    adjudicating configuration per cell); the co-primary's own placement max
    is archived descriptively.
  * Ground background noise is drawn directly in channel space via a Cholesky
    factor of A A^T (equal in distribution to A @ CN(0, sigma^2 I) — A is
    real); the source-stats pass (pass 2) draws the full source-space noise
    explicitly and computes kappa_source and kappa_ground from IDENTICAL
    snapshots, so the washout gap isolates the kernel.
  * Discrete-grid tie-break: if x_r +/- x_w round to the same pole latitude,
    the pair is forced to adjacent distinct pole latitudes.
  * Sensor SNR (descriptive rows) is referenced to the mean total ground
    signal power (FLR + background) per channel-snapshot.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from tier1_flr_coherence import C_IM, C_RHO, im_frac, kappa
from transfer import build_transfer_matrix

SEED = 20260712                     # pinned §8 2026-07-11 (fresh; not Tier 1's)
KM_PER_DEG = 111.19
RESULTS = Path(__file__).resolve().parent.parent / "results" / "A_flr_coherence"

Q_AXIS = (5.0, 10.0, 20.0)
BETA_AXIS = (0.05, 0.10, 0.20)      # per 100 km
D_AXIS = (100.0, 150.0, 250.0)      # km
RHO_AXIS = (0.70, 0.85, 0.95)
N_PLACEMENTS = 41
N_CFG = {64: 400, 1024: 100}        # snapshot count -> n_mc (equal-n with floors)
KAPPA_HOLDS, KAPPA_FAILS = 0.30, 0.10
ALPHA = 0.05
N_RESAMPLES = 10_000                # A1-pinned permutation test
TIER1_EXPECTED_TRACE = 2.0          # two unit-power sources (gap-scale note)

VARIANTS = {  # name -> (vid, lon_halfwidth_deg, sigma_lon_deg, snr_db)
    "primary":  (0, 6.0, None, None),
    "widelon":  (1, 12.0, None, None),
    "gausslon": (2, 6.0, 2.0, None),
    "snr10":    (3, 6.0, None, 10.0),
    "snr20":    (4, 6.0, None, 20.0),
    "image":    (5, 6.0, None, None),
}

# Real IMAGE Finnish meridional chain (descriptive secondary). Geographic
# coordinates from the IMAGE network's published station list,
# https://space.fmi.fi/image/www/index.php?page=stations, retrieved 2026-07-12
# (procedure pinned §8 2026-07-11; exact values recorded here + manifest).
IMAGE_STATIONS = {
    "SOR": (70.54, 22.22), "KEV": (69.76, 27.01), "KIL": (69.06, 20.77),
    "MUO": (68.02, 23.53), "PEL": (66.90, 24.08), "OUJ": (64.52, 27.23),
    "HAN": (62.25, 26.60), "NUR": (60.50, 24.65), "TAR": (58.26, 26.46),
}
IMAGE_SOURCE = ("https://space.fmi.fi/image/www/index.php?page=stations "
                "(retrieved 2026-07-12)")


# ------------------------------------------------------------------ geometry

def synthetic_stations(d_km):
    """Pinned primary chain: 10 stations, lon 20.0E, centered 68.0N."""
    dlat = d_km / KM_PER_DEG
    return 68.0 + (np.arange(10) - 4.5) * dlat, np.full(10, 20.0)


def pole_grid(st_lat, st_lon, lon_halfwidth_deg, ref_lat):
    """Pinned SECS pole grid: lat = station span + 2 station-spacings margin
    each side at half-station spacing; lon = +/-lon_halfwidth about the station
    meridian at ~half-station azimuthal spacing at ref_lat, offset by half a
    pole spacing from the meridian (Gate V V1 coincidence guard)."""
    spacing = (st_lat.max() - st_lat.min()) / (st_lat.size - 1)
    p_dlat = spacing / 2.0
    plat_1d = np.arange(st_lat.min() - 2.0 * spacing,
                        st_lat.max() + 2.0 * spacing + 1e-9, p_dlat)
    meridian = float(np.mean(st_lon))
    dlon = p_dlat / np.cos(np.radians(ref_lat))
    m = int(np.floor((lon_halfwidth_deg - dlon / 2.0) / dlon)) + 1
    half = (np.arange(m) + 0.5) * dlon
    plon_1d = meridian + np.concatenate([-half[::-1], half])
    plat, plon = np.meshgrid(plat_1d, plon_1d, indexing="ij")
    return plat.ravel(), plon.ravel(), plat_1d, plon_1d, meridian


# ------------------------------------------------------------ FLR source model

def flr_profile(plat, plon, x_r_deg, q, beta_per_100km, meridian,
                sigma_lon_deg=None):
    """Complex FLR current profile on the pole grid. R = 1/(u + i) with
    u = (lat - x_r) * 111.19 km / x_w, x_w = 100/(2 Q beta) km (Tier 1's
    linearised omega_r(x); arg sweeps 0 -> -90 -> -180 through resonance).
    Flat in longitude (pinned primary) or Gaussian envelope (descriptive)."""
    x_w_km = 100.0 / (2.0 * q * beta_per_100km)
    u = (plat - x_r_deg) * KM_PER_DEG / x_w_km
    c = 1.0 / (u + 1j)
    if sigma_lon_deg is not None:
        c = c * np.exp(-0.5 * ((plon - meridian) / sigma_lon_deg) ** 2)
    return c, x_w_km


def coherence_pair_indices(plat_1d, plon_1d, x_r_deg, x_w_km, meridian):
    """The two poles nearest x_r +/- 1 resonance half-width (pinned |rho|
    operationalization), at the lon column nearest the station meridian.
    Discrete-grid tie-break: forced to adjacent distinct latitudes."""
    x_w_deg = x_w_km / KM_PER_DEG
    ilo = int(np.argmin(np.abs(plat_1d - (x_r_deg - x_w_deg))))
    ihi = int(np.argmin(np.abs(plat_1d - (x_r_deg + x_w_deg))))
    if ilo == ihi:
        ihi = min(ilo + 1, plat_1d.size - 1)
        ilo = ihi - 1
    jlon = int(np.argmin(np.abs(plon_1d - meridian)))
    n_lon = plon_1d.size
    return ilo * n_lon + jlon, ihi * n_lon + jlon


def calibrate_sigma_bg(c, i, j, rho_target):
    """sigma_bg such that |coherence| between poles i, j equals rho_target:
    |rho|^2 (|c_i|^2 + P)(|c_j|^2 + P) = |c_i|^2 |c_j|^2, P = sigma_bg^2."""
    a2, b2 = abs(c[i]) ** 2, abs(c[j]) ** 2
    r2 = rho_target ** 2
    bq = r2 * (a2 + b2)
    disc = bq * bq - 4.0 * r2 * (r2 - 1.0) * a2 * b2
    p = (-bq + np.sqrt(disc)) / (2.0 * r2)
    return float(np.sqrt(max(p, 0.0)))


# ------------------------------------------------------- batched statistics

def batched_csm_stats(x):
    """Per-trial (kappa top-1, top-2 eigenvalue gap, ImFrac, trace) for a batch
    of snapshot matrices x (n_mc, n_ch, N). Same statistics as Tier 1's kappa()
    and im_frac(), vectorised over trials."""
    n_snap = x.shape[-1]
    s = x @ x.conj().transpose(0, 2, 1) / n_snap
    w, v = np.linalg.eigh(s)
    u = v[:, :, -1]
    re, im = u.real, u.imag
    g11 = np.sum(re * re, axis=1)
    g22 = np.sum(im * im, axis=1)
    g12 = np.sum(re * im, axis=1)
    tr_g = g11 + g22
    lam_min = 0.5 * (tr_g - np.sqrt((g11 - g22) ** 2 + 4.0 * g12 ** 2))
    kap = np.sqrt(np.clip(lam_min, 0.0, None) / tr_g)
    gap = w[:, -1] - w[:, -2]
    imf = (np.linalg.norm(s.imag, axis=(1, 2)) /
           np.linalg.norm(s, axis=(1, 2)))
    trace = np.trace(s, axis1=1, axis2=2).real
    return kap, gap, imf, trace


def _cn(rng, shape):
    return (rng.standard_normal(shape) +
            1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def draw_ground_snapshots(rng, g, chol_aat, sigma_bg, n_mc, n_snap,
                          sigma_sensor=0.0):
    """X = g d^T + sigma_bg * chol(A A^T) Z (+ sensor noise). Equal in
    distribution to A @ (c d^T + sigma_bg E): A is real, so cov(A E) =
    sigma_bg^2 A A^T. Draw order: d, Z, [sensor]."""
    d = _cn(rng, (n_mc, 1, n_snap))
    z = _cn(rng, (n_mc, g.size, n_snap))
    x = g[None, :, None] * d + sigma_bg * (chol_aat @ z)
    if sigma_sensor > 0.0:
        x = x + sigma_sensor * _cn(rng, (n_mc, g.size, n_snap))
    return x


# --------------------------------------- A1-pinned permutation machinery

def perm_test_greater(sample, floor, seed_key, n_resamples=N_RESAMPLES):
    """One-sided two-sample permutation test on the difference of means
    (audit A1, ROADMAP §8 2026-07-10). H0: mean(sample) <= mean(floor).
    Returns the (add-one) p-value; 'indistinguishable from floor' == p >= alpha."""
    x = np.asarray(sample, dtype=np.float64)
    f = np.asarray(floor, dtype=np.float64)
    rng = np.random.default_rng(seed_key)
    obs = x.mean() - f.mean()
    pooled = np.concatenate([x, f])
    perm = rng.permuted(np.tile(pooled, (n_resamples, 1)), axis=1)
    diffs = perm[:, :x.size].mean(axis=1) - perm[:, x.size:].mean(axis=1)
    return float((1 + np.count_nonzero(diffs >= obs)) / (n_resamples + 1))


def classify_cell(trials, floors, pins, seed_key):
    """Classify one realistic-grid cell per §A.4 + the §8 2026-07-11
    aggregation pin, from its per-trial arrays at the adjudicating placement.

    trials: dict with per-trial 'kappa', 'gap', 'imfrac', 'trace' arrays.
    floors: dict with Tier 1 N=64 per-trial 'kappa_phi0', 'gap_incoherent',
            'imfrac_incoherent', 'imfrac_phi0' arrays (audit A2 archive).
    pins:   dict with 'imfrac_holds' (0.394) and 'imfrac_fails' (0.137).
    """
    k_mean = float(np.mean(trials["kappa"]))
    i_mean = float(np.mean(trials["imfrac"]))
    holds = k_mean >= KAPPA_HOLDS and i_mean >= pins["imfrac_holds"]

    # gap conditioning (A1 decision 3): near-degenerate <=> per-trial gap NOT
    # significantly above the incoherent floor's per-trial gap distribution.
    p_gap_raw = perm_test_greater(trials["gap"], floors["gap_incoherent"],
                                  list(seed_key) + [0])
    gap_scaled = trials["gap"] * (TIER1_EXPECTED_TRACE / trials["trace"])
    p_gap_scaled = perm_test_greater(gap_scaled, floors["gap_incoherent"],
                                     list(seed_key) + [1])
    near_deg_raw = p_gap_raw >= ALPHA
    near_deg_scaled = p_gap_scaled >= ALPHA

    p_kappa = perm_test_greater(trials["kappa"], floors["kappa_phi0"],
                                list(seed_key) + [2])
    p_imf = perm_test_greater(trials["imfrac"], floors["imfrac_incoherent"],
                              list(seed_key) + [3])
    p_imf_phi0 = perm_test_greater(trials["imfrac"], floors["imfrac_phi0"],
                                   list(seed_key) + [4])

    kappa_fail = k_mean < KAPPA_FAILS or p_kappa >= ALPHA
    imf_fail = i_mean <= pins["imfrac_fails"] or p_imf >= ALPHA

    def cell_class(near_degenerate):
        if holds:
            return "holds"
        fails = imf_fail if near_degenerate else (kappa_fail and imf_fail)
        return "fails" if fails else "marginal"

    return dict(kappa_mean=k_mean, imfrac_mean=i_mean, holds=holds,
                p_gap_raw=p_gap_raw, p_gap_scaled=p_gap_scaled,
                near_deg_raw=near_deg_raw, near_deg_scaled=near_deg_scaled,
                p_kappa_floor=p_kappa, p_imfrac_floor=p_imf,
                p_imfrac_phi0_floor=p_imf_phi0,
                kappa_fail_arm=kappa_fail, imfrac_fail_arm=imf_fail,
                cell_class_raw=cell_class(near_deg_raw),
                cell_class_scaled=cell_class(near_deg_scaled))


def aggregate_verdict(cell_classes):
    """§8 2026-07-11 pin: holds is an existence claim; fails is universal."""
    if any(c == "holds" for c in cell_classes):
        return "holds"
    if all(c == "fails" for c in cell_classes):
        return "fails"
    return "marginal"


# ------------------------------------------------------------------ sweeps

def build_geometry(st_lat, st_lon, lon_halfwidth_deg, ref_lat):
    plat, plon, plat_1d, plon_1d, meridian = pole_grid(
        st_lat, st_lon, lon_halfwidth_deg, ref_lat)
    tm = build_transfer_matrix(st_lat, st_lon, plat, plon)  # DF-only defaults
    aat = tm.A @ tm.A.T
    chol = np.linalg.cholesky(aat)
    return dict(tm=tm, chol=chol, afro2=float(np.sum(tm.A ** 2)),
                plat=plat, plon=plon, plat_1d=plat_1d, plon_1d=plon_1d,
                meridian=meridian, st_lat=st_lat, st_lon=st_lon)


def run_cell(geom, vid, n_snap, n_mc, idx, q, beta, rho, xr_grid,
             sigma_lon_deg, snr_db):
    """One (Q, beta, d, |rho|) cell: placement sweep, per-placement MC stats,
    argmax placement selection. Returns placement means + per-trial arrays at
    the adjudicating (kappa-argmax) placement."""
    iq, ib, id_, irho = idx
    means = {k: np.empty(len(xr_grid)) for k in ("kappa", "gap", "imfrac")}
    best = None
    for ipl, x_r in enumerate(xr_grid):
        c, x_w_km = flr_profile(geom["plat"], geom["plon"], x_r, q, beta,
                                geom["meridian"], sigma_lon_deg)
        i, j = coherence_pair_indices(geom["plat_1d"], geom["plon_1d"],
                                      x_r, x_w_km, geom["meridian"])
        sigma_bg = calibrate_sigma_bg(c, i, j, rho)
        g = geom["tm"].A @ c
        sigma_sensor = 0.0
        if snr_db is not None:
            p_sig = (np.sum(np.abs(g) ** 2) +
                     sigma_bg ** 2 * geom["afro2"]) / g.size
            sigma_sensor = float(np.sqrt(p_sig / 10.0 ** (snr_db / 10.0)))
        rng = np.random.default_rng(
            [SEED, vid, n_snap, iq, ib, id_, irho, ipl])
        x = draw_ground_snapshots(rng, g, geom["chol"], sigma_bg,
                                  n_mc, n_snap, sigma_sensor)
        kap, gap, imf, trace = batched_csm_stats(x)
        means["kappa"][ipl] = kap.mean()
        means["gap"][ipl] = gap.mean()
        means["imfrac"][ipl] = imf.mean()
        if best is None or kap.mean() > best["kappa"].mean():
            best = dict(kappa=kap, gap=gap, imfrac=imf, trace=trace,
                        ipl=ipl, x_r=x_r, sigma_bg=sigma_bg)
    return means, best


def source_stats(geom, vid, n_snap, n_mc, idx, q, beta, rho, x_r,
                 sigma_lon_deg, chunk=25):
    """Pass 2 at the adjudicating placement: draw the full source-space
    snapshots explicitly and compute kappa_source AND kappa_ground from the
    IDENTICAL snapshots (the washout gap isolates the kernel). Independent
    replicate of pass 1 (separate seed tag)."""
    iq, ib, id_, irho = idx
    c, x_w_km = flr_profile(geom["plat"], geom["plon"], x_r, q, beta,
                            geom["meridian"], sigma_lon_deg)
    i, j = coherence_pair_indices(geom["plat_1d"], geom["plon_1d"],
                                  x_r, x_w_km, geom["meridian"])
    sigma_bg = calibrate_sigma_bg(c, i, j, rho)
    rng = np.random.default_rng([SEED, vid, n_snap, iq, ib, id_, irho, 7])
    a = geom["tm"].A
    out = {k: [] for k in ("kappa_src", "gap_src", "imfrac_src",
                           "kappa_gnd", "imfrac_gnd")}
    done = 0
    while done < n_mc:
        m = min(chunk, n_mc - done)
        d = _cn(rng, (m, 1, n_snap))
        e = _cn(rng, (m, c.size, n_snap))
        s = c[None, :, None] * d + sigma_bg * e
        # ground stats from the same snapshots
        kap_g, _, imf_g, _ = batched_csm_stats(a @ s)
        # source stats via the N x N Gram (same nonzero spectrum as the CSM)
        gram = s.conj().transpose(0, 2, 1) @ s / n_snap
        w, v = np.linalg.eigh(gram)
        utop = (s @ v[:, :, -1:])[:, :, 0]
        re, im = utop.real, utop.imag
        g11 = np.sum(re * re, axis=1)
        g22 = np.sum(im * im, axis=1)
        g12 = np.sum(re * im, axis=1)
        tr_g = g11 + g22
        lam_min = 0.5 * (tr_g - np.sqrt((g11 - g22) ** 2 + 4.0 * g12 ** 2))
        s_src = s @ s.conj().transpose(0, 2, 1) / n_snap
        out["kappa_src"].append(np.sqrt(np.clip(lam_min, 0.0, None) / tr_g))
        out["gap_src"].append(w[:, -1] - w[:, -2])
        out["imfrac_src"].append(np.linalg.norm(s_src.imag, axis=(1, 2)) /
                                 np.linalg.norm(s_src, axis=(1, 2)))
        out["kappa_gnd"].append(kap_g)
        out["imfrac_gnd"].append(imf_g)
        done += m
    return {k: np.concatenate(v) for k, v in out.items()}


def sweep_variant(name, st_lat, st_lon, n_snap, n_mc, d_axis, xr_grid=None):
    """Full (Q, beta, d, |rho|) x placement sweep for one variant. For the
    IMAGE variant d_axis has one entry (the chain's mean spacing, recorded)."""
    vid, lon_hw, sigma_lon, snr_db = VARIANTS[name]
    shape = (len(Q_AXIS), len(BETA_AXIS), len(d_axis), len(RHO_AXIS))
    surf = {k: np.empty(shape + (N_PLACEMENTS,))
            for k in ("kappa", "gap", "imfrac")}
    adj = {k: np.empty(shape) for k in
           ("kappa", "gap", "imfrac", "x_r", "ipl", "sigma_bg",
            "imfrac_ownmax")}
    per_trial = {k: np.empty(shape + (n_mc,))
                 for k in ("kappa", "gap", "imfrac", "trace")}
    geoms = []
    for id_, d_km in enumerate(d_axis):
        if name == "image":
            lat, lon = st_lat, st_lon
            ref_lat = float(np.mean(lat))
        else:
            lat, lon = synthetic_stations(d_km)
            ref_lat = 68.0
        geoms.append(build_geometry(lat, lon, lon_hw, ref_lat))
    for id_, d_km in enumerate(d_axis):
        geom = geoms[id_]
        grid = (np.linspace(geom["st_lat"].min(), geom["st_lat"].max(),
                            N_PLACEMENTS) if xr_grid is None else xr_grid)
        for iq, q in enumerate(Q_AXIS):
            for ib, beta in enumerate(BETA_AXIS):
                for irho, rho in enumerate(RHO_AXIS):
                    idx = (iq, ib, id_, irho)
                    means, best = run_cell(geom, vid, n_snap, n_mc, idx, q,
                                           beta, rho, grid, sigma_lon, snr_db)
                    for k in ("kappa", "gap", "imfrac"):
                        surf[k][idx] = means[k]
                        adj[k][idx] = means[k][best["ipl"]]
                    adj["x_r"][idx] = best["x_r"]
                    adj["ipl"][idx] = best["ipl"]
                    adj["sigma_bg"][idx] = best["sigma_bg"]
                    adj["imfrac_ownmax"][idx] = means["imfrac"].max()
                    for k in ("kappa", "gap", "imfrac", "trace"):
                        per_trial[k][idx] = best[k]
        print(f"    [{name} N={n_snap}] d={d_km:.1f} km done "
              f"({geom['plat'].size} poles)", flush=True)
    return dict(surf=surf, adj=adj, per_trial=per_trial, geoms=geoms)


# -------------------------------------------------------------------- main

def load_floors():
    d = np.load(RESULTS / "floor_distributions.npz")
    return dict(kappa_phi0=d["kappa_phi0_coherent_N64"],
                gap_incoherent=d["gap_incoherent_N64"],
                imfrac_incoherent=d["imfrac_incoherent_N64"],
                imfrac_phi0=d["imfrac_phi0_coherent_N64"])


def load_pins():
    m = json.loads((RESULTS / "manifest.json").read_text())
    return dict(imfrac_holds=m["pins"]["0.3"]["imfrac_threshold"],
                imfrac_fails=m["pins"]["0.1"]["imfrac_threshold"])


def sanity_check_batched_stats():
    """Batched statistics must agree with Tier 1's scalar kappa()/im_frac()."""
    rng = np.random.default_rng(0)
    x = _cn(rng, (3, 6, 32))
    kap, _, imf, _ = batched_csm_stats(x)
    for t in range(3):
        s = x[t] @ x[t].conj().T / 32
        _, v = np.linalg.eigh(s)
        assert abs(kap[t] - kappa(v[:, -1])) < 1e-12
        assert abs(imf[t] - im_frac(s)) < 1e-12


def main():
    sanity_check_batched_stats()
    RESULTS.mkdir(parents=True, exist_ok=True)
    floors = load_floors()
    pins = load_pins()
    image_lat = np.array([v[0] for v in IMAGE_STATIONS.values()])
    image_lon = np.array([v[1] for v in IMAGE_STATIONS.values()])
    image_d = float((image_lat.max() - image_lat.min()) /
                    (image_lat.size - 1) * KM_PER_DEG)

    runs = {}
    print("pass 1: ground sweeps", flush=True)
    runs[("primary", 64)] = sweep_variant("primary", None, None, 64,
                                          N_CFG[64], D_AXIS)
    runs[("primary", 1024)] = sweep_variant("primary", None, None, 1024,
                                            N_CFG[1024], D_AXIS)
    for name in ("widelon", "gausslon", "snr10", "snr20"):
        runs[(name, 64)] = sweep_variant(name, None, None, 64,
                                         N_CFG[64], D_AXIS)
    runs[("image", 64)] = sweep_variant("image", image_lat, image_lon, 64,
                                        N_CFG[64], (image_d,))

    # ---- pass 2: source stats at each primary cell's adjudicating placement
    print("pass 2: source-vs-ground washout (primary, N=64)", flush=True)
    prim = runs[("primary", 64)]
    shape = (len(Q_AXIS), len(BETA_AXIS), len(D_AXIS), len(RHO_AXIS))
    src = {k: np.empty(shape) for k in
           ("kappa_src", "imfrac_src", "kappa_gnd", "imfrac_gnd")}
    src_trials_kappa = np.empty(shape + (N_CFG[64],))
    for iq, q in enumerate(Q_AXIS):
        for ib, beta in enumerate(BETA_AXIS):
            for id_ in range(len(D_AXIS)):
                for irho, rho in enumerate(RHO_AXIS):
                    idx = (iq, ib, id_, irho)
                    st = source_stats(prim["geoms"][id_], VARIANTS["primary"][0],
                                      64, N_CFG[64], idx, q, beta, rho,
                                      prim["adj"]["x_r"][idx], None)
                    src["kappa_src"][idx] = st["kappa_src"].mean()
                    src["imfrac_src"][idx] = st["imfrac_src"].mean()
                    src["kappa_gnd"][idx] = st["kappa_gnd"].mean()
                    src["imfrac_gnd"][idx] = st["imfrac_gnd"].mean()
                    src_trials_kappa[idx] = st["kappa_src"]
        print(f"    Q={q:g} done", flush=True)

    # ---- adjudication (primary, N=64 only)
    print("adjudication (primary, N=64)", flush=True)
    cells = []
    for iq in range(len(Q_AXIS)):
        for ib in range(len(BETA_AXIS)):
            for id_ in range(len(D_AXIS)):
                for irho in range(len(RHO_AXIS)):
                    idx = (iq, ib, id_, irho)
                    trials = {k: prim["per_trial"][k][idx]
                              for k in ("kappa", "gap", "imfrac", "trace")}
                    cls = classify_cell(trials, floors, pins,
                                        [SEED, 90, iq, ib, id_, irho])
                    cls["idx"] = idx
                    cells.append(cls)
    verdict_raw = aggregate_verdict([c["cell_class_raw"] for c in cells])
    verdict_scaled = aggregate_verdict([c["cell_class_scaled"] for c in cells])
    gap_note_moot = verdict_raw == verdict_scaled

    holds_cells = [c["idx"] for c in cells if c["cell_class_raw"] == "holds"]
    best_cell = max(cells, key=lambda c: c["kappa_mean"])

    # ------------------------------------------------------------- outputs
    arrays = {}
    for (name, n_snap), run in runs.items():
        tag = f"{name}_N{n_snap}"
        for k in ("kappa", "gap", "imfrac"):
            arrays[f"{tag}_{k}_placement_mean"] = run["surf"][k]
            arrays[f"{tag}_{k}_adj"] = run["adj"][k]
        for k in ("x_r", "sigma_bg", "imfrac_ownmax"):
            arrays[f"{tag}_{k}_adj"] = run["adj"][k]
    for k in ("kappa", "gap", "imfrac", "trace"):
        arrays[f"primary_N64_{k}_trials_adj"] = \
            runs[("primary", 64)]["per_trial"][k]
    for k, v in src.items():
        arrays[f"primary_N64_{k}_pass2"] = v
    arrays["primary_N64_kappa_src_trials"] = src_trials_kappa
    arrays.update(Q_axis=np.array(Q_AXIS), beta_axis=np.array(BETA_AXIS),
                  d_axis=np.array(D_AXIS), rho_axis=np.array(RHO_AXIS),
                  image_d_km=np.array([image_d]),
                  image_station_lat=image_lat, image_station_lon=image_lon)
    np.savez(RESULTS / "tier2_results.npz", **arrays)

    # summary CSV
    lines = ["variant,N,Q,beta,d_km,rho,x_r_argmax,kappa_ground,gap,imfrac,"
             "imfrac_ownmax,kappa_source,imfrac_source,washout_gap,"
             "p_kappa_floor,p_imfrac_floor,p_gap_raw,p_gap_scaled,"
             "near_deg_raw,near_deg_scaled,cell_class_raw,cell_class_scaled"]
    by_idx = {c["idx"]: c for c in cells}
    axes = dict(primary=D_AXIS, widelon=D_AXIS, gausslon=D_AXIS,
                snr10=D_AXIS, snr20=D_AXIS, image=(image_d,))
    for (name, n_snap), run in runs.items():
        for iq, q in enumerate(Q_AXIS):
            for ib, beta in enumerate(BETA_AXIS):
                for id_, d_km in enumerate(axes[name]):
                    for irho, rho in enumerate(RHO_AXIS):
                        idx = (iq, ib, id_, irho)
                        row = [name, n_snap, q, beta, f"{d_km:.1f}", rho,
                               f"{run['adj']['x_r'][idx]:.4f}",
                               f"{run['adj']['kappa'][idx]:.4f}",
                               f"{run['adj']['gap'][idx]:.6e}",
                               f"{run['adj']['imfrac'][idx]:.4f}",
                               f"{run['adj']['imfrac_ownmax'][idx]:.4f}"]
                        if name == "primary" and n_snap == 64:
                            c = by_idx[idx]
                            row += [f"{src['kappa_src'][idx]:.4f}",
                                    f"{src['imfrac_src'][idx]:.4f}",
                                    f"{src['kappa_src'][idx] - src['kappa_gnd'][idx]:.4f}",
                                    f"{c['p_kappa_floor']:.4f}",
                                    f"{c['p_imfrac_floor']:.4f}",
                                    f"{c['p_gap_raw']:.4f}",
                                    f"{c['p_gap_scaled']:.4f}",
                                    str(c["near_deg_raw"]),
                                    str(c["near_deg_scaled"]),
                                    c["cell_class_raw"],
                                    c["cell_class_scaled"]]
                        else:
                            row += [""] * 11
                        lines.append(",".join(str(v) for v in row))
    lines.append(",".join(["VERDICT", "64"] + [""] * 18 +
                          [verdict_raw, verdict_scaled]))
    (RESULTS / "tier2_summary.csv").write_text("\n".join(lines) + "\n")

    # manifest (audit A3 workflow: runner committed before first run;
    # script_sha256 recorded)
    script = Path(__file__).resolve()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True,
                                cwd=script.parent).stdout.strip()
    except Exception:
        commit = "unknown"
    import secsy
    manifest = dict(
        experiment="Card A Tier 2 (H-A adjudication)",
        seed=SEED, numpy=np.__version__,
        secsy=getattr(secsy, "__version__", "see submodule pin"),
        git_commit=commit,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        preregistration="ROADMAP.md §8 2026-07-11 (design pin) + 2026-07-10 "
                        "(audit A1: kappa=top-1, permutation test alpha=0.05, "
                        "gap conditioning, |rho|=0.95 threshold curve)",
        design=dict(
            stations="10-station synthetic meridional chain, lon 20.0E, "
                     "centered 68.0N, d in {100,150,250} km; ground r = RE "
                     "(transfer.py default)",
            pole_grid="station span + 2 station-spacings margin each side at "
                      "half-station spacing; lon +/-6 deg (widelon: +/-12) at "
                      "~half-station azimuthal spacing at 68N, offset half a "
                      "pole spacing in lon (Gate V V1 guard)",
            sensor_noise="primary noiseless; descriptive snr_db in {10, 20} "
                         "referenced to mean total ground signal power",
            lon_profile="primary flat; descriptive Gaussian sigma_lon = 2 deg",
            n_mc=N_CFG, Q=Q_AXIS, rho=RHO_AXIS, beta_per_100km=BETA_AXIS,
            placements=f"x_r over station lat span, {N_PLACEMENTS} uniform "
                       "steps, max over placement per cell",
            rho_operationalization="sigma_bg calibrated per cell+placement so "
                                   "|coherence| between the two poles nearest "
                                   "x_r +/- 1 half-width (meridian lon column; "
                                   "adjacent-pole tie-break) equals target",
            kappa_source="from sampled source snapshots' CSM at same N "
                         "(pass 2, identical snapshots as its paired ground "
                         "stats; independent replicate of pass 1)",
        ),
        image_chain=dict(stations=IMAGE_STATIONS, source=IMAGE_SOURCE,
                         mean_spacing_km=image_d,
                         note="descriptive only; ref_lat = mean station lat; "
                              "meridian = mean station lon"),
        rng_scheme="np.random.default_rng([SEED, vid, N, iQ, ibeta, id, irho, "
                   "ipl]) per cell+placement (pass 1); ... + [7] source pass; "
                   "[SEED, 90, iQ, ibeta, id, irho, test_id] permutation tests",
        permutation_test=dict(kind="one-sided two-sample, difference of means",
                              n_resamples=N_RESAMPLES, alpha=ALPHA,
                              floors="results/A_flr_coherence/"
                                     "floor_distributions.npz (audit A2)"),
        thresholds=dict(kappa_holds=KAPPA_HOLDS, kappa_fails=KAPPA_FAILS,
                        imfrac_holds=pins["imfrac_holds"],
                        imfrac_fails=pins["imfrac_fails"]),
        operational_readings=[
            "GAP-SCALE NOTE: A1 gap-conditioning run raw as pinned AND "
            "trace-normalized (x 2/trace) as sensitivity; Tier-2 ground gaps "
            "(~tesla^2) vs Tier-1 unit-power floor gaps are scale-mismatched, "
            "so the raw test classifies every cell near-degenerate; flagged "
            "for Strider's review. Verdict identical under both readings: "
            + str(gap_note_moot),
            "co-primary floor = canonical incoherent ImFrac floor (0.068); "
            "p vs phi0-coherent floor archived descriptively",
            "both statistics adjudicated at the kappa-argmax placement; "
            "co-primary own-placement max archived descriptively",
            "ground noise drawn in channel space via chol(A A^T) — equal in "
            "distribution to projecting source-space noise (A real)",
        ],
        verdict=dict(raw=verdict_raw, scaled=verdict_scaled,
                     holds_cells=[list(map(int, i)) for i in holds_cells],
                     best_cell=dict(idx=list(map(int, best_cell["idx"])),
                                    kappa=best_cell["kappa_mean"],
                                    imfrac=best_cell["imfrac_mean"])),
    )
    (RESULTS / "manifest_tier2.json").write_text(json.dumps(manifest, indent=2))

    # ------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adj = runs[("primary", 64)]["adj"]
    fig, axs = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    for ib, beta in enumerate(BETA_AXIS):
        for id_, d_km in enumerate(D_AXIS):
            ax = axs[ib][id_]
            ax.axhspan(KAPPA_HOLDS, 1.0, color="0.92", zorder=0)
            ax.axhspan(0.0, KAPPA_FAILS, color="0.92", zorder=0)
            for irho, rho in enumerate(RHO_AXIS):
                ax.plot(Q_AXIS, adj["kappa"][:, ib, id_, irho], "o-",
                        color=C_RHO[rho], lw=2, ms=5, label=f"|ρ|={rho}")
                ax.plot(Q_AXIS, src["kappa_src"][:, ib, id_, irho], "s--",
                        color=C_RHO[rho], lw=1, ms=4, alpha=0.45)
            ax.set_xscale("log")
            ax.set_xticks(Q_AXIS)
            ax.set_xticklabels([f"{q:g}" for q in Q_AXIS])
            ax.set_title(f"β={beta:g}/100 km, d={d_km:g} km", fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_ylim(0, 1.0)
    axs[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axs[-1]:
        ax.set_xlabel("Q")
    for row in axs:
        row[0].set_ylabel("κ (top-1)")
    fig.suptitle("Card A Tier 2 — κ_ground (solid) vs κ_source (dashed, pass 2)"
                 " at the adjudicating placement; H-A bands 0.30 / 0.10",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_tier2_kappa_surface.png", dpi=160)

    fig2, axs2 = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    for ib, beta in enumerate(BETA_AXIS):
        for id_, d_km in enumerate(D_AXIS):
            ax = axs2[ib][id_]
            ax.axhline(pins["imfrac_holds"], color="0.35", lw=1, ls=":")
            ax.axhline(pins["imfrac_fails"], color="0.35", lw=1, ls=":")
            for irho, rho in enumerate(RHO_AXIS):
                ax.plot(Q_AXIS, adj["imfrac"][:, ib, id_, irho], "o-",
                        color=C_RHO[rho], lw=2, ms=5, label=f"|ρ|={rho}")
            ax.set_xscale("log")
            ax.set_xticks(Q_AXIS)
            ax.set_xticklabels([f"{q:g}" for q in Q_AXIS])
            ax.set_title(f"β={beta:g}/100 km, d={d_km:g} km", fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_ylim(0, 1.0)
    axs2[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axs2[-1]:
        ax.set_xlabel("Q")
    for row in axs2:
        row[0].set_ylabel("‖Im S‖/‖S‖")
    fig2.suptitle("Card A Tier 2 — co-primary ‖Im S_ground‖/‖S_ground‖ at the "
                  "adjudicating placement; pinned thresholds 0.394 / 0.137",
                  fontsize=11)
    fig2.tight_layout()
    fig2.savefig(RESULTS / "fig_tier2_coprimary_surface.png", dpi=160)

    # placement structure at the best cell, primary vs descriptive variants
    bi = best_cell["idx"]
    fig3, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhspan(KAPPA_HOLDS, 1.0, color="0.92", zorder=0)
    ax.axhspan(0.0, KAPPA_FAILS, color="0.92", zorder=0)
    var_colors = {"primary": "#0072B2", "widelon": "#E69F00",
                  "gausslon": "#009E73", "snr10": "#D55E00",
                  "snr20": "#56B4E9", "image": "#CC79A7"}
    for name, col in var_colors.items():
        run = runs[(name, 64)]
        idx = bi if name != "image" else (bi[0], bi[1], 0, bi[3])
        geom = run["geoms"][idx[2]]
        grid = np.linspace(geom["st_lat"].min(), geom["st_lat"].max(),
                           N_PLACEMENTS)
        ax.plot(grid, run["surf"]["kappa"][idx], color=col, lw=1.8, label=name)
    q, beta, d_km, rho = (Q_AXIS[bi[0]], BETA_AXIS[bi[1]], D_AXIS[bi[2]],
                          RHO_AXIS[bi[3]])
    ax.set_xlabel("resonant latitude x_r (deg)")
    ax.set_ylabel("MC-mean κ_ground")
    ax.set_title(f"Card A Tier 2 — placement sweep at the best cell "
                 f"(Q={q:g}, β={beta:g}, d={d_km:g} km, |ρ|={rho}), N=64",
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig3.tight_layout()
    fig3.savefig(RESULTS / "fig_tier2_placement_variants.png", dpi=160)

    # ----------------------------------------------------------- console
    print("=== Card A Tier 2 — H-A adjudication (primary grid, N=64) ===")
    print(f"  verdict (raw gap conditioning, as pinned): {verdict_raw}")
    print(f"  verdict (trace-normalized sensitivity):    {verdict_scaled}")
    print(f"  gap-scale note moot (verdicts agree):      {gap_note_moot}")
    print(f"  holds cells (iQ,ibeta,id,irho): {holds_cells}")
    bc = best_cell
    print(f"  best cell {bc['idx']}: kappa_ground={bc['kappa_mean']:.4f}, "
          f"imfrac={bc['imfrac_mean']:.4f} "
          f"(Q={Q_AXIS[bc['idx'][0]]:g}, beta={BETA_AXIS[bc['idx'][1]]:g}, "
          f"d={D_AXIS[bc['idx'][2]]:g} km, rho={RHO_AXIS[bc['idx'][3]]})")
    n_class = {}
    for c in cells:
        n_class[c["cell_class_raw"]] = n_class.get(c["cell_class_raw"], 0) + 1
    print(f"  cell classes (raw): {n_class}")
    print(f"  kappa_source range (pass 2): "
          f"{src['kappa_src'].min():.4f} .. {src['kappa_src'].max():.4f}")
    print(f"  washout gap (kappa_src - kappa_gnd) at best cell: "
          f"{src['kappa_src'][bc['idx']] - src['kappa_gnd'][bc['idx']]:.4f}")
    print("  descriptive maxima (kappa_adj max over cells): " +
          ", ".join(f"{name}={runs[(name, 64)]['adj']['kappa'].max():.4f}"
                    for name in ("primary", "widelon", "gausslon",
                                 "snr10", "snr20", "image")))
    print(f"  N=1024 primary kappa_adj max: "
          f"{runs[('primary', 1024)]['adj']['kappa'].max():.4f}")


if __name__ == "__main__":
    main()
