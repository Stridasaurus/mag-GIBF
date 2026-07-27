"""run_experiment_B.py — Experiment-B kickoff runner: solver stack gate + B1
(cost/pilot/floor) + B3 (mode-selection surfaces) + the coherent mini-pilot.

Per repo-root `handoff.md` (2026-07-12, corrected 2026-07-20) and
`viability-test/SPEC_experiment_B.md`. Runs, IN ORDER:

  1. Single-source exact-recovery gate (SPEC §S.6.4). Any failure HALTS
     Experiment B (solver bug, not science) — this script raises SystemExit
     before touching B1/B3/mini-pilot if the gate fails.
  2. B1 (SPEC §S.2-B1): d in {1,2,3,5,8} cells x snr_db in {-5,0,5,10,20},
     n_snap=64, oracle K=2, incoherent equal-power pair, grid reduction on
     AND off. Deliverables: signed-gap cost surfaces, the §8-ii pilot SD,
     the floor panel arrays.
  3. B3 (SPEC §S.2-B3): snr_db in {-5,0,5,10,20} x n_snap in
     {8,16,32,64,128}, BOTH AIC and MDL K-hat distributions vs truth K=2.
     Reports (does not run) the B6a n_snap procedure result (§8-vii).
  4. Mini-pilot: 50 trials at phi=90 deg, snr=5 dB, d=3, n_snap=64,
     |rho|=0.85 (SLOT-1), both sparse solvers, grid reduction ON (pinned
     §S.6.2 — the confirmatory setting).

STOPS THERE. Does not run powered B2/B6 — that is gated on Strider's §8-ii
power-calc sign-off, reviewed after this PR lands (handoff.md stop rule).

GEOMETRY-RESCALE PASS, 2026-07-27 (ROADMAP §8 2026-07-27; Strider's §8-ii
sign-off). This script is unchanged in design; it re-runs on the REFINED SECS
grid (pole spacing 63.5 km, 33x17=561 poles, centred source placement — the
full rule and its justification live in simulate.py's module docstring and are
recorded structurally by `simulate.geometry_record()` in the manifest) and with
SPEC §S.4's eigenvalue floor re-expressed in trace-normalized units
(modeselect.py). EVERY pre-registered coordinate is untouched: d=3, phi=90 deg,
snr in {5,10} (B1 sweeps its pinned axis), |rho|=0.85, n_snap=64, tau_r=1
cell, reduction ON at the confirmatory setting.

CELL-UNIT NOTE (load-bearing for the §8-ii sign-off): every adjudication
quantity here is denominated in GRID CELLS — delta_r_bar, tau_r = 1 cell
(§8-v), and the win rule's "resolves >= 1 grid cell smaller". The pins are
literally unchanged, but the refinement shrinks one cell from 106.6 km to
63.5 km, so the same pinned numbers now mean a 1.68x smaller physical
tolerance. Reported in both units below and in the §8 entry.

N_TRIALS NOTE (documented choice): B1/B3 use n_trials=50 per cell — the
archived brief's own `monte_carlo.n_trials` default, and consistent with B1
being explicitly the *pilot* whose job is to measure a variance, not to be
independently powered. The mini-pilot's n_trials=50 IS pinned (handoff.md).
The powered `n_trials` is NOT computed anywhere in this script: it is
Strider's §8-ii sign-off.

ROW NORMALISATION NOTE (documented choice): `row_normalisation="none"`
(transfer.py's own default) is used throughout, identically for every
solver and every cell (never mixed).

SEED: 20260727 (fresh; distinct from Tier 1's 20260706, Tier 2's 20260712 and
the 2026-07-21 kickoff run's 20260721 per the handoff's no-seed-reuse rule).
"""

import hashlib
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from metrics import delta_r_bar, find_peaks_2d, p_sep
from modeselect import estimate_n_sources
from simulate import (POLE_SPACING_KM, build_array_and_grid, build_csm,
                      centroid_grid_index, eigendecompose, eigenmodes,
                      geometry_record, row_center_cols, simulate_snapshots,
                      GRID_SHAPE)
from solvers import FAIRNESS_CONFIG, solve_all, with_reduction

SEED = 20260727
RESULTS = Path(__file__).resolve().parent.parent / "results" / "B_viability"
ROW_NORMALISATION = "none"

D_AXIS = (1, 2, 3, 5, 8)
SNR_AXIS_B1 = (-5.0, 0.0, 5.0, 10.0, 20.0)
N_SNAP_B1 = 64
N_TRIALS_B1 = 50          # documented choice, see module docstring

SNR_AXIS_B3 = (-5.0, 0.0, 5.0, 10.0, 20.0)
NSNAP_AXIS_B3 = (8, 16, 32, 64, 128)
D_B3 = 3                  # documented choice: consistent with the win/B6a d pin
N_TRIALS_B3 = 50

MINIPILOT_PHI_DEG = 90.0
MINIPILOT_SNR_DB = 5.0
MINIPILOT_D = 3
MINIPILOT_ABS_RHO = 0.85  # SLOT-1 pinned midpoint (ROADMAP §8 2026-07-12/SPEC §S.7)
MINIPILOT_N_SNAP = 64
MINIPILOT_N_TRIALS = 50   # pinned, handoff.md

# Source placement is `simulate.row_center_cols(d)` — the pair centred on the
# grid's centre row/column (simulate.PLACEMENT_RULE). The inherited fixed
# `col_a = 1` is retired with the old grid; see simulate.py's docstring.


def _seed_component(x):
    """Deterministic int encoding of a seed-derivation component. Plain ints
    pass through; everything else (stage labels, float snr_db values) is
    CRC32-hashed (stable across runs and processes, unlike Python's built-in
    `hash()` on strings, which is salted per-process unless
    PYTHONHASHSEED is fixed)."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    return zlib.crc32(repr(x).encode())


def _rng(*components):
    """np.random.default_rng([SEED, *components]) per (master_seed, cell,
    trial) — SPEC §S.1 scheme, extended with CRC32 for non-integer
    components so any cell/trial is independently, deterministically
    reproducible."""
    return np.random.default_rng([SEED, *[_seed_component(c) for c in components]])


# --------------------------------------------------------------- gate (S.6.4)

def single_source_gate():
    """SPEC §S.6.4: all three solvers, one isolated DF source at the grid
    cell nearest the array centroid, oracle K=1, n_snap=64, exact cell =
    global argmax (tau=0). Tier (i) noise-free deterministic (all 3 solvers,
    sparse solvers both reduction settings); tier (ii) 20 dB MC x 10 trials
    (sparse solvers only, both reduction settings, per §S.6.4 "L2 runs tier
    (i) only"). ANY failure halts Experiment B."""
    tm = build_array_and_grid(row_normalisation=ROW_NORMALISATION)
    idx = centroid_grid_index(tm)
    failures = []

    # tier (i)
    X, _ = simulate_snapshots(tm, [idx], [1.0], _rng(0, "gate_i"), n_snap=64, snr_db=None)
    S = build_csm(X)
    lam, U = eigendecompose(S)
    V = eigenmodes(lam, U, 1)
    tier_i = {}
    for reduction in (True, False):
        cfg = with_reduction(FAIRNESS_CONFIG, reduction)
        out = solve_all(tm.A, V, cfg)
        for name in ("l2", "gibf", "mmv"):
            if name == "l2" and reduction is False:
                continue  # L2 has no reduction axis; only report it once
            ok = bool(np.argmax(out[name]["I"]) == idx)
            tier_i[f"{name}_reduction_{reduction}"] = ok
            if not ok:
                failures.append(f"tier(i) {name} reduction={reduction}")

    # tier (ii): 20 dB MC x 10 trials, sparse solvers only, both reductions
    tier_ii = {"gibf": {True: [], False: []}, "mmv": {True: [], False: []}}
    for trial in range(10):
        rng = _rng(1, "gate_ii", trial)
        X, _ = simulate_snapshots(tm, [idx], [1.0], rng, n_snap=64, snr_db=20.0)
        S = build_csm(X)
        lam, U = eigendecompose(S)
        V = eigenmodes(lam, U, 1)
        for reduction in (True, False):
            cfg = with_reduction(FAIRNESS_CONFIG, reduction)
            out = solve_all(tm.A, V, cfg)
            for name in ("gibf", "mmv"):
                ok = bool(np.argmax(out[name]["I"]) == idx)
                tier_ii[name][reduction].append(ok)
                if not ok:
                    failures.append(f"tier(ii) {name} reduction={reduction} trial={trial}")

    passed = len(failures) == 0
    return dict(passed=passed, failures=failures, idx=int(idx),
               centroid_lat=float(tm.pole_lat[idx]), centroid_lon=float(tm.pole_lon[idx]),
               tier_i=tier_i,
               tier_ii={k: {str(r): v for r, v in d.items()} for k, d in tier_ii.items()})


# --------------------------------------------------------------------- B1

def run_b1():
    """B1 (SPEC §S.2-B1 / §8 2026-06-30 second desk-check reclassification):
    incoherent equal-power pair, d x snr grid, both reduction settings.
    Returns per-cell per-method per-trial arrays (delta_r_bar, p_sep) plus
    the signed GIBF-MMV delta_r_bar gap."""
    tm = build_array_and_grid(row_normalisation=ROW_NORMALISATION)
    cells = {}
    for reduction in (True, False):
        cfg = with_reduction(FAIRNESS_CONFIG, reduction)
        for d in D_AXIS:
            idx_a, idx_b = row_center_cols(d)
            true_cells = [np.unravel_index(idx_a, GRID_SHAPE),
                         np.unravel_index(idx_b, GRID_SHAPE)]
            for snr_db in SNR_AXIS_B1:
                dr = {"l2": [], "gibf": [], "mmv": []}
                ps = {"l2": [], "gibf": [], "mmv": []}
                for trial in range(N_TRIALS_B1):
                    rng = _rng(2, "b1", int(reduction), d, snr_db, trial)
                    X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                                              n_snap=N_SNAP_B1, snr_db=snr_db)
                    S = build_csm(X)
                    lam, U = eigendecompose(S)
                    V = eigenmodes(lam, U, 2)
                    out = solve_all(tm.A, V, cfg)
                    for name in ("l2", "gibf", "mmv"):
                        I2d = out[name]["I"].reshape(GRID_SHAPE)
                        dr[name].append(delta_r_bar(I2d, true_cells))
                        ps[name].append(p_sep(I2d, true_cells))
                cell = dict(
                    d=d, snr_db=snr_db, reduction=reduction,
                    separation_km=round(d * POLE_SPACING_KM, 2),
                    delta_r_bar={k: v for k, v in dr.items()},
                    p_sep_rate={k: float(np.mean(v)) for k, v in ps.items()},
                    delta_r_bar_mean={k: float(np.mean(v)) for k, v in dr.items()},
                    delta_r_bar_sd={k: float(np.std(v, ddof=1)) for k, v in dr.items()},
                    gap_signed=[g - m for g, m in zip(dr["gibf"], dr["mmv"])],
                )
                cell["gap_mean"] = float(np.mean(cell["gap_signed"]))
                cell["gap_sd"] = float(np.std(cell["gap_signed"], ddof=1))
                cells[(reduction, d, snr_db)] = cell
    return cells


# --------------------------------------------------------------------- B3

def run_b3():
    """B3 (SPEC §S.2-B3): incoherent equal-power pair at fixed d=D_B3
    (documented choice, see module docstring), snr x n_snap grid, both AIC
    and MDL K-hat vs truth K=2. Grid reduction is irrelevant here (mode
    selection precedes any solver call) so it is not swept.

    Raw CSM eigenvalues are passed straight to `estimate_n_sources`, which
    since 2026-07-27 trace-normalizes internally before applying SPEC §S.4's
    floor (modeselect.py docstring; ROADMAP §8 2026-07-27 ruling 2). The
    2026-07-21 run's dual raw/normalized reading is retired: the raw reading
    was a floor-clipping artefact of an absolute floor meeting a ~1e-24
    tesla^2 spectrum, and preserving it as a code path would just carry the
    defect forward. It stays reproducible at commit 12ff377."""
    tm = build_array_and_grid(row_normalisation=ROW_NORMALISATION)
    idx_a, idx_b = row_center_cols(D_B3)
    cells = {}
    for snr_db in SNR_AXIS_B3:
        for n_snap in NSNAP_AXIS_B3:
            k_mdl, k_aic = [], []
            for trial in range(N_TRIALS_B3):
                rng = _rng(3, "b3", snr_db, n_snap, trial)
                X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                                          n_snap=n_snap, snr_db=snr_db)
                S = build_csm(X)
                lam, U = eigendecompose(S)
                kh_mdl, _, _, _ = estimate_n_sources(lam, n_snap, criterion="mdl")
                kh_aic, _, _, _ = estimate_n_sources(lam, n_snap, criterion="aic")
                k_mdl.append(kh_mdl)
                k_aic.append(kh_aic)
            k_mdl = np.array(k_mdl)
            k_aic = np.array(k_aic)
            cells[(snr_db, n_snap)] = dict(
                snr_db=snr_db, n_snap=n_snap,
                mdl_error_rate=float(np.mean(k_mdl != 2)),
                aic_error_rate=float(np.mean(k_aic != 2)),
                mdl_mean=float(np.mean(k_mdl)), aic_mean=float(np.mean(k_aic)),
                mdl_khat_hist={int(k): int(np.sum(k_mdl == k)) for k in sorted(set(k_mdl.tolist()))},
                aic_khat_hist={int(k): int(np.sum(k_aic == k)) for k in sorted(set(k_aic.tolist()))},
            )
    return cells


def b1_max_sd_cell(b1_cells, reduction=True):
    """The B1 grid cell (reduction-ON, per §S.6.2) with the largest observed
    signed-gap SD. Neither ROADMAP.md nor SPEC_experiment_B.md names a
    specific B1 cell as 'the pilot SD' — B1 as a whole serves pilot duty
    (SPEC §S.2-B1: '...the §8-ii pilot SD...'). Reported as the conservative
    alternative to whichever single matched cell is picked as a reference
    (see run_experiment_B.main's ceiling-effect finding)."""
    on_cells = {k: v for k, v in b1_cells.items() if k[0] is reduction}
    best_key = max(on_cells, key=lambda k: on_cells[k]["gap_sd"])
    return best_key, on_cells[best_key]


def b6a_nsnap_from_b3(b3_cells, snr_db=5.0):
    """§8-vii procedure: largest n_snap in {8,16,32,64,128} where B3's MDL
    K-hat error rate >= 20% at the given SNR (fallback 8). Computed and
    REPORTED here; B6a itself is NOT run (out of this handoff's scope)."""
    candidates = [n for n in NSNAP_AXIS_B3
                 if b3_cells[(snr_db, n)]["mdl_error_rate"] >= 0.20]
    return max(candidates) if candidates else 8


# --------------------------------------------------------------- mini-pilot

def run_minipilot():
    """50-trial mini-pilot at phi=90 deg / 5 dB / d=3, |rho|=0.85 (SLOT-1),
    both solvers, grid reduction ON (pinned §S.6.2 — the confirmatory
    setting; the §8-ii SD inputs are read from reduction-ON rows)."""
    tm = build_array_and_grid(row_normalisation=ROW_NORMALISATION)
    idx_a, idx_b = row_center_cols(MINIPILOT_D)
    true_cells = [np.unravel_index(idx_a, GRID_SHAPE),
                 np.unravel_index(idx_b, GRID_SHAPE)]
    rho = MINIPILOT_ABS_RHO * np.exp(1j * np.deg2rad(MINIPILOT_PHI_DEG))
    cfg = with_reduction(FAIRNESS_CONFIG, True)

    dr = {"l2": [], "gibf": [], "mmv": []}
    ps = {"l2": [], "gibf": [], "mmv": []}
    npk = {"l2": [], "gibf": [], "mmv": []}
    for trial in range(MINIPILOT_N_TRIALS):
        rng = _rng(4, "minipilot", trial)
        X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                                  n_snap=MINIPILOT_N_SNAP, snr_db=MINIPILOT_SNR_DB,
                                  coherence=rho)
        S = build_csm(X)
        lam, U = eigendecompose(S)
        V = eigenmodes(lam, U, 2)
        out = solve_all(tm.A, V, cfg)
        for name in ("l2", "gibf", "mmv"):
            I2d = out[name]["I"].reshape(GRID_SHAPE)
            dr[name].append(delta_r_bar(I2d, true_cells))
            ps[name].append(p_sep(I2d, true_cells))
            # instrumentation only (never adjudicating): how the peak-finder
            # behaves under refinement, at unchanged rel_thresh/max_peaks.
            npk[name].append(len(find_peaks_2d(I2d)))

    gap = [g - m for g, m in zip(dr["gibf"], dr["mmv"])]
    return dict(
        phi_deg=MINIPILOT_PHI_DEG, snr_db=MINIPILOT_SNR_DB, d=MINIPILOT_D,
        separation_km=round(MINIPILOT_D * POLE_SPACING_KM, 2),
        abs_rho=MINIPILOT_ABS_RHO, n_snap=MINIPILOT_N_SNAP,
        n_trials=MINIPILOT_N_TRIALS, reduction=True,
        true_cells=[[int(c[0]), int(c[1])] for c in true_cells],
        delta_r_bar={k: v for k, v in dr.items()},
        p_sep_rate={k: float(np.mean(v)) for k, v in ps.items()},
        delta_r_bar_mean={k: float(np.mean(v)) for k, v in dr.items()},
        delta_r_bar_sd={k: float(np.std(v, ddof=1)) for k, v in dr.items()},
        n_peaks_hist={m: {int(k): int(np.sum(np.array(v) == k))
                          for k in sorted(set(v))} for m, v in npk.items()},
        gap_signed=gap, gap_mean=float(np.mean(gap)), gap_sd=float(np.std(gap, ddof=1)),
    )


# ------------------------------------------------------------------- main

def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    geom = geometry_record()
    print("=== Geometry (REFINED 2026-07-27, ROADMAP §8) ===")
    print(f"  pole grid {geom['n_pole_lat']}x{geom['n_pole_lon']} = "
         f"{geom['n_poles']} poles, spacing {geom['pole_spacing_km']} km "
         f"(isotropic); station array unchanged "
         f"({geom['n_station_lat']}x{geom['n_station_lon']}, lon spacing "
         f"{geom['station_lon_spacing_km']} km)")
    print(f"  d axis in km: {geom['d_axis_km']}  |  confirmatory d=3 = "
         f"{geom['confirmatory_d_km']} km = 1 station spacing")
    print(f"  tau_r = 1 cell = {geom['tau_r_km']} km (was 106.6 km); "
         f"overcompleteness {geom['overcompleteness']}x")

    print("=== Single-source exact-recovery gate (SPEC S.6.4) ===")
    gate = single_source_gate()
    (RESULTS / "gate_result.json").write_text(json.dumps(gate, indent=2))
    if not gate["passed"]:
        print("GATE: FAIL —", gate["failures"])
        print("Experiment B HALTED (solver bug, not science). See "
              f"{RESULTS / 'gate_result.json'}")
        sys.exit(1)
    print("GATE: PASS")

    print("=== B1 (cost/pilot/floor) ===")
    b1_cells = run_b1()
    b1_serializable = {f"{r}|{d}|{s}": v for (r, d, s), v in b1_cells.items()}
    (RESULTS / "b1_results.json").write_text(json.dumps(b1_serializable, indent=2))
    pilot_cell = b1_cells[(True, 3, 5.0)]
    print(f"B1 matched cell (d=3, snr=5dB, reduction=ON, mirroring the "
         f"mini-pilot's d/coordinates): GIBF-MMV gap mean="
         f"{pilot_cell['gap_mean']:.4f} sd={pilot_cell['gap_sd']:.4f}")
    max_sd_key, max_sd_cell = b1_max_sd_cell(b1_cells, reduction=True)
    print(f"B1 max-SD cell (reduction=ON) across the whole grid: "
         f"d={max_sd_key[1]}, snr={max_sd_key[2]}dB -> gap sd="
         f"{max_sd_cell['gap_sd']:.4f} (mean={max_sd_cell['gap_mean']:.4f})")

    print("=== B3 (mode-selection surfaces) ===")
    b3_cells = run_b3()
    b3_serializable = {f"{s}|{n}": v for (s, n), v in b3_cells.items()}
    (RESULTS / "b3_results.json").write_text(json.dumps(b3_serializable, indent=2))
    b6a_nsnap = b6a_nsnap_from_b3(b3_cells, snr_db=5.0)
    mdl_err_5db = {n: b3_cells[(5.0, n)]["mdl_error_rate"] for n in NSNAP_AXIS_B3}
    print(f"MDL error rate at 5 dB across n_snap: {mdl_err_5db}")
    print(f"B6a n_snap candidate (from B3 MDL error >= 20% at 5dB, "
         f"fallback 8): {b6a_nsnap}")

    print("=== Mini-pilot (50 trials, phi=90, 5dB, d=3, reduction ON) ===")
    minipilot = run_minipilot()
    (RESULTS / "minipilot_results.json").write_text(json.dumps(minipilot, indent=2))
    print(f"Mini-pilot ({minipilot['separation_km']} km separation): "
         f"GIBF-MMV gap mean={minipilot['gap_mean']:.4f} "
         f"sd={minipilot['gap_sd']:.4f}")
    print(f"  per-method delta_r_bar mean (cells): "
         f"{ {k: round(v, 3) for k, v in minipilot['delta_r_bar_mean'].items()} }")
    print(f"  per-method P_sep: {minipilot['p_sep_rate']}  |  "
         f"detected-peak counts: {minipilot['n_peaks_hist']}")

    # Degeneracy diagnosis. A zero gap SD is the crude symptom; the condition
    # that actually matters is whether either PRE-REGISTERED rule can be
    # satisfied at this cell at ANY n_trials. A sparse solver pinned at
    # delta_r_bar == 0.0 on every trial makes the win rule's "GIBF >= 20%
    # lower" unsatisfiable (20% below zero is negative) and the mirrored loss
    # rule unreachable (a degenerate [0, 0] CI can never fail to overlap), and
    # P_sep == 1.00 for both sparse solvers kills the second win-rule arm too
    # -- all of which is true even when a stray trial makes the SD nonzero.
    # Diagnose by the per-method arrays, never by the gap SD alone.
    grid_diag = float(np.hypot(GRID_SHAPE[0] - 1, GRID_SHAPE[1] - 1))
    mp_mean = minipilot["delta_r_bar_mean"]
    mp_all = minipilot["delta_r_bar"]
    sparse_at_ceiling = [m for m in ("gibf", "mmv")
                         if all(v == 0.0 for v in mp_all[m])]
    sparse_at_floor = [m for m in ("gibf", "mmv")
                       if all(abs(v - grid_diag) < 1e-9 for v in mp_all[m])]
    at_ceiling = len(sparse_at_ceiling) > 0
    degenerate = minipilot["gap_sd"] == 0.0
    if at_ceiling or sparse_at_floor or degenerate:
        if at_ceiling:
            mode = (f"CEILING — {'/'.join(sparse_at_ceiling)} recover(s) both "
                    "sources exactly (delta_r_bar = 0.000) on EVERY trial, and "
                    f"P_sep = {minipilot['p_sep_rate']}")
        elif sparse_at_floor:
            mode = (f"FLOOR — {'/'.join(sparse_at_floor)} pinned at the "
                    f"unmatched-source penalty (grid diagonal = "
                    f"{grid_diag:.3f} cells) on every trial")
        else:
            mode = "TIED but at neither extreme — inspect the per-trial arrays"
        print(f"\nFINDING: {mode}. The pre-registered win rule (GIBF Δr̄ >= 20% "
             "lower; or smallest d with P_sep >= 0.8 at least 1 cell smaller) "
             "and the mirrored loss rule CANNOT be satisfied at this cell at "
             "any n_trials while a sparse solver sits at the ceiling — a "
             "nonzero gap SD makes the VARIANCE INPUT usable, not the CELL "
             "adjudicable. Reported as a design finding, NOT repaired by "
             "further parameter changes (stop rule).")

    variance_input_matched = max(pilot_cell["gap_sd"], minipilot["gap_sd"])
    variance_input_conservative = max(max_sd_cell["gap_sd"], minipilot["gap_sd"])
    print(f"\n§8-ii power-calc variance inputs (REPORTED ONLY — n_trials is "
         f"Strider's sign-off and is NOT computed here):")
    print(f"  matched-cell reading = max(B1 d=3/5dB SD, mini-pilot SD) = "
         f"max({pilot_cell['gap_sd']:.4f}, {minipilot['gap_sd']:.4f}) = "
         f"{variance_input_matched:.4f}")
    print(f"  conservative reading = max(B1 grid-max SD [d={max_sd_key[1]}, "
         f"{max_sd_key[2]}dB], mini-pilot SD) = max({max_sd_cell['gap_sd']:.4f}, "
         f"{minipilot['gap_sd']:.4f}) = {variance_input_conservative:.4f}")
    print("  UNITS: all gap SDs are in GRID CELLS; one cell is now "
         f"{POLE_SPACING_KM:.2f} km (was 106.6 km before the 2026-07-27 "
         "refinement). tau_r = 1 cell and the win rule's '>= 1 grid cell' "
         "are pinned in cells and unchanged; their physical meaning moved.")
    print("STOP — powered B2/B6 wait on Strider's §8-ii power-calc sign-off; "
         "n_trials itself is NOT computed here.")

    script = Path(__file__).resolve()
    manifest = dict(
        experiment="Experiment B kickoff: solver stack + B1 + B3 + mini-pilot",
        seed=SEED, numpy=np.__version__,
        git_commit=_git_commit(),
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        preregistration="handoff.md (2026-07-12, assignee corrected 2026-07-20); "
                        "SPEC_experiment_B.md S.1/S.2/S.4/S.6/S.10; "
                        "ROADMAP.md S8-ii...viii + the 2026-07-27 entry",
        geometry=geom,
        geometry_supersedes=("the 2026-07-21 kickoff run's inherited "
                             "config/base.yaml geometry (11x11 poles, 106.6 km "
                             "lon cell, fixed col_a=1), whose d=3 was a 320 km "
                             "separation and returned a degenerate zero-variance "
                             "confirmatory cell; see ROADMAP §8 2026-07-27"),
        unit_note=("all delta_r_bar / gap / tau_r quantities are in GRID CELLS; "
                   f"one cell = {POLE_SPACING_KM:.3f} km after the 2026-07-27 "
                   "refinement (was 106.6 km along lon, 200.4 km along lat). "
                   "Every pinned numeral (tau_r=1, '>= 1 grid cell', 20%) is "
                   "unchanged; its physical meaning moved with the cell."),
        eig_floor_units=("SPEC §S.4 floor 1e-18 re-expressed in "
                         "TRACE-NORMALIZED units (lam * 2/trace) inside "
                         "modeselect.py, 2026-07-27 — numeral unchanged, "
                         "units fixed; estimate_n_sources is now scale-free "
                         "and pytest-pinned across 48 decades"),
        n_trials_b1_b3=N_TRIALS_B1,
        row_normalisation=ROW_NORMALISATION,
        fairness_config=dict(eps_frac=FAIRNESS_CONFIG.eps_frac,
                             p_norm=FAIRNESS_CONFIG.p_norm,
                             weight_floor=FAIRNESS_CONFIG.weight_floor,
                             beta=FAIRNESS_CONFIG.beta,
                             max_iter=FAIRNESS_CONFIG.max_iter,
                             tol=FAIRNESS_CONFIG.tol,
                             min_active_factor=FAIRNESS_CONFIG.min_active_factor),
        gate_passed=gate["passed"],
        b1_matched_cell=dict(d=3, snr_db=5.0, reduction=True,
                             gap_mean=pilot_cell["gap_mean"], gap_sd=pilot_cell["gap_sd"]),
        b1_max_sd_cell=dict(d=int(max_sd_key[1]), snr_db=float(max_sd_key[2]), reduction=True,
                            gap_mean=max_sd_cell["gap_mean"], gap_sd=max_sd_cell["gap_sd"]),
        b6a_nsnap_candidate=b6a_nsnap,
        b3_mdl_error_rate_5db=mdl_err_5db,
        minipilot=dict(gap_mean=minipilot["gap_mean"], gap_sd=minipilot["gap_sd"],
                      n_trials=minipilot["n_trials"],
                      separation_km=minipilot["separation_km"],
                      delta_r_bar_mean=minipilot["delta_r_bar_mean"],
                      p_sep_rate=minipilot["p_sep_rate"],
                      n_peaks_hist=minipilot["n_peaks_hist"]),
        confirmatory_cell_gap_sd_is_zero=bool(degenerate),
        confirmatory_cell_at_ceiling=bool(at_ceiling),
        confirmatory_cell_ceiling_methods=sparse_at_ceiling,
        confirmatory_cell_adjudicable=not (at_ceiling or bool(sparse_at_floor)),
        confirmatory_cell_note=(
            "A sparse solver pinned at delta_r_bar = 0.000 on every trial makes "
            "the pre-registered win rule ('GIBF Δr̄ >= 20% lower' — 20% below "
            "zero is negative) and the mirrored loss rule (a degenerate [0, 0] "
            "CI can never fail to overlap) unsatisfiable at this cell at ANY "
            "n_trials, and P_sep = 1.00 for both sparse solvers closes the "
            "second win-rule arm as well. A nonzero gap SD makes the §8-ii "
            "VARIANCE INPUT usable; it does not make the CELL adjudicable. "
            "Read `confirmatory_cell_adjudicable`, not the SD, before sizing "
            "a confirmatory run."),
        power_calc_variance_input_matched_reading=variance_input_matched,
        power_calc_variance_input_conservative_reading=variance_input_conservative,
        power_calc_status="PENDING Strider's S8-ii sign-off (incl. which SD "
                          "reading to adopt); n_trials NOT yet computed/run; "
                          "powered B2/B6 NOT executed",
        runtime_s=round(time.time() - t0, 1),
    )
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest + results to {RESULTS}")


if __name__ == "__main__":
    main()
